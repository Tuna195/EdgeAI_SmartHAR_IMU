"""
sim_peakdet.py  --  Mo phong REP-COUNTER bang PEAKDET (delta) — OFFLINE
========================================================================
Thuat toan peakdet (Billauer): bam day chay (last_min) va dinh chay (last_max);
khi tin hieu roi day/dinh xa hon DELTA -> chot day/dinh do. 1 valley + 1 peak =
1 rep (khoa pha: cuc dau dinh E1, cham cuc doi E2 = 1 rep). BAT BIEN offset DC.

Pipeline (KHOP du dinh firmware, CAUSAL):
  CSV raw -> Z-score (HAR_MEAN/INV_STD) -> MA causal (smooth_win) -> peakdet(delta)

Config doc tu rep_config.py (REP_CONFIG). Them bai moi = sua rep_config.py.

Dung:
  python sim_peakdet.py --data_dir data                  # dem rep moi file + idle FP
  python sim_peakdet.py --data_dir data --plot bicep_curl # ve 1 bai -> PNG
  python sim_peakdet.py --data_dir data --plot all        # ve het
  # Tune nhanh 1 bai (quet delta), in rep/file:
  python sim_peakdet.py --data_dir data --sweep bicep_curl --sweep_delta 0.6,0.8,1.0,1.2
  # Override delta 1 lan (khong sua config):
  python sim_peakdet.py --data_dir data --plot bicep_curl --delta 0.9
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rep_config import AXES, FS, HAR_MEAN, HAR_INV_STD, REP_CONFIG

_MEAN = np.asarray(HAR_MEAN, dtype=np.float64)
_INV_STD = np.asarray(HAR_INV_STD, dtype=np.float64)


# ═════════════════════════════════════════════════════════════
# Z-score + load
# ═════════════════════════════════════════════════════════════
def zscore(raw):
    return (raw - _MEAN) * _INV_STD


def load_csv_z(path):
    raw = pd.read_csv(path)[AXES].values.astype(np.float64)
    return zscore(raw)


def load_files(folder):
    out = []
    for f in sorted(Path(folder).glob('*.csv')):
        try:
            z = load_csv_z(f)
        except Exception:
            continue
        if len(z) >= 100:
            out.append((f.name, z))
    return out


# ═════════════════════════════════════════════════════════════
# Smoothing — MA causal (mirror firmware: ringbuffer, tre (win-1)/2 mau)
# ═════════════════════════════════════════════════════════════
def smooth_causal(sig, win):
    if win is None or win <= 1:
        return np.asarray(sig, dtype=np.float64)
    sig = np.asarray(sig, dtype=np.float64)
    out = np.empty_like(sig)
    s = 0.0
    buf = np.zeros(win, dtype=np.float64)
    head = 0
    cnt = 0
    for i, x in enumerate(sig):
        if cnt < win:
            buf[cnt] = x
            s += x
            cnt += 1
        else:
            s -= buf[head]
            buf[head] = x
            s += x
            head = (head + 1) % win
        out[i] = s / cnt
    return out


# ═════════════════════════════════════════════════════════════
# PEAKDET — bam day/dinh chay, chot khi roi xa > delta. Auto-detect pha dau.
#   Tra ve list (type, pos): type = +1 (peak/max) hoac -1 (valley/min).
# ═════════════════════════════════════════════════════════════
def peakdet(sig, delta, min_abs=0.0):
    """min_abs: cuc tri phai vuot +-min_abs (z-units) moi duoc chot -> loc idle
    bien do nho. 0 = tat (peakdet thuan, bat bien offset nhung de FP idle)."""
    n = len(sig)
    events = []
    mn, mx = np.inf, -np.inf
    mnpos = mxpos = 0
    lookformax = None   # None = chua biet chieu dau; True = cho peak; False = cho valley
    for i in range(n):
        x = sig[i]
        if x > mx:
            mx, mxpos = x, i
        if x < mn:
            mn, mnpos = x, i

        if lookformax is None:
            # Quyet dinh chieu dau khi da roi 1 cuc xa hon delta
            if x > mn + delta:
                lookformax = True            # dang di len -> cuc se chot la PEAK
                mxpos = i; mx = x
            elif x < mx - delta:
                lookformax = False           # dang di xuong -> cuc se chot la VALLEY
                mnpos = i; mn = x
            continue

        if lookformax:
            if x < mx - delta:               # da roi DINH du xa -> chot peak
                if min_abs <= 0.0 or mx >= min_abs:   # min_abs=0 -> TAT cong
                    events.append((+1, mxpos))
                mn, mnpos = x, i
                lookformax = False
        else:
            if x > mn + delta:               # da roi DAY du xa -> chot valley
                if min_abs <= 0.0 or mn <= -min_abs:  # min_abs=0 -> TAT cong
                    events.append((-1, mnpos))
                mx, mxpos = x, i
                lookformax = True
    return events


# ═════════════════════════════════════════════════════════════
# Dem rep tu chuoi su kien peakdet (khoa pha + debounce + reset timeout)
#   1 rep = E1 (cuc dau) roi cham cuc DOI E2. Reset pha neu nghi qua lau.
# ═════════════════════════════════════════════════════════════
def count_reps(z, cfg):
    axis = cfg['axis']
    delta = cfg['delta']
    min_gap = int(round(cfg['min_gap_ms'] / 1000.0 * FS))
    reset_to = int(round(cfg.get('reset_timeout_ms', 3000) / 1000.0 * FS))

    sig = smooth_causal(z[:, axis], cfg.get('smooth_win', 1))
    events = peakdet(sig, delta, cfg.get('min_abs', 0.0))

    reps = 0
    markers = []
    e1_type = 0       # 0 = chua dinh pha
    saw_e1 = False
    last_rep = None
    last_ev = None
    for typ, pos in events:
        # Reset pha neu cach su kien truoc qua lau (vd nghi giua set)
        if last_ev is not None and (pos - last_ev) > reset_to:
            e1_type = 0
            saw_e1 = False
        last_ev = pos

        if e1_type == 0:
            e1_type = typ
            saw_e1 = True
            continue
        if saw_e1:
            if typ != e1_type:                       # cham cuc doi = xong 1 chu ky
                saw_e1 = False
                if last_rep is None or (pos - last_rep) >= min_gap:
                    reps += 1
                    last_rep = pos
                    markers.append(pos)
        else:
            if typ == e1_type:                       # quay lai E1 -> mo rep ke
                saw_e1 = True
    return reps, markers, sig, events


# ═════════════════════════════════════════════════════════════
# Reporting
# ═════════════════════════════════════════════════════════════
def run_counts(data_dir):
    data_dir = Path(data_dir)
    print("\n" + "=" * 72)
    print("  DEM REP / FILE  (PEAKDET delta)   TRAI=01-05, PHAI=06-10")
    print("=" * 72)
    for ex, cfg in REP_CONFIG.items():
        folder = data_dir / ex
        if not folder.exists():
            continue
        print(f"\n[{ex}]  axis={AXES[cfg['axis']]}  delta={cfg['delta']}  "
              f"min_gap={cfg['min_gap_ms']}ms  smooth={cfg['smooth_win']}")
        for idx, (fname, z) in enumerate(load_files(folder)):
            grp = "TRAI" if idx < 5 else "PHAI"
            reps, _, _, _ = count_reps(z, cfg)
            print(f"   [{grp}] {fname:24s} dur={len(z)/FS:6.1f}s  reps={reps:3d}")

    idle = data_dir / 'idle'
    if idle.exists():
        print(f"\n[idle]  (false-positive — moi file chay het {len(REP_CONFIG)} detector)")
        for fname, z in load_files(idle):
            fp = max(count_reps(z, c)[0] for c in REP_CONFIG.values())
            flag = "  <-- FP!" if fp > 0 else "  OK"
            print(f"   {fname:24s} dur={len(z)/FS:6.1f}s  max_fp={fp}{flag}")


def sweep_delta(data_dir, ex, deltas):
    folder = Path(data_dir) / ex
    files = load_files(folder)
    base = dict(REP_CONFIG[ex])
    print(f"\n[SWEEP delta] {ex}  axis={AXES[base['axis']]}  files={len(files)}")
    header = "   delta  " + "  ".join(f"{f[0].split('_')[-1].replace('.csv',''):>4}" for f in files)
    print(header)
    for d in deltas:
        cfg = dict(base, delta=d)
        counts = [count_reps(z, cfg)[0] for _, z in files]
        row = "  ".join(f"{c:>4d}" for c in counts)
        print(f"   {d:5.2f}  {row}")


def plot_exercise(data_dir, ex, out_dir, delta_override=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg = dict(REP_CONFIG[ex])
    if delta_override is not None:
        cfg['delta'] = delta_override
    files = load_files(Path(data_dir) / ex)
    if not files:
        print(f"  (khong co data cho {ex})")
        return
    pick = [0] + ([5] if len(files) > 5 else [])
    pick = [p for p in pick if p < len(files)]
    n = len(pick)
    fig, axs = plt.subplots(n, 1, figsize=(14, 3.4 * n), squeeze=False)
    for row, k in enumerate(pick):
        fname, z = files[k]
        reps, markers, sig, events = count_reps(z, cfg)
        t = np.arange(len(sig)) / FS
        ax = axs[row][0]
        ax.plot(t, z[:, cfg['axis']], color='#cccccc', lw=0.6, label=f"z[{AXES[cfg['axis']]}]")
        ax.plot(t, sig, color='#1f77b4', lw=0.9, label=f"smooth(win={cfg['smooth_win']})")
        for typ, pos in events:
            ax.plot(pos / FS, sig[pos], 'r^' if typ > 0 else 'gv', ms=5)
        for m in markers:
            ax.axvline(m / FS, color='green', lw=0.8, alpha=0.5)
        grp = "TRAI" if k < 5 else "PHAI"
        ax.set_title(f"[{grp}] {fname}  |  REPS = {reps}  "
                     f"(axis={AXES[cfg['axis']]}, delta={cfg['delta']}, gap={cfg['min_gap_ms']}ms)")
        ax.set_xlabel('s')
        ax.legend(loc='upper right', fontsize=8)
    fig.suptitle(f"{ex}  — PEAKDET (delta)   ^=peak  v=valley", fontsize=12)
    fig.tight_layout()
    out = Path(out_dir) / f"sim_peakdet_{ex}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[plot] da luu: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--plot', default=None, help='ten bai / "all"')
    ap.add_argument('--out_dir', default='model_output')
    ap.add_argument('--delta', type=float, default=None, help='override delta (chi cho --plot)')
    ap.add_argument('--sweep', default=None, help='ten bai de quet delta')
    ap.add_argument('--sweep_delta', default='0.6,0.8,1.0,1.2,1.5',
                    help='danh sach delta, ngan cach bang dau phay')
    ap.add_argument('--min_abs', type=float, default=None,
                    help='override min_abs cho TAT CA bai (test loc idle)')
    ap.add_argument('--axis', type=int, default=None,
                    help='override axis (0..5) cho bai dang --sweep/--plot (test truc)')
    args = ap.parse_args()

    if args.min_abs is not None:
        for c in REP_CONFIG.values():
            c['min_abs'] = args.min_abs
    if args.axis is not None:
        tgt = args.sweep or args.plot
        if tgt in REP_CONFIG:
            REP_CONFIG[tgt]['axis'] = args.axis

    if args.sweep:
        deltas = [float(x) for x in args.sweep_delta.split(',')]
        sweep_delta(args.data_dir, args.sweep, deltas)
    elif args.plot == 'all':
        for ex in REP_CONFIG:
            plot_exercise(args.data_dir, ex, args.out_dir, args.delta)
    elif args.plot:
        plot_exercise(args.data_dir, args.plot, args.out_dir, args.delta)
    else:
        run_counts(args.data_dir)


if __name__ == '__main__':
    main()
