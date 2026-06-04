"""
sim_fixed.py -- Mo phong detector TOI GIAN (Schmitt nguong CO DINH)
====================================================================
Mirror Y HET src/rep_detector.h: Schmitt 2 nguong co dinh tren z-score 1 truc gyro
+ debounce min_gap. KHONG envelope/peak-hold/adaptive.

Doc CSV raw -> Z-score (HAR_MEAN/INV_STD, KHOP firmware) -> detector -> dem rep.

Dung:
  python sim_fixed.py --data_dir data                 # dem rep moi file + idle FP
  python sim_fixed.py --data_dir data --plot bicep_curl   # ve bieu do 1 bai -> PNG
  python sim_fixed.py --data_dir data --plot all          # ve het 4 bai
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ─── COPY tu src/norm_params.h (thu tu ax,ay,az,gx,gy,gz) ─────
HAR_MEAN = np.array([0.15883821, 0.01432643, -0.02781012,
                     -0.44101688, 0.47294450, -0.21489604], dtype=np.float64)
HAR_INV_STD = np.array([1.48762882, 1.90575035, 1.96947049,
                        0.04014855, 0.02478677, 0.02976099], dtype=np.float64)
AXES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
FS = 50.0

# ─── Cau hinh per-bai — KHOP src/main.cpp EXCFG ──────────────
#   (axis, T, min_gap_ms);  gx=3, gy=4, gz=5
#   T = min(T_trai, T_phai) suy tu data (analyze_axes.py)
EXCFG = {
    'bicep_curl':     (4, 0.5, 2000),
    'lateral_raise':  (4, 0.8,  850),
    'shoulder_press': (3, 0.6, 1910),
    'tricep_ext':     (5, 1.1,  970),
}


def zscore(raw):
    return (raw - HAR_MEAN) * HAR_INV_STD


def load_csv_z(path):
    df = pd.read_csv(path)
    raw = df[AXES].values.astype(np.float64)
    return zscore(raw)


# ─── Detector: mirror 1:1 rep_detector.h ─────────────────────
def count_reps(z, axis, T, min_gap_ms):
    """Mirror rep_detector.h (KHOA-PHA): cuc dau dinh E1, cham cuc doi E2 = 1 rep."""
    gap = int(round(min_gap_ms / 1000.0 * FS))
    e1 = 0          # 0 chua dinh pha; +1 neu E1=+T; -1 neu E1=-T
    saw = False     # dang trong 1 rep (da cham E1, cho E2)?
    last = None
    reps = 0
    markers = []
    sig = z[:, axis]
    for i in range(len(sig)):
        hi = sig[i] > T
        lo = sig[i] < -T
        if e1 == 0:
            if hi:
                e1, saw = 1, True
            elif lo:
                e1, saw = -1, True
            continue
        if saw:
            e2 = lo if e1 > 0 else hi
            if e2:
                saw = False
                if last is None or (i - last) >= gap:
                    reps += 1
                    last = i
                    markers.append(i)
        else:
            if (hi if e1 > 0 else lo):
                saw = True
    return reps, markers


def load_files(folder):
    files = sorted(Path(folder).glob('*.csv'))
    out = []
    for f in files:
        try:
            z = load_csv_z(f)
        except Exception:
            continue
        if len(z) >= 100:
            out.append((f.name, z))
    return out


def run_counts(data_dir):
    data_dir = Path(data_dir)
    print("\n" + "=" * 70)
    print("  DEM REP / FILE  (Schmitt co dinh)  TRAI=01-05, PHAI=06-10")
    print("=" * 70)
    for ex, (axis, T, gap) in EXCFG.items():
        folder = data_dir / ex
        if not folder.exists():
            continue
        print(f"\n[{ex}]  axis={AXES[axis]}  T={T}  min_gap={gap}ms")
        for idx, (fname, z) in enumerate(load_files(folder)):
            grp = "TRAI" if idx < 5 else "PHAI"
            reps, _ = count_reps(z, axis, T, gap)
            print(f"   [{grp}] {fname:24s} dur={len(z)/FS:6.1f}s  reps={reps:3d}")

    # idle false positives: chay TAT CA detector tren idle
    idle = data_dir / 'idle'
    if idle.exists():
        print(f"\n[idle]  (false-positive check — moi file chay het 4 detector)")
        for fname, z in load_files(idle):
            fp = max(count_reps(z, a, T, g)[0] for (a, T, g) in EXCFG.values())
            flag = "  <-- FP!" if fp > 0 else "  OK"
            print(f"   {fname:24s} dur={len(z)/FS:6.1f}s  max_fp={fp}{flag}")


def plot_exercise(data_dir, ex, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    axis, T, gap = EXCFG[ex]
    files = load_files(Path(data_dir) / ex)
    if not files:
        print(f"  (khong co data cho {ex})")
        return
    # 1 file TRAI (idx 0) + 1 file PHAI (idx 5 neu co)
    pick = [0] + ([5] if len(files) > 5 else [])
    pick = [p for p in pick if p < len(files)]
    n = len(pick)
    fig, axs = plt.subplots(n, 1, figsize=(14, 3.4 * n), squeeze=False)
    for row, k in enumerate(pick):
        fname, z = files[k]
        reps, markers = count_reps(z, axis, T, gap)
        t = np.arange(len(z)) / FS
        ax = axs[row][0]
        ax.plot(t, z[:, axis], color='#1f77b4', lw=0.8, label=f'z[{AXES[axis]}]')
        ax.axhline(T,  color='red',    ls='--', lw=0.9, label=f'+T={T}')
        ax.axhline(-T, color='orange', ls='--', lw=0.9, label=f'-T={-T}')
        for m in markers:
            ax.axvline(m / FS, color='green', lw=0.8, alpha=0.6)
        grp = "TRAI" if k < 5 else "PHAI"
        ax.set_title(f"[{grp}] {fname}  |  REPS = {reps}  (axis={AXES[axis]}, T={T}, gap={gap}ms)")
        ax.set_xlabel('s')
        ax.legend(loc='upper right', fontsize=8)
    fig.suptitle(f"{ex}  — Schmitt nguong CO DINH", fontsize=12)
    fig.tight_layout()
    out = Path(out_dir) / f"sim_fixed_{ex}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[plot] da luu: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--plot', default=None, help='ten bai / "all" de ve PNG')
    ap.add_argument('--out_dir', default='model_output')
    args = ap.parse_args()

    if args.plot == 'all':
        for ex in EXCFG:
            plot_exercise(args.data_dir, ex, args.out_dir)
    elif args.plot:
        plot_exercise(args.data_dir, args.plot, args.out_dir)
    else:
        run_counts(args.data_dir)


if __name__ == '__main__':
    main()
