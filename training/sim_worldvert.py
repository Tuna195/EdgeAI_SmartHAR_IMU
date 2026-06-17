"""
sim_worldvert.py  --  THU NGHIEM rep-counter RIENG cho shoulder_press (world-vertical)
======================================================================================
SANDBOX doc lap: CHI doc data/shoulder_press. KHONG dung rep_config.py, sim_peakdet.py
hay config cua bicep/lateral/tricep. Cac bai kia van chay peakdet nhu cu.

Y tuong (khac han peakdet tren 1 truc):
  1. Uoc luong huong trong luc g_hat bang COMPLEMENTARY FILTER:
       - gyro xoay g_hat (nhanh, bam tay nghieng, mu voi gia toc thang)
       - accel keo nhe g_hat ve (cham, chong troi gyro)
  2. a_lin  = a - g_hat                 (bo trong luc)
  3. a_up   = -dot(a_lin, g_hat_unit)   (gia toc theo phuong DUNG that; + = len)
  4. v_up   = leaky_integrate(a_up)     (van toc dung, leaky chong troi)
  5. Dem rep bang HYSTERESIS tren v_up  (vuot +thr roi tut -thr = 1 rep)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

FS = 50.0
DT = 1.0 / FS


def load_raw(path):
    df = pd.read_csv(path)
    a = df[['ax', 'ay', 'az']].values.astype(np.float64)
    g = df[['gx', 'gy', 'gz']].values.astype(np.float64)
    return a, g


def gravity_complementary(a, gyro, alpha=0.02, gyro_in_deg=False):
    """g_hat[N,3]: huong trong luc trong he truc thiet bi (gyro xoay + accel keo)."""
    n = len(a)
    if gyro_in_deg:
        gyro = gyro * (np.pi / 180.0)
    g_hat = np.empty((n, 3))
    gh = a[:min(10, n)].mean(axis=0).copy()
    for i in range(n):
        w = gyro[i]
        gh = gh - np.cross(w, gh) * DT      # gyro xoay: dg/dt = -(w x g)
        gh = (1 - alpha) * gh + alpha * a[i]  # accel keo nhe ve
        g_hat[i] = gh
    return g_hat


def gravity_lowpass(a, alpha=0.02):
    """So sanh: chi low-pass accel, KHONG dung gyro."""
    n = len(a)
    g_hat = np.empty((n, 3))
    gh = a[:min(10, n)].mean(axis=0).copy()
    for i in range(n):
        gh = (1 - alpha) * gh + alpha * a[i]
        g_hat[i] = gh
    return g_hat


def vertical_velocity(a, g_hat, leak=0.985):
    n = len(a)
    a_up = np.empty(n)
    v_up = np.empty(n)
    v = 0.0
    for i in range(n):
        gh = g_hat[i]
        gh_u = gh / (np.linalg.norm(gh) + 1e-9)
        au = -np.dot(a[i] - gh, gh_u)       # + = di len (nguoc trong luc)
        v = leak * v + au * DT
        a_up[i] = au
        v_up[i] = v
    return a_up, v_up


def count_hysteresis(v, thr, min_gap_samp):
    """Vuot +thr (cam ket len) -> tut -thr (cam ket xuong) = 1 rep."""
    reps, markers = 0, []
    last = -10**9
    armed_up = False
    for i, x in enumerate(v):
        if x > thr:
            armed_up = True
        if armed_up and x < -thr:
            if i - last >= min_gap_samp:
                reps += 1
                markers.append(i)
                last = i
            armed_up = False
    return reps, markers


def measure_travel(v, peak_thr):
    """Cat v_up thanh cac 'run' cung dau; voi run co dinh vuot peak_thr,
    tinh dien tich = |∫v dt| = quang duong nua nhip (don vi: (g.s).s, leaky lam
    no NHO hon met that, nhung dung de SO SANH that-vs-gian)."""
    n = len(v)
    ups, dns = [], []
    i = 0
    while i < n:
        s = 1 if v[i] > 0 else (-1 if v[i] < 0 else 0)
        if s == 0:
            i += 1
            continue
        j = i
        area = 0.0
        peak = 0.0
        while j < n and (1 if v[j] > 0 else (-1 if v[j] < 0 else 0)) == s:
            area += v[j] * DT
            peak = max(peak, abs(v[j]))
            j += 1
        if peak >= peak_thr:
            (ups if s > 0 else dns).append(abs(area))
        i = j
    return np.array(ups), np.array(dns)


def analyze(path, alpha, leak, gyro_in_deg, label):
    a, g = load_raw(path)
    n = len(a)
    rest = a[:min(20, n)].mean(axis=0)
    gmax = np.abs(g).max()
    print(f"\n--- {label}: {path.name}  ({n} mau, {n/FS:.1f}s)")
    print(f"    accel nghi: ax={rest[0]:+.3f} ay={rest[1]:+.3f} az={rest[2]:+.3f}"
          f"  |g|={np.linalg.norm(rest):.3f}")
    print(f"    gyro |max|={gmax:.2f}  -> coi nhu {'deg/s' if gyro_in_deg else 'rad/s'}")

    g_cf = gravity_complementary(a, g, alpha=alpha, gyro_in_deg=gyro_in_deg)
    g_lp = gravity_lowpass(a, alpha=alpha)
    au_cf, v_cf = vertical_velocity(a, g_cf, leak=leak)
    au_lp, v_lp = vertical_velocity(a, g_lp, leak=leak)

    cos = np.sum(g_cf * g_lp, axis=1) / (
        np.linalg.norm(g_cf, axis=1) * np.linalg.norm(g_lp, axis=1) + 1e-9)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    print(f"    v_up(CF) std={v_cf.std():.4f}  range=[{v_cf.min():+.3f},{v_cf.max():+.3f}]")
    print(f"    goc lech CF-vs-LP: mean={ang.mean():.1f} max={ang.max():.1f} deg"
          f"  (lon => tay lac nhanh, can gyro)")
    print(f"    {'thr':>8} | reps(CF) reps(LP)")
    for k in (0.4, 0.6, 0.8, 1.0, 1.2):
        thr = k * v_cf.std()
        r_cf, _ = count_hysteresis(v_cf, thr, int(0.5 * FS))
        r_lp, _ = count_hysteresis(v_lp, thr, int(0.5 * FS))
        print(f"    {k:.1f}*std | {r_cf:6d}  {r_lp:6d}   (thr={thr:.3f})")
    return dict(a=a, g_cf=g_cf, g_lp=g_lp, v_cf=v_cf, v_lp=v_lp, au_cf=au_cf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data/shoulder_press')
    ap.add_argument('--alpha', type=float, default=0.02)
    ap.add_argument('--leak', type=float, default=0.985)
    ap.add_argument('--gyro_deg', action='store_true')
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()

    folder = Path(args.dir)
    files = sorted(folder.glob('*.csv'))
    if not files:
        print(f"khong thay csv trong {folder}")
        return
    picks = []
    if len(files) > 0:
        picks.append(('TRAI', files[0]))
    if len(files) > 5:
        picks.append(('PHAI', files[5]))

    res = {}
    for label, f in picks:
        res[label] = analyze(f, args.alpha, args.leak, args.gyro_deg, label)

    if args.plot and res:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        keys = list(res.keys())
        fig, axs = plt.subplots(len(keys), 2, figsize=(15, 4 * len(keys)), squeeze=False)
        for r, k in enumerate(keys):
            d = res[k]
            t = np.arange(len(d['a'])) / FS
            ax0 = axs[r][0]
            ax0.plot(t, d['a'][:, 0], lw=0.5, label='ax', color='#bbb')
            ax0.plot(t, d['a'][:, 1], lw=0.5, label='ay', color='#fbb')
            ax0.plot(t, d['a'][:, 2], lw=0.5, label='az', color='#bbf')
            ax0.plot(t, d['au_cf'], lw=0.9, label='a_up (CF)', color='k')
            ax0.set_title(f"{k}: accel raw + gia toc dung a_up")
            ax0.legend(fontsize=7, loc='upper right')
            ax1 = axs[r][1]
            ax1.plot(t, d['v_lp'], lw=0.8, label='v_up (low-pass)', color='#e67')
            ax1.plot(t, d['v_cf'], lw=1.1, label='v_up (complementary)', color='#1a7')
            ax1.axhline(0, color='k', lw=0.4)
            thr = 0.6 * d['v_cf'].std()
            ax1.axhline(thr, color='g', ls='--', lw=0.5)
            ax1.axhline(-thr, color='g', ls='--', lw=0.5)
            ax1.set_title(f"{k}: van toc dung v_up  (sin sach = thang)")
            ax1.legend(fontsize=7, loc='upper right')
        fig.tight_layout()
        out = folder.parent.parent / 'model_output' / 'sim_worldvert.png'
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=110)
        print(f"\n[plot] luu: {out}")


if __name__ == '__main__':
    main()
