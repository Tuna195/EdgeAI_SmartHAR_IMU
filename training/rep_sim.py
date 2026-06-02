"""
rep_sim.py  --  Offline Rep-Counter Simulator (NGUON CHAN LY)
=================================================================
Mo phong Y HET thuat toan EnvelopePeakDetector + RepValidator cua firmware
(src/rep_counter.cpp) de tinh chinh rep counter ma KHONG can cam board.

Quan trong (fidelity):
  - CSV trong training/data/** la RAW (saved/IMU_sample.cpp ghi thang imu.data.*).
  - Firmware (sau khi BO EMA front-end) dua thang raw -> Z-score -> detector.
  - => Tool nay doc CSV raw -> Z-score (HAR_MEAN/INV_STD) -> detector. KHONG filter.

File nay vua la cong cu, vua la module dung chung:
  - phase0_find_config.py import compute_envelope()/autocorr_score() tu day
    de Phase 0 va detector dung CHUNG 1 thuat toan envelope.

Cach dung:
  # Phan tich + derive config + dem rep tat ca exercise:
  python rep_sim.py --data_dir data

  # Ve chi tiet 1 file (signal/envelope/threshold/peak) -> PNG:
  python rep_sim.py --data_dir data --plot bicep_curl

  # Quet DC-EMA alpha de chon gia tri tot nhat:
  python rep_sim.py --data_dir data --sweep_alpha
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# NORM PARAMS -- COPY tu src/norm_params.h (thu tu: ax,ay,az,gx,gy,gz)
# ─────────────────────────────────────────────────────────────
HAR_MEAN = np.array([
     0.15883821,  # ax
     0.01432643,  # ay
    -0.02781012,  # az
    -0.44101688,  # gx
     0.47294450,  # gy
    -0.21489604,  # gz
], dtype=np.float64)

HAR_INV_STD = np.array([
    1.48762882,  # ax
    1.90575035,  # ay
    1.96947049,  # az
    0.04014855,  # gx
    0.02478677,  # gy
    0.02976099,  # gz
], dtype=np.float64)

AXES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
FS = 50.0  # Hz

_MIN_GAP_OVERRIDE = 0  # 0 = auto theo period; >0 = ep min_gap_ms (test)

# ─── Tuning defaults (khop voi rep_counter.h sau khi sua) ─────
PD_MA_WIN_DEFAULT         = 25       # Moving avg = 0.5s @ 50Hz
PD_MEAN_EMA_ALPHA_DEFAULT = 0.0067   # DC removal cham: tau~3s, cutoff~0.05Hz
                                     # (cu 0.05 -> tau 0.4s, an mat dai rep 0.25-0.5Hz)
THRESH_HIGH_K_DEFAULT     = 0.6      # high = mean_peak * high_k
THRESH_LOW_K_DEFAULT      = 0.3      # low  = high * low_k

# best_axis hien tai (tu exercise_config.h) -- tool se RE-DERIVE lai nhung
# dung lam fallback. 0=ax 1=ay 2=az 3=gx 4=gy 5=gz
IDLE_NAME = 'idle'


# ═════════════════════════════════════════════════════════════
# 1. EnvelopePeakDetector -- MIRROR Y HET src/rep_counter.cpp
# ═════════════════════════════════════════════════════════════
class EnvelopePeakDetector:
    """Ban sao Python 1:1 cua EnvelopePeakDetector (firmware).

    update(val) nhan 1 mau Z-score CUA TRUC da chon, tra ve True khi peak.
    """

    def __init__(self, alpha=PD_MEAN_EMA_ALPHA_DEFAULT, ma_win=PD_MA_WIN_DEFAULT):
        self.alpha = float(alpha)
        self.ma_win = int(ma_win)
        self.thresh_high = 1e9
        self.thresh_low = 1e9
        self.reset()

    def configure(self, thresh_high, thresh_low):
        self.thresh_high = float(thresh_high)
        self.thresh_low = float(thresh_low)
        self.reset()

    def reset(self):
        self.mean_ema = 0.0
        self.mean_init = False
        self.ma_buf = np.zeros(self.ma_win, dtype=np.float64)
        self.ma_head = 0
        self.ma_count = 0
        self.ma_sum = 0.0
        self.envelope = 0.0
        self.above_high = False

    def update(self, val):
        # ── Buoc 1: DC removal qua slow EMA ──────────────────
        if not self.mean_init:
            self.mean_ema = val
            self.mean_init = True
        else:
            self.mean_ema = (1.0 - self.alpha) * self.mean_ema + self.alpha * val

        # ── Buoc 2: Rectify ──────────────────────────────────
        rectified = abs(val - self.mean_ema)

        # ── Buoc 3: Moving Average (circular buffer) ──────────
        if self.ma_count < self.ma_win:
            self.ma_buf[self.ma_count] = rectified
            self.ma_sum += rectified
            self.ma_count += 1
            if self.ma_count < self.ma_win:
                return False
            self.envelope = self.ma_sum / self.ma_win
            self.ma_head = 0
        else:
            oldest = self.ma_buf[self.ma_head]
            self.ma_buf[self.ma_head] = rectified
            self.ma_head = (self.ma_head + 1) % self.ma_win
            self.ma_sum = self.ma_sum - oldest + rectified
            self.envelope = self.ma_sum / self.ma_win

        # ── Buoc 4: Hysteresis Detection ──────────────────────
        if not self.above_high and self.envelope > self.thresh_high:
            self.above_high = True
            return False
        if self.above_high and self.envelope < self.thresh_low:
            self.above_high = False
            return True
        return False


# ═════════════════════════════════════════════════════════════
# 1b. AdaptiveDetector (v2) -- Schmitt tren tin hieu da khu DC,
#     dai nguong TI LE theo bien do tuc thoi (envelope) => thich nghi
#     tung buoi tap. Dem 1 lan / chu ky dao dong (het nhan doi).
# ═════════════════════════════════════════════════════════════
K_BAND_DEFAULT = 0.6    # T = k * envelope
T_MIN_DEFAULT  = 0.20   # san tuyet doi (z-units) chong idle noise
SMOOTH_WIN_DEFAULT = 9  # MA ngan lam muot centered truoc Schmitt (~0.18s) chong rung
PEAK_DECAY_DEFAULT = 0.995  # dinh-envelope giu-cham: T bam bien do nhip, khong tut giua nhip


class AdaptiveDetector:
    """centered = val - slow_dc, lam muot MA ngan ; envelope = MA(|centered_smooth|).
    Schmitt tren centered DA MUOT: vuot +T roi xuong duoi -T => 1 rep.
    Lam muot de moi nhip chi ban 1 dinh (het double do rung tan so cao).
    """

    def __init__(self, alpha=PD_MEAN_EMA_ALPHA_DEFAULT, ma_win=PD_MA_WIN_DEFAULT,
                 k_band=K_BAND_DEFAULT, t_min=T_MIN_DEFAULT, smooth_win=SMOOTH_WIN_DEFAULT,
                 peak_decay=PEAK_DECAY_DEFAULT):
        self.alpha = float(alpha)
        self.ma_win = int(ma_win)
        self.k_band = float(k_band)
        self.t_min = float(t_min)
        self.smooth_win = int(smooth_win)
        self.peak_decay = float(peak_decay)
        self.reset()

    def reset(self):
        self.mean_ema = 0.0
        self.warmup_sum = 0.0
        self.ma_buf = np.zeros(self.ma_win, dtype=np.float64)
        self.ma_head = 0
        self.ma_count = 0
        self.ma_sum = 0.0
        self.envelope = 0.0
        self.env_peak = 0.0   # dinh-envelope giu-cham (peak hold + slow decay)
        self.above = False   # da vuot +T (dang o nua tren cua chu ky)
        self.T = self.t_min
        self.sm_buf = np.zeros(self.smooth_win, dtype=np.float64)
        self.sm_head = 0
        self.sm_count = 0
        self.sm_sum = 0.0

    def update(self, val):
        # DC: 0.5s dau dung trung binh cong don (settle nhanh), sau do slow EMA
        if self.ma_count < self.ma_win:
            self.warmup_sum += val
            self.mean_ema = self.warmup_sum / (self.ma_count + 1)
        else:
            self.mean_ema = (1.0 - self.alpha) * self.mean_ema + self.alpha * val
        centered_raw = val - self.mean_ema

        # Lam muot centered (MA ngan) -> chong rung -> 1 dinh/nhip
        if self.sm_count < self.smooth_win:
            self.sm_buf[self.sm_count] = centered_raw
            self.sm_sum += centered_raw
            self.sm_count += 1
        else:
            self.sm_sum -= self.sm_buf[self.sm_head]
            self.sm_buf[self.sm_head] = centered_raw
            self.sm_sum += centered_raw
            self.sm_head = (self.sm_head + 1) % self.smooth_win
        centered = self.sm_sum / self.sm_count
        rectified = abs(centered)

        if self.ma_count < self.ma_win:
            self.ma_buf[self.ma_count] = rectified
            self.ma_sum += rectified
            self.ma_count += 1
            if self.ma_count < self.ma_win:
                return False
            self.envelope = self.ma_sum / self.ma_win
            self.ma_head = 0
        else:
            oldest = self.ma_buf[self.ma_head]
            self.ma_buf[self.ma_head] = rectified
            self.ma_head = (self.ma_head + 1) % self.ma_win
            self.ma_sum = self.ma_sum - oldest + rectified
            self.envelope = self.ma_sum / self.ma_win

        # Peak-hold: T bam theo DINH bien do (giu cao suot set) thay vi envelope
        # tuc thoi → wobble nho giua 2 nhip khong vuot nguong.
        self.env_peak = max(self.envelope, self.env_peak * self.peak_decay)
        self.T = self.k_band * self.env_peak
        if self.T < self.t_min:
            self.T = self.t_min

        if not self.above and centered > self.T:
            self.above = True
            return False
        if self.above and centered < -self.T:
            self.above = False
            return True   # 1 chu ky day du = 1 rep
        return False


def count_reps_adaptive(zsig, axis, alpha, ma_win, k_band, t_min, min_gap_ms):
    det = AdaptiveDetector(alpha, ma_win, k_band, t_min)
    min_gap_samples = int(round(min_gap_ms / 1000.0 * FS))
    reps = 0
    last_i = None
    peaks = []
    centered = np.zeros(len(zsig))
    env = np.zeros(len(zsig))
    Tarr = np.zeros(len(zsig))
    for i in range(len(zsig)):
        fired = det.update(zsig[i, axis])
        centered[i] = zsig[i, axis] - det.mean_ema
        env[i] = det.envelope
        Tarr[i] = det.T
        if fired:
            if last_i is None or (i - last_i) >= min_gap_samples:
                reps += 1
                last_i = i
                peaks.append(i)
    return reps, peaks, centered, env, Tarr


def compute_envelope(signal_1d, alpha=PD_MEAN_EMA_ALPHA_DEFAULT,
                     ma_win=PD_MA_WIN_DEFAULT):
    """Tra ve mang envelope tinh bang DUNG thuat toan detector (nhan qua).

    Dung chung cho phase0_find_config.py de 1 nguon chan ly.
    Cac mau truoc khi MA day (ma_win-1 mau dau) co envelope = 0 (giong firmware).
    """
    sig = np.asarray(signal_1d, dtype=np.float64)
    n = len(sig)
    env = np.zeros(n, dtype=np.float64)

    mean_ema = 0.0
    mean_init = False
    ma_buf = np.zeros(int(ma_win), dtype=np.float64)
    ma_head = 0
    ma_count = 0
    ma_sum = 0.0

    for i in range(n):
        val = sig[i]
        if not mean_init:
            mean_ema = val
            mean_init = True
        else:
            mean_ema = (1.0 - alpha) * mean_ema + alpha * val
        rectified = abs(val - mean_ema)

        if ma_count < ma_win:
            ma_buf[ma_count] = rectified
            ma_sum += rectified
            ma_count += 1
            if ma_count < ma_win:
                env[i] = 0.0
                continue
            ma_head = 0
        else:
            oldest = ma_buf[ma_head]
            ma_buf[ma_head] = rectified
            ma_head = (ma_head + 1) % ma_win
            ma_sum = ma_sum - oldest + rectified
        env[i] = ma_sum / ma_win

    return env


# ═════════════════════════════════════════════════════════════
# 2. Load + Z-score
# ═════════════════════════════════════════════════════════════
def load_csv_raw(path):
    """Doc 1 CSV raw -> (N,6) float."""
    df = pd.read_csv(path)
    return df[AXES].values.astype(np.float64)


def zscore(raw):
    """z = (x - mean) * inv_std  -- KHOP cong thuc firmware."""
    return (raw - HAR_MEAN) * HAR_INV_STD


def load_exercise(folder):
    """Load + concat tat ca CSV cua 1 exercise -> (N,6) Z-score, va list (name, arr) tung file."""
    files = sorted(Path(folder).glob('*.csv'))
    per_file = []
    for f in files:
        raw = load_csv_raw(f)
        if len(raw) < 100:
            continue
        per_file.append((f.name, zscore(raw)))
    concat = np.vstack([a for _, a in per_file]) if per_file else np.empty((0, 6))
    return concat, per_file


# ═════════════════════════════════════════════════════════════
# 3. Autocorrelation -- uoc luong chu ky rep (giong Phase 0)
# ═════════════════════════════════════════════════════════════
def autocorr_score(signal, max_samples=6000):
    # Gioi han do dai de uoc luong chu ky cho nhanh (6000 mau = 120s = thua reps)
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) > max_samples:
        signal = signal[:max_samples]
    n = len(signal)
    if n < 100:
        return 0.0, 0.0
    s = signal - np.mean(signal)
    if np.std(s) < 1e-9:
        return 0.0, 0.0
    s = s / np.std(s)

    max_lag = n // 2
    # Autocorrelation qua FFT  (O(n log n) thay vi O(n^2))
    fsize = 1 << int(np.ceil(np.log2(2 * n - 1)))
    f = np.fft.rfft(s, fsize)
    acf = np.fft.irfft(f * np.conj(f), fsize)[:max_lag]
    acf = acf / acf[0]

    search_start = 75   # 1.5s -- bo qua noise/tremor
    search_end = min(250, max_lag - 1)
    if search_end <= search_start:
        return 0.0, 0.0
    window = acf[search_start:search_end]
    peak_idx = int(np.argmax(window)) + search_start
    return float(acf[peak_idx]), peak_idx / FS


# ═════════════════════════════════════════════════════════════
# 4. Derive threshold tu envelope (giong Phase 0, dung envelope detector)
# ═════════════════════════════════════════════════════════════
def find_thresholds(envelope, period_samples, high_k, low_k):
    if period_samples < 5:
        period_samples = 25
    temp_thresh = np.mean(envelope) + np.std(envelope)
    peaks = []
    min_dist = max(period_samples // 2, 10)
    i = 0
    n = len(envelope)
    while i < n:
        if envelope[i] > temp_thresh:
            end = min(i + period_samples, n)
            local_peak = int(np.argmax(envelope[i:end])) + i
            if not peaks or (local_peak - peaks[-1]) >= min_dist:
                peaks.append(local_peak)
            i = local_peak + min_dist
        else:
            i += 1

    if len(peaks) < 2:
        high = float(np.percentile(envelope, 80) * high_k)
        return high, high * low_k
    mean_peak = float(np.mean(envelope[peaks]))
    high = mean_peak * high_k
    return high, high * low_k


# ═════════════════════════════════════════════════════════════
# 5. Phan tich 1 exercise -> chon best_axis, period, threshold
# ═════════════════════════════════════════════════════════════
def analyze_exercise(name, concat, alpha, ma_win, high_k, low_k):
    """Tra ve dict config (re-derived) cho 1 exercise.

    best_axis chon theo VARIANCE (std) -- nhat quan giua tay trai/phai (truc xoay
    giai phau giong nhau), thay vi autocorr (de bi nhieu).
    """
    stds = concat.std(axis=0)
    best_axis = int(np.argmax(stds))

    best_env = compute_envelope(concat[:, best_axis], alpha, ma_win)
    sc, period_s = autocorr_score(best_env)
    period_samples = int(period_s * FS) if period_s > 0 else 25
    high, low = find_thresholds(best_env, period_samples, high_k, low_k)
    if _MIN_GAP_OVERRIDE > 0:
        min_gap_ms = _MIN_GAP_OVERRIDE
    else:
        min_gap_ms = max(int(0.5 * period_s * 1000), 400) if period_s > 0 else 800

    return {
        'name': name,
        'best_axis': best_axis,
        'autocorr': round(sc, 3),
        'period_s': round(period_s, 2),
        'thresh_high': round(high, 4),
        'thresh_low': round(low, 4),
        'min_gap_ms': min_gap_ms,
        'axis_std': [round(s, 2) for s in stds],
    }


# ═════════════════════════════════════════════════════════════
# 6. Dem rep cua 1 file (detector + debounce) -- mo phong firmware
# ═════════════════════════════════════════════════════════════
def count_reps(zsig, axis, thresh_high, thresh_low, alpha, ma_win, min_gap_ms):
    """Chay detector tren 1 trace -> (rep_count, peak_indices, envelope_array).

    Mo phong RepValidator: debounce >= min_gap_ms giua 2 rep.
    (Khong gan State Machine -- coi nhu luon ACTIVE -- de danh gia THUAN detector.)
    """
    det = EnvelopePeakDetector(alpha, ma_win)
    det.configure(thresh_high, thresh_low)

    min_gap_samples = int(round(min_gap_ms / 1000.0 * FS))
    reps = 0
    last_peak_i = None
    peak_idx = []
    env = np.zeros(len(zsig))

    for i in range(len(zsig)):
        fired = det.update(zsig[i, axis])
        env[i] = det.envelope
        if fired:
            if last_peak_i is None or (i - last_peak_i) >= min_gap_samples:
                reps += 1
                last_peak_i = i
                peak_idx.append(i)
    return reps, peak_idx, env


# ═════════════════════════════════════════════════════════════
# 7. Reporting / Plot
# ═════════════════════════════════════════════════════════════
def _count(zsig, cfg, alpha, ma_win, adaptive, k_band, t_min):
    """Wrapper: tra ve (reps, peaks). Chon detector co dinh hoac adaptive."""
    if adaptive:
        r, p, _, _, _ = count_reps_adaptive(zsig, cfg['best_axis'], alpha, ma_win,
                                            k_band, t_min, cfg['min_gap_ms'])
        return r, p
    r, p, _ = count_reps(zsig, cfg['best_axis'], cfg['thresh_high'], cfg['thresh_low'],
                         alpha, ma_win, cfg['min_gap_ms'])
    return r, p


def run_analysis(data_dir, alpha, ma_win, high_k, low_k, adaptive, k_band, t_min):
    data_dir = Path(data_dir)
    folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])

    configs = {}
    mode = f"ADAPTIVE(k={k_band}, t_min={t_min})" if adaptive else f"FIXED(high_k={high_k}, low_k={low_k})"
    print("\n" + "=" * 78)
    print(f"  RE-DERIVE CONFIG  (alpha={alpha}, ma_win={ma_win})  detector={mode}")
    print("=" * 78)
    for folder in folders:
        name = folder.name
        if name == IDLE_NAME:
            continue
        concat, _ = load_exercise(folder)
        if len(concat) < 100:
            continue
        cfg = analyze_exercise(name, concat, alpha, ma_win, high_k, low_k)
        configs[name] = cfg
        print(f"\n[{name}] axis={cfg['best_axis']}({AXES[cfg['best_axis']]}) "
              f"period={cfg['period_s']}s autocorr={cfg['autocorr']} min_gap={cfg['min_gap_ms']}ms")
        if not adaptive:
            print(f"   thresh_high={cfg['thresh_high']}  thresh_low={cfg['thresh_low']}")
        print(f"   axis_std={cfg['axis_std']}")

    # ── Dem rep per-file, tach nhom TRAI(01-05)/PHAI(06-10) ──
    print("\n" + "=" * 78)
    print("  REP COUNT PER FILE   (TRAI=01-05, PHAI=06-10)")
    print("=" * 78)
    for folder in folders:
        name = folder.name
        _, per_file = load_exercise(folder)
        is_idle = (name == IDLE_NAME)
        cfg = configs.get(name)
        print(f"\n[{name}]" + ("" if is_idle else f"  axis={AXES[cfg['best_axis']]}"))
        if not is_idle and cfg is None:
            print("   (khong du data)")
            continue
        for idx, (fname, zsig) in enumerate(per_file):
            grp = "TRAI" if idx < 5 else "PHAI"
            if is_idle:
                fp = 0
                for c in configs.values():
                    r, _ = _count(zsig, c, alpha, ma_win, adaptive, k_band, t_min)
                    fp = max(fp, r)
                flag = "  <-- FALSE POSITIVE!" if fp > 0 else "  OK"
                print(f"   {fname:30s} dur={len(zsig)/FS:6.1f}s  idle_fp={fp}{flag}")
            else:
                reps, _ = _count(zsig, cfg, alpha, ma_win, adaptive, k_band, t_min)
                print(f"   [{grp}] {fname:24s} dur={len(zsig)/FS:6.1f}s  detect={reps:3d}")

    return configs


def plot_exercise(data_dir, exercise, alpha, ma_win, high_k, low_k, out_dir,
                  adaptive, k_band, t_min):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder = Path(data_dir) / exercise
    concat, per_file = load_exercise(folder)
    cfg = analyze_exercise(exercise, concat, alpha, ma_win, high_k, low_k)
    axis = cfg['best_axis']

    # Ve 1 file TRAI (idx 0) + 1 file PHAI (idx 5) de so 2 tay
    pick = [0, len(per_file) // 2 if len(per_file) > 5 else min(1, len(per_file) - 1)]
    pick = sorted(set([p for p in pick if 0 <= p < len(per_file)]))
    n = len(pick)
    fig, axs = plt.subplots(n, 1, figsize=(14, 3.4 * n), squeeze=False)
    for row, k in enumerate(pick):
        fname, zsig = per_file[k]
        t = np.arange(len(zsig)) / FS
        ax = axs[row][0]
        if adaptive:
            reps, peaks, centered, env, Tarr = count_reps_adaptive(
                zsig, axis, alpha, ma_win, k_band, t_min, cfg['min_gap_ms'])
            ax.plot(t, centered, color='#bbbbbb', lw=0.6, label=f'centered[{AXES[axis]}]')
            ax.plot(t, Tarr, color='red', lw=0.9, label='+T')
            ax.plot(t, -Tarr, color='orange', lw=0.9, label='-T')
        else:
            reps, peaks, env = count_reps(zsig, axis, cfg['thresh_high'],
                                          cfg['thresh_low'], alpha, ma_win, cfg['min_gap_ms'])
            ax.plot(t, zsig[:, axis], color='#bbbbbb', lw=0.6, label=f'z[{AXES[axis]}]')
            ax.plot(t, env, color='#1f77b4', lw=1.2, label='envelope')
            ax.axhline(cfg['thresh_high'], color='red', ls='--', lw=0.9, label='high')
            ax.axhline(cfg['thresh_low'], color='orange', ls='--', lw=0.9, label='low')
        for p in peaks:
            ax.axvline(p / FS, color='green', lw=0.7, alpha=0.6)
        grp = "TRAI" if k < 5 else "PHAI"
        ax.set_title(f"[{grp}] {fname}  |  detect={reps} reps  (period~{cfg['period_s']}s)")
        ax.set_xlabel('s')
        ax.legend(loc='upper right', fontsize=8)
    mode = f"ADAPTIVE k={k_band} t_min={t_min}" if adaptive else \
           f"high={cfg['thresh_high']} low={cfg['thresh_low']}"
    fig.suptitle(f"{exercise}  axis={axis}({AXES[axis]})  {mode}  alpha={alpha}", fontsize=11)
    fig.tight_layout()
    out = Path(out_dir) / f"rep_sim_{exercise}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"[plot] da luu: {out}")


def sweep_alpha(data_dir, ma_win, high_k, low_k):
    alphas = [0.05, 0.02, 0.013, 0.0067, 0.004, 0.002]
    print("\n" + "=" * 78)
    print("  SWEEP DC-EMA ALPHA  (tong rep detect / tong false-positive idle)")
    print("=" * 78)
    print(f"  {'alpha':>8} {'tau(s)':>7} {'cutoff(Hz)':>10}   tong_rep   idle_FP_files")
    for a in alphas:
        configs = {}
        data_dir_p = Path(data_dir)
        total_rep = 0
        for folder in sorted(data_dir_p.iterdir()):
            if not folder.is_dir() or folder.name == IDLE_NAME:
                continue
            concat, per_file = load_exercise(folder)
            if len(concat) < 100:
                continue
            cfg = analyze_exercise(folder.name, concat, a, ma_win, high_k, low_k)
            configs[folder.name] = cfg
            for _, zsig in per_file:
                r, _, _ = count_reps(zsig, cfg['best_axis'], cfg['thresh_high'],
                                     cfg['thresh_low'], a, ma_win, cfg['min_gap_ms'])
                total_rep += r
        # idle false positives
        idle_fp_files = 0
        idle_folder = data_dir_p / IDLE_NAME
        if idle_folder.exists():
            _, idle_files = load_exercise(idle_folder)
            for _, zsig in idle_files:
                hit = 0
                for c in configs.values():
                    r, _, _ = count_reps(zsig, c['best_axis'], c['thresh_high'],
                                         c['thresh_low'], a, ma_win, c['min_gap_ms'])
                    hit = max(hit, r)
                if hit > 0:
                    idle_fp_files += 1
        tau = 1.0 / a / FS
        cutoff = a / (2 * np.pi * (1.0 / FS))
        print(f"  {a:>8} {tau:>7.2f} {cutoff:>10.3f}   {total_rep:>8}   {idle_fp_files}")


# ═════════════════════════════════════════════════════════════
# 8. Main
# ═════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Offline rep-counter simulator (mirror firmware)")
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--alpha', type=float, default=PD_MEAN_EMA_ALPHA_DEFAULT)
    ap.add_argument('--ma_win', type=int, default=PD_MA_WIN_DEFAULT)
    ap.add_argument('--high_k', type=float, default=THRESH_HIGH_K_DEFAULT)
    ap.add_argument('--low_k', type=float, default=THRESH_LOW_K_DEFAULT)
    ap.add_argument('--plot', default=None, help='ten exercise de ve PNG')
    ap.add_argument('--out_dir', default='model_output')
    ap.add_argument('--sweep_alpha', action='store_true')
    ap.add_argument('--adaptive', action='store_true', help='dung AdaptiveDetector (Schmitt thich nghi)')
    ap.add_argument('--k_band', type=float, default=K_BAND_DEFAULT)
    ap.add_argument('--t_min', type=float, default=T_MIN_DEFAULT)
    ap.add_argument('--min_gap', type=int, default=0, help='override min_gap_ms (0=auto theo period)')
    args = ap.parse_args()

    global _MIN_GAP_OVERRIDE
    _MIN_GAP_OVERRIDE = args.min_gap

    if not Path(args.data_dir).exists():
        print(f"ERROR: khong tim thay '{args.data_dir}'")
        return

    if args.sweep_alpha:
        sweep_alpha(args.data_dir, args.ma_win, args.high_k, args.low_k)
        return

    if args.plot:
        plot_exercise(args.data_dir, args.plot, args.alpha, args.ma_win,
                      args.high_k, args.low_k, args.out_dir,
                      args.adaptive, args.k_band, args.t_min)
        return

    configs = run_analysis(args.data_dir, args.alpha, args.ma_win, args.high_k, args.low_k,
                           args.adaptive, args.k_band, args.t_min)

    # In block C++ de copy vao exercise_config.h
    print("\n" + "=" * 78)
    print("  COPY vao src/exercise_config.h  (EXERCISE_CONFIGS, index theo class)")
    print("=" * 78)
    order = ['bicep_curl', 'idle', 'lateral_raise', 'shoulder_press', 'tricep_ext']
    for nm in order:
        if nm == 'idle':
            print('    {"idle",            0, 0},  // khong dung')
            continue
        c = configs.get(nm)
        if not c:
            continue
        print(f'    {{"{nm}",{" "*(16-len(nm))}{c["best_axis"]}, {c["min_gap_ms"]}}},  '
              f'// axis={AXES[c["best_axis"]]} period={c["period_s"]}s')


if __name__ == '__main__':
    main()
