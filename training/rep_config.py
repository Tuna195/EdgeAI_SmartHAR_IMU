"""
rep_config.py  --  CAU HINH REP-DETECTION PER-BAI  (1 nguon chan ly)
=====================================================================
Tach rieng de DE MO RONG: them bai moi = them 1 dong vao REP_CONFIG.
Dung chung cho sim_peakdet.py (offline) va sau nay sinh block C++ cho firmware.

Thuat toan dem: peakdet (Billauer) bang DELTA tren z-score 1 truc gyro.
  - delta = bien do swing TOI THIEU (z-units) de chot 1 dinh / day.
            Dieu kien:  noise < delta < bien_do_rep_that.
  - 1 valley + 1 peak = 1 rep  (khoa pha: cuc dau dinh pha E1, cham cuc doi = 1 rep).
  - peakdet BAT BIEN voi offset DC -> khong can tru mean, ben hon Schmitt +-T.

Truong cau hinh moi bai:
  axis            : 0=ax 1=ay 2=az 3=gx 4=gy 5=gz  (truc chinh, ban tu chon)
  delta           : nguong swing toi thieu (z-units)
  min_gap_ms      : debounce — khoang cach toi thieu giua 2 rep (chong rung)
  smooth_win      : cua so MA causal lam muot truoc peakdet (mau @50Hz; 1=tat)
  reset_timeout_ms: khong co dinh/day nao trong khoang nay -> reset pha (ve idle)
  min_abs         : san bien do TUYET DOI — cuc tri phai vuot +-min_abs moi tinh.
                    0 = tat (peakdet thuan). Chi la knob phu giam FP idle.

LUU Y IDLE: data idle CO chuyen dong that (z gyro vuot >1.6 nhieu lan) nen MOI
detector deu FP cao tren idle. Idle duoc loai o tang CLASSIFIER (detector chi
chay khi model phan loai la bai tap), KHONG tune detector theo idle FP.

Ghi chu truc mac dinh (theo memory + analyze_axes.py) — BAN se chot lai:
  bicep_curl=gy, lateral_raise=gy, shoulder_press=gx, tricep_ext=gz
"""

# Thu tu trsuc khop norm_params.h / firmware
AXES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
FS = 50.0  # Hz

# Z-score params — COPY tu src/norm_params.h (thu tu ax,ay,az,gx,gy,gz)
HAR_MEAN = [0.15883821, 0.01432643, -0.02781012,
            -0.44101688, 0.47294450, -0.21489604]
HAR_INV_STD = [1.48762882, 1.90575035, 1.96947049,
               0.04014855, 0.02478677, 0.02976099]

# ─── Config per-bai ──────────────────────────────────────────
# delta khoi diem (se tune); BAN dua truc chinh -> cap nhat 'axis'.
REP_CONFIG = {
    # gy: bicep_curl_01 = 20 rep (ground truth), on dinh ca dai delta 0.5..1.5.
    # (Da thu ax: file_01=19 va lateral-ax tay TRAI ~0 -> gy tot hon han.)
    'bicep_curl':     dict(axis=4, delta=1.0, min_gap_ms=600, smooth_win=5, reset_timeout_ms=3000, min_abs=0.0),
    'lateral_raise':  dict(axis=4, delta=1.0, min_gap_ms=600, smooth_win=5, reset_timeout_ms=3000, min_abs=0.0),
    # ay: tot hon han gx cu. PHAI dem ~20 sach; TRAI bien do YEU (shoulder it xoay)
    # nen delta phai thap 0.5 va TRAI van hoi under-count (gioi han vat ly).
    'shoulder_press': dict(axis=1, delta=0.5, min_gap_ms=600, smooth_win=5, reset_timeout_ms=3000, min_abs=0.0),
    # ay: plateau rat phang 0.5..1.5 -> chon 0.9 (bien chong nhieu tot).
    'tricep_ext':     dict(axis=1, delta=0.9, min_gap_ms=600, smooth_win=5, reset_timeout_ms=3000, min_abs=0.0),
}
