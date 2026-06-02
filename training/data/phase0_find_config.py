"""
phase0_find_config.py  --  DEPRECATED (shim)
============================================
Thuat toan tim config (best_axis / nguong) da chuyen het sang:

        training/rep_sim.py     <-- 1 NGUON CHAN LY, khop 100% voi firmware

Ly do: ban Phase 0 cu khu DC bang "mean toan cuc" (np.mean), khac voi detector
firmware -> nguong bi lech. Detector hien da doi sang ADAPTIVE Schmitt (nguong
tu thich nghi theo bien do), nen khong con derive nguong tuyet doi nua. rep_sim.py
chon best_axis theo VARIANCE (nhat quan tay trai/phai) va in thang block C++ cho
exercise_config.h.

File nay giu lai de lenh cu khong gay loi -- no goi thang rep_sim (che do adaptive).

Cach dung (khuyen nghi chay truc tiep rep_sim):
    python training/rep_sim.py --data_dir training/data --adaptive
    python training/rep_sim.py --data_dir training/data --adaptive --plot bicep_curl
"""

import argparse
import sys
from pathlib import Path

# Them thu muc training/ vao path de import rep_sim
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rep_sim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='training/data')
    args = ap.parse_args()

    print("=" * 70)
    print("  phase0_find_config.py DEPRECATED  ->  dung training/rep_sim.py")
    print("  Chay phan tich tuong duong (ADAPTIVE detector) duoi day:")
    print("=" * 70)

    if not Path(args.data_dir).exists():
        print(f"ERROR: khong tim thay '{args.data_dir}'. Vi du: --data_dir training/data")
        return

    rep_sim.run_analysis(
        args.data_dir,
        rep_sim.PD_MEAN_EMA_ALPHA_DEFAULT,
        rep_sim.PD_MA_WIN_DEFAULT,
        rep_sim.THRESH_HIGH_K_DEFAULT,
        rep_sim.THRESH_LOW_K_DEFAULT,
        True,                          # adaptive
        rep_sim.K_BAND_DEFAULT,
        rep_sim.T_MIN_DEFAULT,
    )


if __name__ == '__main__':
    main()
