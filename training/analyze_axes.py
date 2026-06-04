"""
analyze_axes.py -- De xuat truc + nguong T, TACH RIENG tung tay (TRAI/PHAI).
Vi deo tay trai vs phai -> thiet bi lat guong -> truc gyro troi co the KHAC nhau.
Voi moi (bai, tay): std 6 truc -> best gyro axis; T = 0.45*p95(|z|) cua truc do.
Nhan tay: bicep_curl 1-3=TRAI,4-10=PHAI ; bai khac 1-5=TRAI,6-10=PHAI.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HAR_MEAN = np.array([0.15883821, 0.01432643, -0.02781012,
                     -0.44101688, 0.47294450, -0.21489604])
HAR_INV_STD = np.array([1.48762882, 1.90575035, 1.96947049,
                        0.04014855, 0.02478677, 0.02976099])
AXES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
GYRO = [3, 4, 5]
DATA = Path('data')
LEFT_N = {'bicep_curl': 3}   # bicep ghi nham: 1-3 trai; mac dinh 5


def zload(f):
    raw = pd.read_csv(f)[AXES].values.astype(float)
    return (raw - HAR_MEAN) * HAR_INV_STD


def report(tag, zlist):
    if not zlist:
        print(f"   {tag}: (khong co file)")
        return
    z = np.vstack(zlist)
    std = z.std(axis=0)
    best = GYRO[int(np.argmax(std[GYRO]))]
    T = round(0.45 * np.percentile(np.abs(z[:, best]), 95), 1)
    stds = "  ".join(f"{AXES[j]}={std[j]:.2f}" for j in range(6))
    print(f"   {tag}: best gyro = {AXES[best]}  T={T}   | std: {stds}")


for ex in ['bicep_curl', 'lateral_raise', 'shoulder_press', 'tricep_ext']:
    fs = sorted((DATA / ex).glob('*.csv'))
    if not fs:
        continue
    ln = LEFT_N.get(ex, 5)
    print(f"\n[{ex}]")
    report('TRAI', [zload(f) for f in fs[:ln]])
    report('PHAI', [zload(f) for f in fs[ln:]])
