"""
eval_generalization.py  --  Đánh giá generalization THẬT của model
====================================================================
Lý do: trainAI.py chia train/val theo *window* ngẫu nhiên → window cùng 1 file
(chồng 50%, cùng người/buổi/hướng đeo) rơi vào cả train lẫn val → rò rỉ mức
session → accuracy 100% bị thổi phồng.

Script này KHÔNG đụng tới model đang deploy. Nó tái dùng load_csv/sliding_window/
build_model/augment_dataset từ trainAI.py nhưng chia dữ liệu THEO FILE:

  --mode grouped    : GroupKFold theo file (không file nào nằm cả 2 phía)
                      → accuracy THẬT khi gặp buổi tập "lạ".
  --mode cross_arm  : train tay này / test tay kia (01-05=trái, 06-10=phải)
                      → độ bền với hướng/vị trí đeo. Chạy cả 2 chiều.

Không dùng EarlyStopping nhìn vào test (tránh peeking) — train số epoch cố định.

Cách dùng (chạy trong thư mục training/):
    python eval_generalization.py --mode grouped   --data_dir data
    python eval_generalization.py --mode cross_arm --data_dir data
    python eval_generalization.py --mode both      --data_dir data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix

import trainAI as T  # tái dùng: CFG, load_csv, sliding_window, build_model, augment_dataset


# ─────────────────────────────────────────────────────────────
# Load dữ liệu CÓ NHÓM (group = file, arm = trái/phải)
# ─────────────────────────────────────────────────────────────
def load_grouped(data_dir):
    """Trả về X(N,150,6), y(N,), groups(N,) = id file, arms(N,) = 'L'/'R'."""
    Xs, ys, groups, arms = [], [], [], []
    gid = 0
    for label_idx, cls in enumerate(T.CFG["classes"]):
        cdir = Path(data_dir) / cls
        files = sorted(cdir.glob("*.csv"))
        for pos, f in enumerate(files):
            data = T.load_csv(f)
            if data is None:
                continue
            wins, labs = T.sliding_window(data, label_idx)
            if not wins:
                continue
            # 01-05 = trái, 06-10 = phải (theo thứ tự file đã sort trong từng class)
            arm = 'L' if pos < 5 else 'R'
            for w, l in zip(wins, labs):
                Xs.append(w); ys.append(l); groups.append(gid); arms.append(arm)
            gid += 1
    return (np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32),
            np.array(groups), np.array(arms))


def train_eval(Xtr_raw, ytr, Xva_raw, yva, epochs, augment=True, rotate=False, verbose=0):
    """Train 1 lần (epoch cố định, KHÔNG peeking test) → trả (acc, y_pred).

    Scaler fit trên train GỐC (chưa rotate) — khớp cách firmware deploy.
    rotate=True → rotation-augment raw train TRƯỚC khi áp scaler (robust hướng đeo).
    """
    mean = Xtr_raw.reshape(-1, 6).mean(axis=0)
    std = Xtr_raw.reshape(-1, 6).std(axis=0)
    std = np.where(std == 0, 1e-8, std)

    if rotate:
        Xtr_raw, ytr = T.augment_rotation(Xtr_raw, ytr)

    Xtr = (Xtr_raw - mean) / std
    Xva = (Xva_raw - mean) / std
    if augment:
        Xtr, ytr = T.augment_dataset(Xtr, ytr)

    model = T.build_model(n_classes=len(T.CFG["classes"]))
    model.fit(Xtr, ytr, epochs=epochs, batch_size=T.CFG["batch_size"], verbose=verbose)
    y_pred = np.argmax(model.predict(Xva, verbose=0), axis=1)
    acc = float(np.mean(y_pred == yva))
    return acc, y_pred


def _report(yva, y_pred):
    names = T.CFG["classes"]
    print(classification_report(yva, y_pred, target_names=names,
                                labels=list(range(len(names))), zero_division=0))
    print("Confusion matrix (rows=thật, cols=đoán):")
    cm = confusion_matrix(yva, y_pred, labels=list(range(len(names))))
    print("        " + " ".join(f"{n[:6]:>6}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n[:6]:>6} " + " ".join(f"{v:6d}" for v in cm[i]))


# ─────────────────────────────────────────────────────────────
# Mode 1: GroupKFold theo file
# ─────────────────────────────────────────────────────────────
def run_grouped(data_dir, epochs, folds, augment, rotate):
    X, y, groups, _ = load_grouped(data_dir)
    n_files = len(np.unique(groups))
    print("\n" + "=" * 70)
    print(f"  MODE grouped — GroupKFold theo file ({folds} folds, {n_files} files, "
          f"{len(X)} windows, epochs={epochs}, rotate={rotate})")
    print("=" * 70)

    gkf = GroupKFold(n_splits=folds)
    accs = []
    for k, (tr, va) in enumerate(gkf.split(X, y, groups), 1):
        acc, _ = train_eval(X[tr], y[tr], X[va], y[va], epochs, augment, rotate)
        n_va_files = len(np.unique(groups[va]))
        print(f"  Fold {k}/{folds}: test {n_va_files} files, {len(va)} windows "
              f"→ acc = {acc*100:.1f}%")
        accs.append(acc)
    print(f"\n  ➜ Accuracy THẬT (leave-files-out): {np.mean(accs)*100:.1f}% "
          f"± {np.std(accs)*100:.1f}%   (so với 100% của split cũ)")


# ─────────────────────────────────────────────────────────────
# Mode 2: Cross-arm
# ─────────────────────────────────────────────────────────────
def run_cross_arm(data_dir, epochs, augment, rotate):
    X, y, _, arms = load_grouped(data_dir)
    print("\n" + "=" * 70)
    print(f"  MODE cross_arm — train tay này / test tay kia (epochs={epochs}, rotate={rotate})")
    print("=" * 70)

    for train_arm, test_arm in [('L', 'R'), ('R', 'L')]:
        tr = (arms == train_arm)
        va = (arms == test_arm)
        label = "TRÁI→PHẢI" if train_arm == 'L' else "PHẢI→TRÁI"
        print(f"\n── {label}  (train {tr.sum()} win / test {va.sum()} win) ──")
        acc, y_pred = train_eval(X[tr], y[tr], X[va], y[va], epochs, augment, rotate)
        print(f"  acc = {acc*100:.1f}%")
        _report(y[va], y_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--mode', choices=['grouped', 'cross_arm', 'both'], default='both')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--aug', type=int, default=2, help='aug_factor cho eval (nhỏ hơn để nhanh)')
    ap.add_argument('--no_aug', action='store_true')
    ap.add_argument('--rotate', action='store_true', help='bật rotation augmentation (robust hướng đeo)')
    ap.add_argument('--rot_factor', type=int, default=2)
    ap.add_argument('--rot_max_deg', type=float, default=180.0)
    args = ap.parse_args()

    if not Path(args.data_dir).exists():
        print(f"ERROR: không thấy '{args.data_dir}'")
        return

    T.CFG["aug_factor"] = args.aug  # giảm aug để eval nhanh hơn
    T.CFG["rot_factor"] = args.rot_factor
    T.CFG["rot_max_deg"] = args.rot_max_deg
    augment = not args.no_aug

    if args.mode in ('grouped', 'both'):
        run_grouped(args.data_dir, args.epochs, args.folds, augment, args.rotate)
    if args.mode in ('cross_arm', 'both'):
        run_cross_arm(args.data_dir, args.epochs, augment, args.rotate)


if __name__ == '__main__':
    main()
