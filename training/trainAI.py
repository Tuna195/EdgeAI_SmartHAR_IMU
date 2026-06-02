"""
trainAI.py — IMU Gesture Recognition Training Pipeline
=======================================================
Target Hardware : ESP32-S3 (TFLite Micro)
Classes         : bicep_curl, idle, lateral_raise, shoulder_press, tricep_ext
Input           : CSV files với 6 cột (ax, ay, az, gx, gy, gz) @ 50Hz
Output          : model.keras | model.tflite (int8) | scaler_params.json

Cách chạy:
    pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
    python trainAI.py
"""

import os
import sys

# Buộc stdout/stderr dùng UTF-8 trên Windows (tránh lỗi cp1252 với ký tự đặc biệt)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models, callbacks

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ─────────────────────────────────────────────────────────────
# 0. CẤU HÌNH TRUNG TÂM — chỉnh ở đây, không cần sửa chỗ khác
# ─────────────────────────────────────────────────────────────
CFG = {
    # Paths
    "data_dir"      : "data",
    "output_dir"    : "model_output",

    # Classes — pushup & running bị bỏ qua tự động
    "classes"       : ["bicep_curl", "idle", "lateral_raise",
                       "shoulder_press", "tricep_ext"],

    # Tên 6 cột trong CSV (thứ tự phải khớp với firmware)
    "columns"       : ["ax", "ay", "az", "gx", "gy", "gz"],

    # Windowing
    "window_size"   : 150,      # samples (= 3.0 giây @ 50Hz)
    "stride"        : 75,       # 50% overlap
    "trim_windows"  : 2,        # bỏ N windows đầu + N windows cuối mỗi file

    # Augmentation
    "aug_factor"    : 4,        # số bản aug tạo thêm cho mỗi window gốc
    "jitter_std"    : 0.05,     # nhiễu Gaussian
    "scale_range"   : (0.9, 1.1),
    "shift_max"     : 10,       # samples (time shift)

    # Rotation augmentation (raw-space, TRƯỚC normalize) — DUNG SAI lệch đeo NHẸ.
    # Xoay proper (det +1) áp CÙNG ma trận cho accel & gyro (đều là 3-vector).
    #
    # ⚠️ Premise dự án = hướng đeo CỐ ĐỊNH. Góc lớn (90°/180°) xoay quanh trục bất
    # kỳ sẽ ÁNH XẠ gy↔gz↔gx → gộp lẫn bicep/tricep/lateral (đã quan sát thực tế).
    # → Chỉ cho lệch NHẸ ±15° (dung sai đeo hơi xê dịch) mà KHÔNG conflate trục.
    # 0 = tắt hẳn (đeo cực kỳ cố định).
    "rot_factor"    : 1,        # số bản xoay thêm cho mỗi window (0 = tắt)
    "rot_max_deg"   : 15.0,     # biên độ góc xoay (±) quanh trục ngẫu nhiên — NHẸ

    # Training
    "test_size"     : 0.20,
    "batch_size"    : 32,
    "epochs"        : 80,
    "lr"            : 0.001,
    "patience"      : 15,       # EarlyStopping
    "seed"          : 42,
}

np.random.seed(CFG["seed"])
tf.random.set_seed(CFG["seed"])
os.makedirs(CFG["output_dir"], exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING & WINDOWING
# ─────────────────────────────────────────────────────────────
def load_csv(filepath: Path) -> np.ndarray | None:
    """
    Đọc một file CSV và trả về numpy array shape (N, 6).
    Tự động bỏ qua file lỗi hoặc thiếu cột.
    """
    try:
        df = pd.read_csv(filepath)

        # Kiểm tra đủ cột
        missing = [c for c in CFG["columns"] if c not in df.columns]
        if missing:
            print(f"  ⚠️  Bỏ qua {filepath.name} — thiếu cột: {missing}")
            return None

        data = df[CFG["columns"]].dropna().values.astype(np.float32)

        # Tối thiểu phải có đủ 1 window
        if len(data) < CFG["window_size"]:
            print(f"  ⚠️  Bỏ qua {filepath.name} — quá ít samples ({len(data)})")
            return None

        return data

    except Exception as e:
        print(f"  ❌  Lỗi đọc {filepath.name}: {e}")
        return None


def sliding_window(data: np.ndarray, label: int) -> tuple[list, list]:
    """
    Cắt data thành các windows với stride, bỏ N windows đầu/cuối.
    Trả về (windows list, labels list).
    """
    W = CFG["window_size"]
    S = CFG["stride"]
    T = CFG["trim_windows"]

    windows, labels = [], []
    n_windows = (len(data) - W) // S + 1

    for i in range(n_windows):
        # Bỏ qua N windows đầu và N windows cuối
        if i < T or i >= (n_windows - T):
            continue
        start = i * S
        window = data[start : start + W]
        if len(window) == W:
            windows.append(window)
            labels.append(label)

    return windows, labels


def load_all_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load toàn bộ data từ CFG["data_dir"], chỉ lấy các class trong CFG["classes"].
    Trả về X (N, 150, 6), y (N,), groups (N,) = id FILE của mỗi window.

    groups dùng để chia train/val THEO FILE (không để window cùng 1 file rơi
    vào cả 2 phía → tránh rò rỉ session làm accuracy ảo).
    """
    print("\n" + "="*55)
    print("  BƯỚC 1: LOAD DATA & SLIDING WINDOW")
    print("="*55)

    all_windows, all_labels, all_groups = [], [], []
    file_id = 0  # id duy nhất cho mỗi file (xuyên suốt mọi class)

    for label_idx, class_name in enumerate(CFG["classes"]):
        class_dir = Path(CFG["data_dir"]) / class_name
        if not class_dir.exists():
            print(f"\n❌ Không tìm thấy thư mục: {class_dir}")
            continue

        csv_files = sorted(class_dir.glob("*.csv"))
        if not csv_files:
            print(f"\n⚠️  Không có file CSV trong: {class_dir}")
            continue

        class_count = 0
        print(f"\n[{label_idx}] {class_name} ({len(csv_files)} files):")

        for csv_path in csv_files:
            data = load_csv(csv_path)
            if data is None:
                continue

            wins, labs = sliding_window(data, label_idx)
            all_windows.extend(wins)
            all_labels.extend(labs)
            all_groups.extend([file_id] * len(wins))  # mọi window của file này cùng group
            file_id += 1
            class_count += len(wins)
            print(f"      ✅ {csv_path.name:30s} → {len(wins):3d} windows")

        print(f"      → Tổng: {class_count} windows cho class '{class_name}'")

    X = np.array(all_windows, dtype=np.float32)  # (N, 150, 6)
    y = np.array(all_labels,  dtype=np.int32)     # (N,)
    groups = np.array(all_groups, dtype=np.int32)  # (N,)

    print(f"\n📦 Dataset gốc: {X.shape} | Labels: {y.shape} | Files: {file_id}")
    _print_class_distribution(y, "Phân bố trước augmentation")
    return X, y, groups


def _print_class_distribution(y: np.ndarray, title: str = ""):
    if title:
        print(f"\n  [{title}]")
    for i, cls in enumerate(CFG["classes"]):
        count = np.sum(y == i)
        bar = "█" * (count // 5)
        print(f"    {cls:20s}: {count:4d}  {bar}")


# ─────────────────────────────────────────────────────────────
# 2. Z-SCORE NORMALIZATION  (fit / apply tách riêng)
# ─────────────────────────────────────────────────────────────
def fit_scaler(X_train_raw: np.ndarray, save: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Tính mean/std CHỈ trên X_train_raw GỐC (chip-hướng-vào thật) → khớp với
    dữ liệu firmware gặp lúc inference. KHÔNG fit trên bản đã rotation-augment
    (sẽ làm scaler "trung bình hoá hướng" → lệch với deployment thật).
    """
    mean = X_train_raw.reshape(-1, 6).mean(axis=0)
    std  = X_train_raw.reshape(-1, 6).std(axis=0)
    std  = np.where(std == 0, 1e-8, std)   # tránh chia 0

    scaler_params = {"mean": mean.tolist(), "std": std.tolist(), "axes": CFG["columns"]}
    if save:
        scaler_path = Path(CFG["output_dir"]) / "scaler_params.json"
        with open(scaler_path, "w") as f:
            json.dump(scaler_params, f, indent=2)
        print(f"\n✅ Scaler params đã lưu: {scaler_path}")
    print(f"   mean = {[f'{v:.4f}' for v in mean.tolist()]}")
    print(f"   std  = {[f'{v:.4f}' for v in std.tolist()]}")
    print(f"   Fit trên {len(X_train_raw):,} window GỐC (chưa rotate, chưa augment).")
    return mean, std, scaler_params


def apply_scaler(X_raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """z = (x - mean) / std — dùng đúng scaler đã fit trên train gốc."""
    return (X_raw - mean) / std


# ─────────────────────────────────────────────────────────────
# 2b. ROTATION AUGMENTATION (raw-space, TRƯỚC normalize)
# ─────────────────────────────────────────────────────────────
def _rot_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Ma trận xoay 3x3 (Rodrigues) — proper rotation, det = +1."""
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.eye(3, dtype=np.float32)
    x, y, z = axis / n
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ], dtype=np.float32)


def augment_rotation(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sinh rot_factor bản XOAY cho mỗi window (giữ cả bản gốc).
    Áp CÙNG ma trận xoay R cho bộ 3 accel [0:3] và bộ 3 gyro [3:6] — đúng vật lý
    (cả hai là 3-vector dưới proper rotation; KHÔNG dùng cho phản chiếu trái/phải).
    Mô phỏng lệch/đổi hướng đeo (gồm "chip ra ngoài" ≈ xoay ~180°) mà KHÔNG cần data mới.
    """
    factor = CFG.get("rot_factor", 0)
    if factor <= 0:
        return X, y

    max_rad = np.deg2rad(CFG.get("rot_max_deg", 180.0))
    # CHỈ xoay các BÀI TẬP, KHÔNG xoay idle: tránh class idle "phình to" (idle ở
    # mọi hướng) nuốt mất bài tập (nhất là bicep yếu).
    idle_idx = CFG["classes"].index("idle") if "idle" in CFG["classes"] else -1
    ex_mask = (y != idle_idx)
    X_ex, y_ex = X[ex_mask], y[ex_mask]

    print("\n" + "="*55)
    print("  BƯỚC 1b: ROTATION AUGMENTATION (raw) — CHỈ bài tập, KHÔNG idle")
    print("="*55)
    print(f"  Xoay {len(X_ex)}/{len(X)} window (bỏ {np.sum(~ex_mask)} window idle)")

    out_X, out_y = [X], [y]  # giữ TẤT CẢ bản gốc (gồm idle)
    for p in range(factor):
        Xr = np.empty_like(X_ex)
        for i in range(len(X_ex)):
            axis = np.random.normal(size=3)
            angle = np.random.uniform(-max_rad, max_rad)
            R = _rot_matrix(axis, angle)
            Xr[i, :, 0:3] = X_ex[i, :, 0:3] @ R.T   # accel
            Xr[i, :, 3:6] = X_ex[i, :, 3:6] @ R.T   # gyro
        out_X.append(Xr)
        out_y.append(y_ex)
        print(f"  Rotate pass {p+1}/{factor}: +{len(Xr)} window bài tập (±{CFG['rot_max_deg']}°)")

    X_out = np.concatenate(out_X, axis=0)
    y_out = np.concatenate(out_y, axis=0)
    print(f"\n📦 Sau rotation: {X_out.shape}")
    return X_out, y_out


# ─────────────────────────────────────────────────────────────
# 3. DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────
def augment_jitter(window: np.ndarray) -> np.ndarray:
    """Thêm nhiễu Gaussian nhỏ — giả lập sensor noise."""
    noise = np.random.normal(0, CFG["jitter_std"], window.shape).astype(np.float32)
    return window + noise


def augment_scaling(window: np.ndarray) -> np.ndarray:
    """Nhân toàn bộ window với scalar ngẫu nhiên — giả lập cường độ khác nhau."""
    lo, hi = CFG["scale_range"]
    scale = np.random.uniform(lo, hi)
    return (window * scale).astype(np.float32)


def augment_time_shift(window: np.ndarray) -> np.ndarray:
    """Dịch chuỗi thời gian ±N samples — giả lập timing khác nhau."""
    shift = np.random.randint(-CFG["shift_max"], CFG["shift_max"] + 1)
    return np.roll(window, shift, axis=0).astype(np.float32)

def augment_dataset(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Tạo aug_factor bản augment cho mỗi window gốc.
    Mỗi bản áp dụng 1–2 kỹ thuật ngẫu nhiên để tăng đa dạng.
    """
    print("\n" + "="*55)
    print("  BƯỚC 2: DATA AUGMENTATION")
    print("="*55)

    aug_fns = [augment_jitter, augment_scaling,
               augment_time_shift]

    X_aug_list = [X]
    y_aug_list = [y]

    for i in range(CFG["aug_factor"]):
        X_batch = []
        for window in X:
            # Chọn 1–2 augmentation ngẫu nhiên và áp dụng tuần tự
            chosen = np.random.choice(aug_fns,
                                      size=np.random.randint(1, 3),
                                      replace=False)
            w = window.copy()
            for fn in chosen:
                w = fn(w)
            X_batch.append(w)
        X_aug_list.append(np.array(X_batch, dtype=np.float32))
        y_aug_list.append(y)
        print(f"  Pass {i+1}/{CFG['aug_factor']}: +{len(X_batch)} windows")

    X_out = np.concatenate(X_aug_list, axis=0)
    y_out = np.concatenate(y_aug_list, axis=0)

    # Shuffle
    idx = np.random.permutation(len(X_out))
    X_out, y_out = X_out[idx], y_out[idx]

    print(f"\n📦 Dataset sau augmentation: {X_out.shape}")
    _print_class_distribution(y_out, "Phân bố sau augmentation")
    return X_out, y_out


# ─────────────────────────────────────────────────────────────
# 4. BUILD MODEL
# ─────────────────────────────────────────────────────────────
def build_model(n_classes: int) -> tf.keras.Model:
    """
    1D-CNN tối ưu cho TinyML / ESP32-S3.
    Flash ~25-35KB | Peak SRAM ~12-18KB | Inference <30ms
    """
    inp = layers.Input(shape=(CFG["window_size"], len(CFG["columns"])),
                       name="imu_input")

    # Block 1
    x = layers.Conv1D(16, kernel_size=7, padding="same", name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)          # → (75, 16)

    # Block 2
    x = layers.Conv1D(32, kernel_size=5, padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)          # → (37, 32)

    # Classifier head
    x = layers.Flatten(name="flatten")(x)                          # → (1184,)
    x = layers.Dense(32, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inp, out, name="GestureNet_1DCNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CFG["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────────────────────
def train(model: tf.keras.Model,
          X_train, y_train,
          X_val,   y_val) -> tf.keras.callbacks.History:

    print("\n" + "="*55)
    print("  BƯỚC 3: TRAINING")
    print("="*55)
    model.summary()

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=CFG["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-5,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(Path(CFG["output_dir"]) / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CFG["epochs"],
        batch_size=CFG["batch_size"],
        callbacks=cb_list,
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────────────────────
# 6. EVALUATION & PLOTS
# ─────────────────────────────────────────────────────────────
def evaluate(model: tf.keras.Model,
             X_val: np.ndarray, y_val: np.ndarray,
             history: tf.keras.callbacks.History) -> tuple[float, dict]:

    print("\n" + "="*55)
    print("  BƯỚC 4: EVALUATION")
    print("="*55)

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

    # ── Classification Report ────────────────────────────────
    report_str = classification_report(y_val, y_pred,
                                       target_names=CFG["classes"])
    print("\n📊 Classification Report:")
    print(report_str)

    val_acc = np.mean(y_pred == y_val)
    print(f"✅ Val Accuracy: {val_acc*100:.2f}%")

    # ── Per-class accuracy ───────────────────────────────────
    per_class_acc = {}
    for i, cls in enumerate(CFG["classes"]):
        mask = (y_val == i)
        per_class_acc[cls] = float(
            np.mean(y_pred[mask] == y_val[mask]) if mask.sum() > 0 else 0.0
        )

    # ── Plot 1: Training curves ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History", fontsize=13, fontweight="bold")

    axes[0].plot(history.history["accuracy"],     label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    curve_path = Path(CFG["output_dir"]) / "training_curves.png"
    plt.savefig(curve_path, dpi=120)
    plt.close()
    print(f"📈 Training curves → {curve_path}")

    # ── Plot 2: Confusion matrix ─────────────────────────────
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CFG["classes"],
                yticklabels=CFG["classes"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    cm_path = Path(CFG["output_dir"]) / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"📊 Confusion matrix → {cm_path}")

    # ── Plot 3: Per-class accuracy bar chart ─────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    classes_list = list(per_class_acc.keys())
    accs_pct     = [per_class_acc[c] * 100 for c in classes_list]
    bar_colors   = ["#2ecc71" if a >= 90 else "#f39c12" if a >= 75 else "#e74c3c"
                    for a in accs_pct]
    bars = ax.barh(classes_list, accs_pct, color=bar_colors)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy", fontsize=13, fontweight="bold")
    for bar, acc in zip(bars, accs_pct):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", fontsize=10)
    ax.axvline(90, color="gray", linestyle="--", alpha=0.5, label="90% line")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    pc_path = Path(CFG["output_dir"]) / "per_class_accuracy.png"
    plt.savefig(pc_path, dpi=120)
    plt.close()
    print(f"📊 Per-class accuracy → {pc_path}")

    return val_acc, {
        "val_acc"      : float(val_acc),
        "per_class_acc": per_class_acc,
        "report_str"   : report_str,
        "epochs_run"   : len(history.history["loss"]),
    }


# ─────────────────────────────────────────────────────────────
# 7. EXPORT — Keras + TFLite int8 Quantization
# ─────────────────────────────────────────────────────────────
def export_tflite(model: tf.keras.Model,
                  X_ref: np.ndarray):
    """
    Export model ra 2 định dạng:
      1. model.keras     — backup đầy đủ
      2. model.tflite    — int8 quantized, dùng trên ESP32-S3

    X_ref: một tập nhỏ (~200 windows) làm representative dataset
           cho quantization calibration.
    """
    print("\n" + "="*55)
    print("  BƯỚC 5: EXPORT TFLITE (int8)")
    print("="*55)

    # 5a. Lưu full model
    keras_path = Path(CFG["output_dir"]) / "model.keras"
    model.save(keras_path)
    print(f"✅ Keras model → {keras_path}")

    # 5b. Representative dataset generator (bắt buộc cho int8)
    def representative_dataset():
        # Lấy tối đa 200 samples từ X_ref
        samples = X_ref[:200]
        for sample in samples:
            # TFLite cần shape (1, 150, 6) và dtype float32
            yield [sample[np.newaxis, :, :].astype(np.float32)]

    # 5c. Convert với full int8 quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path = Path(CFG["output_dir"]) / "model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"✅ TFLite int8 model → {tflite_path}")
    print(f"   Model size: {size_kb:.1f} KB")

    if size_kb > 50:
        print("  ⚠️  Model > 50KB — cân nhắc giảm filters nếu Flash bị hạn chế")
    else:
        print("  ✅ Kích thước ổn, phù hợp cho ESP32-S3")

    # 5d. Verify: chạy thử inference trên TFLite để kiểm tra
    print("\n🔍 Verify TFLite inference...")
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    inp_details  = interpreter.get_input_details()
    out_details  = interpreter.get_output_details()
    inp_scale,  inp_zp  = inp_details[0]["quantization"]
    out_scale,  out_zp  = out_details[0]["quantization"]

    test_sample = X_ref[0:1].astype(np.float32)
    q_input = (test_sample / inp_scale + inp_zp).astype(np.int8)

    interpreter.set_tensor(inp_details[0]["index"], q_input)
    interpreter.invoke()

    q_output = interpreter.get_tensor(out_details[0]["index"])
    probs = (q_output.astype(np.float32) - out_zp) * out_scale
    pred_class = CFG["classes"][np.argmax(probs)]

    print(f"   Sample test → predicted: '{pred_class}' ✅")
    print(f"   Input  quantization: scale={inp_scale:.6f}, zp={inp_zp}")
    print(f"   Output quantization: scale={out_scale:.6f}, zp={out_zp}")

    # Lưu quantization params để dùng trong firmware
    quant_info = {
        "input_scale"  : float(inp_scale),
        "input_zp"     : int(inp_zp),
        "output_scale" : float(out_scale),
        "output_zp"    : int(out_zp),
        "classes"      : CFG["classes"],
        "window_size"  : CFG["window_size"],
        "n_axes"       : len(CFG["columns"]),
    }
    quant_path = Path(CFG["output_dir"]) / "quant_params.json"
    with open(quant_path, "w") as f:
        json.dump(quant_info, f, indent=2)
    print(f"\n✅ Quant params → {quant_path}")
    return size_kb


# ─────────────────────────────────────────────────────────────
# 7b. SAVE TRAINING REPORT
# ─────────────────────────────────────────────────────────────
def save_training_report(report_info: dict,
                         n_train: int, n_val: int,
                         t_start: float,
                         tflite_size_kb: float | None = None):
    """
    Tổng hợp toàn bộ số liệu cần cho báo cáo vào training_report.txt.
    Đồng thời in đầy đủ ra terminal để theo dõi trực tiếp.
    """
    duration_min = (time.time() - t_start) / 60
    timestamp    = time.strftime("%Y-%m-%d %H:%M:%S")
    sep          = "═" * 55

    lines = [
        sep,
        "  GestureNet — TRAINING REPORT",
        f"  Generated  : {timestamp}",
        sep, "",
        "[ DATASET ]",
        f"  Classes      : {', '.join(CFG['classes'])}",
        f"  Window size  : {CFG['window_size']} samples ({CFG['window_size'] / 50:.1f}s @ 50Hz)",
        f"  Train windows: {n_train:,}  |  Val windows: {n_val:,}",
        f"  Augmentation : jitter / scaling / time_shift  (aug_factor={CFG['aug_factor']})",
        "",
        "[ TRAINING ]",
        f"  Epochs run   : {report_info['epochs_run']} / {CFG['epochs']}",
        f"  Training time: {duration_min:.1f} min",
        f"  Best val acc : {report_info['val_acc'] * 100:.2f}%",
        "",
        "[ CLASSIFICATION REPORT ]",
        report_info["report_str"],
        "[ PER-CLASS ACCURACY ]",
    ]

    for cls, acc in report_info["per_class_acc"].items():
        bar  = "█" * int(acc * 20)
        flag = "✅" if acc >= 0.90 else "⚠️ " if acc >= 0.75 else "❌"
        lines.append(f"  {flag} {cls:20s}: {acc * 100:5.1f}%  {bar}")

    lines.append("")

    if tflite_size_kb is not None:
        lines += [
            "[ MODEL SIZE ]",
            f"  TFLite int8  : {tflite_size_kb:.1f} KB",
            "",
        ]

    lines += [
        "[ OUTPUT FILES ]",
        "  training_curves.png    — Learning curves (Loss & Accuracy)",
        "  confusion_matrix.png   — Confusion matrix",
        "  per_class_accuracy.png — Per-class accuracy bar chart",
        "  model.tflite           — Quantized int8 model for ESP32-S3",
        "  scaler_params.json     — Z-score normalization params",
        "  quant_params.json      — TFLite quantization params",
        "",
        sep,
    ]

    report_text = "\n".join(lines)

    # In ra terminal
    print("\n" + "─" * 55)
    print("  📄 TRAINING REPORT SUMMARY")
    print("─" * 55)
    print(report_text)

    # Lưu file
    report_path = Path(CFG["output_dir"]) / "training_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\n✅ Training report → {report_path}")


# ─────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "█"*55)
    print("  GestureNet Training Pipeline  v1.0")
    print("  Target: ESP32-S3 | TFLite Micro")
    print("█"*55)

    # ── Load raw data (kèm group = id file) ──────────────────
    X_raw, y_raw, groups_raw = load_all_data()

    if len(X_raw) == 0:
        print("\n❌ Không có data nào được load! Kiểm tra lại thư mục data/")
        return

    # ── BƯỚC 1: Split THEO FILE — Val là file HOÀN TOÀN LẠ ───
    #    Dùng GroupShuffleSplit để KHÔNG window nào của cùng 1 file rơi vào
    #    cả train lẫn val. Tránh rò rỉ session (window chồng 50% + cùng buổi/
    #    người/hướng đeo) — vốn làm accuracy bị thổi phồng lên ~100% giả tạo.
    #    Augmentation & Normalization vẫn chạy sau, chỉ trên train.
    gss = GroupShuffleSplit(n_splits=1, test_size=CFG["test_size"],
                            random_state=CFG["seed"])
    train_idx, val_idx = next(gss.split(X_raw, y_raw, groups_raw))
    X_train_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
    y_train, y_val = y_raw[train_idx], y_raw[val_idx]
    n_val_files = len(np.unique(groups_raw[val_idx]))
    print(f"\n📂 Split theo file — Train: {X_train_raw.shape} | "
          f"Val: {X_val_raw.shape} ({n_val_files} file lạ)")
    _print_class_distribution(y_val, "Phân bố Val (theo file)")

    # ── BƯỚC 2: FIT SCALER trên RAW Train GỐC ────────────────
    #    ⚠️  Fit mean/std trên train GỐC (chưa rotate) để scaler khớp với dữ
    #    liệu firmware gặp thật (chip-hướng-vào). Lưu scaler_params.json ở đây.
    print("\n" + "="*55)
    print("  NORMALIZATION  (fit scaler = X_train_raw GỐC)")
    print("="*55)
    mean, std, _ = fit_scaler(X_train_raw)

    # ── BƯỚC 2b: ROTATION AUGMENT trên RAW (chỉ train; val giữ nguyên) ──
    #    Xoay proper trên raw accel+gyro → robust hướng đeo. Phải ở RAW-space
    #    (xoay sau Z-score sẽ méo vì mỗi trục bị scale khác nhau).
    X_train_raw, y_train = augment_rotation(X_train_raw, y_train)

    # ── BƯỚC 2c: APPLY scaler (transform cả train-đã-rotate lẫn val) ──
    X_train_norm = apply_scaler(X_train_raw, mean, std)
    X_val        = apply_scaler(X_val_raw,   mean, std)
    print(f"\n📂 After normalize — Train: {X_train_norm.shape} | Val: {X_val.shape}")

    # ── BƯỚC 3: Augment (jitter/scale/shift) CHỈ trên X_train_norm ──
    #    Trên dữ liệu đã Z-score — chỉ là nhiễu tương đối, không đụng scaler.
    #    Val không được augment — đại diện cho data thực tế firmware gửi lên.
    X_train, y_train = augment_dataset(X_train_norm, y_train)
    print(f"📂 After augment  — Train: {X_train.shape} | Val: {X_val.shape}")

    # ── Build & Train ────────────────────────────────────────
    model   = build_model(n_classes=len(CFG["classes"]))
    t_start = time.time()
    history = train(model, X_train, y_train, X_val, y_val)

    # ── Evaluate ─────────────────────────────────────────────
    val_acc, report_info = evaluate(model, X_val, y_val, history)

    # ── Export ───────────────────────────────────────────────
    tflite_kb = None
    if val_acc >= 0.85:
        tflite_kb = export_tflite(model, X_val)
    else:
        print(f"\n⚠️  Val accuracy {val_acc*100:.1f}% < 85% — chưa export TFLite.")
        print("   Gợi ý: thu thêm data hoặc tăng aug_factor trong CFG.")
        model.save(Path(CFG["output_dir"]) / "model_low_acc.keras")

    # ── Training Report ──────────────────────────────────────
    save_training_report(report_info,
                         n_train=len(X_train), n_val=len(X_val),
                         t_start=t_start,
                         tflite_size_kb=tflite_kb)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "█"*55)
    print(f"  ✅ DONE!  Val Accuracy = {val_acc*100:.2f}%")
    print(f"  Output → ./{CFG['output_dir']}/")
    print("█"*55 + "\n")


if __name__ == "__main__":
    main()