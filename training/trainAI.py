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
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
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


def load_all_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load toàn bộ data từ CFG["data_dir"], chỉ lấy các class trong CFG["classes"].
    Trả về X shape (N, 150, 6) và y shape (N,).
    """
    print("\n" + "="*55)
    print("  BƯỚC 1: LOAD DATA & SLIDING WINDOW")
    print("="*55)

    all_windows, all_labels = [], []

    for label_idx, class_name in enumerate(CFG["classes"]):
        class_dir = Path(CFG["data_dir"]) / class_name
        if not class_dir.exists():
            print(f"\n❌ Không tìm thấy thư mục: {class_dir}")
            continue

        csv_files = sorted(class_dir.glob("*.csv"))
        if not csv_files:
            print(f"\n⚠️  Không có file CSV trong: {class_dir}")
            continue

        class_windows, class_labels = [], []
        print(f"\n[{label_idx}] {class_name} ({len(csv_files)} files):")

        for csv_path in csv_files:
            data = load_csv(csv_path)
            if data is None:
                continue

            wins, labs = sliding_window(data, label_idx)
            class_windows.extend(wins)
            class_labels.extend(labs)
            print(f"      ✅ {csv_path.name:30s} → {len(wins):3d} windows")

        print(f"      → Tổng: {len(class_windows)} windows cho class '{class_name}'")
        all_windows.extend(class_windows)
        all_labels.extend(class_labels)

    X = np.array(all_windows, dtype=np.float32)  # (N, 150, 6)
    y = np.array(all_labels,  dtype=np.int32)     # (N,)

    print(f"\n📦 Dataset gốc: {X.shape} | Labels: {y.shape}")
    _print_class_distribution(y, "Phân bố trước augmentation")
    return X, y


def _print_class_distribution(y: np.ndarray, title: str = ""):
    if title:
        print(f"\n  [{title}]")
    for i, cls in enumerate(CFG["classes"]):
        count = np.sum(y == i)
        bar = "█" * (count // 5)
        print(f"    {cls:20s}: {count:4d}  {bar}")


# ─────────────────────────────────────────────────────────────
# 2. Z-SCORE NORMALIZATION
# ─────────────────────────────────────────────────────────────
def normalize(X_train: np.ndarray,
              X_val:   np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Z-score normalization — đúng thứ tự, không Data Leakage.

    Nguyên tắc:
      - mean & std được FIT CHỈ trên X_train (Val không tham gia tính toán).
      - X_val chỉ được TRANSFORM bằng mean/std của Train.
      - Đảm bảo Val là "người lạ hoàn toàn" — phản ánh đúng năng lực thật.
    """
    # mean/std shape (6,) — chỉ tính từ Train set
    mean = X_train.reshape(-1, 6).mean(axis=0)
    std  = X_train.reshape(-1, 6).std(axis=0)
    std  = np.where(std == 0, 1e-8, std)   # tránh chia 0

    # Transform cả hai bằng CÙNG một scaler
    X_train_norm = (X_train - mean) / std
    X_val_norm   = (X_val   - mean) / std

    scaler_params = {
        "mean": mean.tolist(),
        "std" : std.tolist(),
        "axes": CFG["columns"],
    }

    scaler_path = Path(CFG["output_dir"]) / "scaler_params.json"
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=2)

    print(f"\n✅ Scaler params đã lưu: {scaler_path}")
    print(f"   mean = {[f'{v:.4f}' for v in mean.tolist()]}")
    print(f"   std  = {[f'{v:.4f}' for v in std.tolist()]}")
    print(f"   Fit trên {len(X_train):,} Train windows — Val không tham gia.")

    return X_train_norm, X_val_norm, scaler_params


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


def augment_axis_flip(window: np.ndarray) -> np.ndarray:
    """
    Lật ngẫu nhiên 1 trong 3 trục gia tốc (ax/ay/az).
    Giả lập đeo thiết bị ở tay trái hoặc góc khác.
    """
    axis = np.random.randint(0, 3)          # chỉ flip ax, ay, az
    result = window.copy()
    result[:, axis] = -result[:, axis]
    return result.astype(np.float32)


def augment_dataset(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Tạo aug_factor bản augment cho mỗi window gốc.
    Mỗi bản áp dụng 1–2 kỹ thuật ngẫu nhiên để tăng đa dạng.
    """
    print("\n" + "="*55)
    print("  BƯỚC 2: DATA AUGMENTATION")
    print("="*55)

    aug_fns = [augment_jitter, augment_scaling,
               augment_time_shift, augment_axis_flip]

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
             history: tf.keras.callbacks.History):

    print("\n" + "="*55)
    print("  BƯỚC 4: EVALUATION")
    print("="*55)

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

    print("\n📊 Classification Report:")
    print(classification_report(y_val, y_pred,
                                target_names=CFG["classes"]))

    val_acc = np.mean(y_pred == y_val)
    print(f"✅ Val Accuracy: {val_acc*100:.2f}%")

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

    return val_acc


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


# ─────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "█"*55)
    print("  GestureNet Training Pipeline  v1.0")
    print("  Target: ESP32-S3 | TFLite Micro")
    print("█"*55)

    # ── Load raw data ────────────────────────────────────────
    X_raw, y_raw = load_all_data()

    if len(X_raw) == 0:
        print("\n❌ Không có data nào được load! Kiểm tra lại thư mục data/")
        return

    # ── BƯỚC 1: Split TRƯỚC — Val được cô lập ngay từ đầu ───
    #    Augmentation và Normalization chưa được chạy ở đây.
    #    Val set sẽ không tham gia vào bất kỳ phép tính thống kê nào.
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw, y_raw,
        test_size=CFG["test_size"],
        random_state=CFG["seed"],
        stratify=y_raw,
    )
    print(f"\n📂 Raw split — Train: {X_train_raw.shape} | Val: {X_val_raw.shape}")

    # ── BƯỚC 2: Normalize — fit trên RAW Train, transform cả hai ──
    #
    #    ⚠️  PHẢI normalize TRƯỚC khi augment, không phải sau.
    #    Lý do: augment_axis_flip lật dấu trục Z (1.0g → -1.0g).
    #    Nếu normalize SAU augment, mean_Z bị kéo về ~0 (nửa +1, nửa -1),
    #    ghi nhận sai vào scaler_params.json → firmware tính offset sai → model sụp.
    #    Normalize trên X_train_RAW đảm bảo mean/std phản ánh đúng vật lý cảm biến.
    print("\n" + "="*55)
    print("  NORMALIZATION  (fit=X_train_raw, trước augmentation)")
    print("="*55)
    X_train_norm, X_val, scaler_params = normalize(X_train_raw, X_val_raw)
    print(f"\n📂 After normalize — Train: {X_train_norm.shape} | Val: {X_val.shape}")

    # ── BƯỚC 3: Augment CHỈ trên X_train_norm ────────────────
    #    Jitter / Scaling / TimeShift / AxisFlip trên dữ liệu đã Z-score
    #    hoàn toàn hợp lệ — chỉ là nhiễu tương đối, không ảnh hưởng mean/std thật.
    #    Val không được augment — đại diện cho data thực tế firmware gửi lên.
    X_train, y_train = augment_dataset(X_train_norm, y_train)
    print(f"📂 After augment  — Train: {X_train.shape} | Val: {X_val.shape}")

    # ── Build & Train ────────────────────────────────────────
    model = build_model(n_classes=len(CFG["classes"]))
    history = train(model, X_train, y_train, X_val, y_val)

    # ── Evaluate ─────────────────────────────────────────────
    val_acc = evaluate(model, X_val, y_val, history)

    # ── Export ───────────────────────────────────────────────
    if val_acc >= 0.85:
        export_tflite(model, X_val)
    else:
        print(f"\n⚠️  Val accuracy {val_acc*100:.1f}% < 85% — chưa export TFLite.")
        print("   Gợi ý: thu thêm data hoặc tăng aug_factor trong CFG.")
        # Vẫn lưu Keras model để debug
        model.save(Path(CFG["output_dir"]) / "model_low_acc.keras")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "█"*55)
    print(f"  ✅ DONE!  Val Accuracy = {val_acc*100:.2f}%")
    print(f"  Output → ./{CFG['output_dir']}/")
    print("█"*55 + "\n")


if __name__ == "__main__":
    main()