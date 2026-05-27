// ai_inference.cpp
// ═══════════════════════════════════════════════════════════════
// HAR Inference Engine — Implementation
// Luồng xử lý cho mỗi lần inference:
//
//   ring_buf[150][6]  (float, raw IMU)
//       │
//       ▼ Z-Score: z = (x - mean) * inv_std
//   z_buf[150][6]    (float, normalized)
//       │
//       ▼ Quantize: q = clamp(round(z / inp_scale + inp_zp), -128, 127)
//   input tensor     (int8, shape: 1×150×6)
//       │
//       ▼ TFLite Micro Interpreter invoke()
//   output tensor    (int8, shape: 1×5)
//       │
//       ▼ Dequantize: prob = (q - out_zp) * out_scale
//   scores[5]        (float, softmax probabilities)
//       │
//       ▼ argmax → class_index, confidence
// ═══════════════════════════════════════════════════════════════

#include "ai_inference.h"
#include "model_data.h"
#include "norm_params.h"

// TFLite Micro headers (từ thư viện tflite-micro-arduino-examples)
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <Arduino.h> // millis(), Serial
#include <algorithm> // std::max, std::min
#include <cmath>     // roundf

// ─────────────────────────────────────────────────────────────
// TENSOR ARENA
// 20KB tĩnh — đủ cho mô hình 1D-CNN (Peak SRAM ~12-18KB).
// Nếu harInit() thất bại với lỗi kAllocateTensors, tăng lên 24KB.
// alignas(16) bắt buộc: TFLite Micro yêu cầu alignment để SIMD trên S3.
// ─────────────────────────────────────────────────────────────
static constexpr size_t TENSOR_ARENA_SIZE = 20 * 1024;
static uint8_t tensor_arena[TENSOR_ARENA_SIZE] alignas(16);

// ─────────────────────────────────────────────────────────────
// TFLite Micro objects (static — tồn tại suốt vòng đời firmware)
// ─────────────────────────────────────────────────────────────
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter *error_reporter = &micro_error_reporter;

// Op resolver: khai báo ĐÚNG và ĐỦ các ops mà model sử dụng.
// 1D-CNN sau khi convert TFLite sẽ dùng:
//   Conv1D     → CONV_2D  (TFLite reshape 1D thành 2D nội bộ)
//   MaxPool1D  → MAX_POOL_2D
//   Dense      → FULLY_CONNECTED
//   BatchNorm  → đã được fold vào Conv trong quá trình quantization
//   ReLU       → fused trong CONV_2D / FULLY_CONNECTED
//   Softmax    → SOFTMAX
//   Flatten    → RESHAPE
// Giải pháp: Tạm thời dùng AllOpsResolver để không bị lỗi thiếu toán tử.
// Trên ESP32-S3 có tới 4MB-8MB Flash, nên tốn thêm 30KB cho AllOpsResolver là
// hoàn toàn xứng đáng.
static tflite::AllOpsResolver op_resolver;

static const tflite::Model *tfl_model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input_tensor = nullptr;
static TfLiteTensor *output_tensor = nullptr;

static bool g_initialized = false;

// ─────────────────────────────────────────────────────────────
// INTERNAL HELPER: Quantize một giá trị float → int8
// Công thức: q = round(f / scale + zero_point), clamp [-128, 127]
// ─────────────────────────────────────────────────────────────
static inline int8_t quantize(float f, float scale, int32_t zero_point) {
  float q = roundf(f / scale) + static_cast<float>(zero_point);
  // clamp về [-128, 127] tránh overflow khi ép kiểu
  if (q < -128.0f)
    q = -128.0f;
  if (q > 127.0f)
    q = 127.0f;
  return static_cast<int8_t>(q);
}

// ─────────────────────────────────────────────────────────────
// INTERNAL HELPER: Dequantize int8 → float probability
// Công thức: prob = (q - zero_point) * scale
// ─────────────────────────────────────────────────────────────
static inline float dequantize(int8_t q, float scale, int32_t zero_point) {
  return (static_cast<float>(q) - static_cast<float>(zero_point)) * scale;
}

// ─────────────────────────────────────────────────────────────
// harInit()
// ─────────────────────────────────────────────────────────────
bool harInit() {
  Serial.println("[HAR] Initializing TFLite Micro...");

  // 1. Parse model từ mảng byte trong Flash
  tfl_model = tflite::GetModel(gesture_model_tflite);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[HAR] ❌ Schema version mismatch: model=%d, runtime=%d\n",
                  tfl_model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  // 2. Đăng ký các ops
  // (Đã dùng AllOpsResolver nên TFLite sẽ tự động load MỌI toán tử cần thiết)

  // 3. Khởi tạo interpreter
  //    Dùng placement new vì MicroInterpreter không thể dùng dynamic alloc
  static uint8_t interpreter_buf[sizeof(tflite::MicroInterpreter)];
  interpreter = new (interpreter_buf) tflite::MicroInterpreter(
      tfl_model, op_resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);

  // 4. Cấp phát tensors trong arena
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println(
        "[HAR] ❌ AllocateTensors() failed — thử tăng TENSOR_ARENA_SIZE");
    return false;
  }

  // 5. Lưu con trỏ tensor để dùng sau (tránh gọi interpreter->input() mỗi lần)
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  // 6. Kiểm tra shape tensor đúng kỳ vọng
  //    Input  : [1, 150, 6] int8
  //    Output : [1, 5]      int8
  bool shape_ok = (input_tensor->dims->size == 3) &&
                  (input_tensor->dims->data[1] == HAR_WINDOW_SIZE) &&
                  (input_tensor->dims->data[2] == HAR_NUM_AXES) &&
                  (output_tensor->dims->data[1] == HAR_NUM_CLASSES);

  if (!shape_ok) {
    Serial.printf("[HAR] ❌ Tensor shape mismatch!\n"
                  "       Input : [%d, %d, %d] (expect [1, %d, %d])\n"
                  "       Output: [%d, %d]     (expect [1, %d])\n",
                  input_tensor->dims->data[0], input_tensor->dims->data[1],
                  input_tensor->dims->data[2], HAR_WINDOW_SIZE, HAR_NUM_AXES,
                  output_tensor->dims->data[0], output_tensor->dims->data[1],
                  HAR_NUM_CLASSES);
    return false;
  }

  g_initialized = true;

  size_t used_bytes = interpreter->arena_used_bytes();
  Serial.printf("[HAR] ✅ Init OK  |  Arena used: %u / %u bytes\n", used_bytes,
                TENSOR_ARENA_SIZE);
  return true;
}

// ─────────────────────────────────────────────────────────────
// harRunInference()
// ─────────────────────────────────────────────────────────────
HarStatus harRunInference(const float ring_buf[][6], HarResult &result) {
  if (!g_initialized) {
    return HarStatus::NOT_READY;
  }

  uint32_t t_start = millis();

  // ── STAGE 1: Z-Score + Quantize → Input Tensor ───────────
  //
  // Với mỗi sample i và mỗi axis j:
  //   z   = (ring_buf[i][j] - HAR_MEAN[j]) * HAR_INV_STD[j]
  //   q   = round(z / HAR_INPUT_SCALE) + HAR_INPUT_ZERO_POINT
  //   q   = clamp(q, -128, 127)
  //
  // Layout input tensor: flat array, index = i*HAR_NUM_AXES + j
  // (TFLite lưu tensor theo hàng ngang, giống C row-major)

  int8_t *inp_data = input_tensor->data.int8;

  for (int i = 0; i < HAR_WINDOW_SIZE; i++) {
    for (int j = 0; j < HAR_NUM_AXES; j++) {
      float z = (ring_buf[i][j] - HAR_MEAN[j]) * HAR_INV_STD[j];
      inp_data[i * HAR_NUM_AXES + j] =
          quantize(z, HAR_INPUT_SCALE, HAR_INPUT_ZERO_POINT);
    }
  }

  // ── STAGE 2: Inference ───────────────────────────────────
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    return HarStatus::INVOKE_FAIL;
  }

  // ── STAGE 3: Dequantize Output → Softmax Probabilities ───
  //
  // output tensor: int8 shape [1, HAR_NUM_CLASSES]
  // prob[k] = (out_int8[k] - HAR_OUTPUT_ZERO_POINT) * HAR_OUTPUT_SCALE
  //
  // Lỗi thường gặp: quên bước này → scores toàn số âm hoặc > 1.0

  const int8_t *out_data = output_tensor->data.int8;

  float best_score = -1.0f;
  int best_idx = 0;

  for (int k = 0; k < HAR_NUM_CLASSES; k++) {
    float prob =
        dequantize(out_data[k], HAR_OUTPUT_SCALE, HAR_OUTPUT_ZERO_POINT);
    result.scores[k] = prob;
    if (prob > best_score) {
      best_score = prob;
      best_idx = k;
    }
  }

  // ── STAGE 4: Điền kết quả ────────────────────────────────
  result.class_index = best_idx;
  result.confidence = best_score;
  result.class_name = HAR_CLASS_NAMES[best_idx];
  result.inference_time_ms = millis() - t_start;

  return HarStatus::OK;
}

// ─────────────────────────────────────────────────────────────
// harPrintModelInfo()  — debug helper
// ─────────────────────────────────────────────────────────────
void harPrintModelInfo() {
  Serial.println("\n[HAR] ══ Model Info ══════════════════════════════");
  Serial.printf("       Model size      : %u bytes (%.1f KB)\n",
                gesture_model_tflite_len, gesture_model_tflite_len / 1024.0f);
  Serial.printf("       Window size     : %d samples\n", HAR_WINDOW_SIZE);
  Serial.printf("       Axes            : %d\n", HAR_NUM_AXES);
  Serial.printf("       Classes         : %d\n", HAR_NUM_CLASSES);
  Serial.printf("       Input  quant    : scale=%.6f  zp=%d\n", HAR_INPUT_SCALE,
                HAR_INPUT_ZERO_POINT);
  Serial.printf("       Output quant    : scale=%.6f  zp=%d\n",
                HAR_OUTPUT_SCALE, HAR_OUTPUT_ZERO_POINT);

  Serial.println("       Z-Score params  : [axis]  mean      inv_std");
  const char *axes_names[] = {"ax", "ay", "az", "gx", "gy", "gz"};
  for (int j = 0; j < HAR_NUM_AXES; j++) {
    Serial.printf("                         [%s]   %+.4f   %.4f\n",
                  axes_names[j], HAR_MEAN[j], HAR_INV_STD[j]);
  }

  Serial.println("       Classes:");
  for (int k = 0; k < HAR_NUM_CLASSES; k++) {
    Serial.printf("                         [%d] %s\n", k, HAR_CLASS_NAMES[k]);
  }
  Serial.println("[HAR] ═══════════════════════════════════════════════\n");
}