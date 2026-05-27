#pragma once
// ai_inference.h
// ═══════════════════════════════════════════════════════════════
// HAR (Human Activity Recognition) Inference Engine
// Target  : ESP32-S3 + TFLite Micro
// Model   : 1D-CNN int8 quantized
// Input   : Ring buffer 150 samples × 6 axes (ax,ay,az,gx,gy,gz)
// Pipeline: Z-Score (float) → Quantize (int8) → Inference → Dequantize
// ═══════════════════════════════════════════════════════════════

#include <cstddef>
#include <cstdint>

// ─── Kết quả trả về sau mỗi lần inference ────────────────────
struct HarResult {
  int class_index;            // index của class thắng [0, NUM_CLASSES-1]
  float confidence;           // xác suất [0.0, 1.0]
  float scores[5];            // softmax scores tất cả các class
  const char *class_name;     // con trỏ tới tên class (không cần free)
  uint32_t inference_time_ms; // thời gian inference (debug)
};

// ─── Trạng thái engine ───────────────────────────────────────
enum class HarStatus {
  OK = 0,
  NOT_READY = 1,   // chưa gọi harInit() hoặc init thất bại
  INVOKE_FAIL = 2, // TFLite interpreter invoke thất bại
};

// ─────────────────────────────────────────────────────────────
// API công khai — gọi theo thứ tự: Init → RunInference (lặp lại)
// ─────────────────────────────────────────────────────────────

/**
 * Khởi tạo TFLite Micro interpreter và cấp phát tensor arena.
 * Gọi một lần duy nhất trong setup().
 *
 * @return true nếu thành công, false nếu model hoặc arena lỗi.
 */
bool harInit();

/**
 * Chạy một lượt inference từ Ring Buffer hiện tại.
 *
 * @param ring_buf  Con trỏ tới mảng float [HAR_WINDOW_SIZE][HAR_NUM_AXES]
 *                  (150 samples × 6 axes, thứ tự: ax,ay,az,gx,gy,gz)
 * @param result    Output: struct HarResult sẽ được ghi vào đây.
 * @return          HarStatus::OK nếu thành công.
 */
HarStatus harRunInference(const float ring_buf[][6], HarResult &result);

/**
 * In thông tin model và thông số norm ra Serial (dùng để debug).
 * Gọi sau harInit().
 */
void harPrintModelInfo();