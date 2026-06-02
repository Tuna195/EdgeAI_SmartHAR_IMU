#ifndef DATA_CAPTURE_MODE
// main.cpp — NỀN MỚI (rebuild/rep-detection)
// ═══════════════════════════════════════════════════════════════
// Bắt đầu LẠI TỪ ĐẦU: chỉ classify (single-core), KHÔNG rep counter.
//
// Luồng: đọc IMU @50Hz → ring buffer → mỗi 1.5s (STRIDE) chạy inference
//        → in class thắng + confidence + ĐIỂM CẢ 5 BÀI.
//
// In đủ 5 điểm để thấy rõ model phân vân giữa bài nào (vd bicep↔tricep),
// phục vụ KIỂM NGHIỆM CLASSIFY trên board trước khi xây lại rep detection.
//
// Rep detection sẽ được thêm DẦN lên nền này (giữ mọi thứ tối giản, dễ nhìn).
// ═══════════════════════════════════════════════════════════════

#include "SparkFun_BMI270_Arduino_Library.h"
#include "ai_inference.h"
#include "norm_params.h"
#include <Arduino.h>
#include <Wire.h>

BMI270 imu;

static const unsigned long SAMPLING_PERIOD_MS = 20;   // 50Hz
static unsigned long previousMillis = 0;

// Ring buffer RAW (model train trên raw → Z-score trong ai_inference).
static float ring_buf[HAR_WINDOW_SIZE][HAR_NUM_AXES];
static int   head_index     = 0;
static int   sample_count   = 0;
static int   stride_counter = 0;
static const int STRIDE = HAR_WINDOW_SIZE / 2;        // 75 mẫu (1.5s)

void setup() {
  Serial.begin(115200);
  delay(3000);

  Wire.begin(D4, D5);
  delay(100);

  if (imu.beginI2C(0x68) != BMI2_OK) {
    Serial.println("❌ LỖI: Không thể khởi tạo BMI270!");
    while (1)
      ;
  }
  if (!harInit()) {
    Serial.println("❌ LỖI: Khởi tạo AI (TFLite) thất bại!");
    while (1)
      ;
  }
  harPrintModelInfo();
  Serial.println("✅ CLASSIFY-ONLY (nen moi) | in moi 1.5s: class + conf + diem 5 bai");
}

void loop() {
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis < SAMPLING_PERIOD_MS) return;
  previousMillis = currentMillis;

  imu.getSensorData();
  ring_buf[head_index][0] = imu.data.accelX;
  ring_buf[head_index][1] = imu.data.accelY;
  ring_buf[head_index][2] = imu.data.accelZ;
  ring_buf[head_index][3] = imu.data.gyroX;
  ring_buf[head_index][4] = imu.data.gyroY;
  ring_buf[head_index][5] = imu.data.gyroZ;
  head_index = (head_index + 1) % HAR_WINDOW_SIZE;

  bool buffer_full = (sample_count < HAR_WINDOW_SIZE)
                         ? (++sample_count == HAR_WINDOW_SIZE)
                         : true;
  if (!buffer_full || (++stride_counter < STRIDE)) return;
  stride_counter = 0;

  // Flatten ring buffer → mảng tuyến tính (cũ nhất → mới nhất).
  static float flat_buf[HAR_WINDOW_SIZE][HAR_NUM_AXES];
  int older = HAR_WINDOW_SIZE - head_index;
  memcpy(&flat_buf[0][0], &ring_buf[head_index][0],
         older * HAR_NUM_AXES * sizeof(float));
  if (head_index > 0)
    memcpy(&flat_buf[older][0], &ring_buf[0][0],
           head_index * HAR_NUM_AXES * sizeof(float));

  HarResult result;
  if (harRunInference(flat_buf, result) != HarStatus::OK) {
    Serial.println("LOI Inference!");
    return;
  }

  Serial.printf("\xF0\x9F\x8F\x8B %-15s %5.1f%% | ", result.class_name,
                result.confidence * 100.0f);
  for (int k = 0; k < HAR_NUM_CLASSES; k++)
    Serial.printf("%s:%4.1f%% ", HAR_CLASS_NAMES[k], result.scores[k] * 100.0f);
  Serial.printf("| %dms\n", result.inference_time_ms);
}
#endif
