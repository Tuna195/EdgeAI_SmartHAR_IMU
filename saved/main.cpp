#ifndef DATA_CAPTURE_MODE
// main.cpp — rebuild/rep-detection
// ═══════════════════════════════════════════════════════════════
// Classify (single-core) + REP DETECTOR TỐI GIẢN (Schmitt ngưỡng cố định).
//
// Gate đơn giản nhất: classifier ra bài X → đếm rep của X (trục gyro của X);
// ra idle → chốt set (in tổng, reset). Chưa vote/pending gì cả — thêm sau
// nếu cần. Mục tiêu: nhìn rõ detector chạy đúng không.
// ═══════════════════════════════════════════════════════════════

#include "SparkFun_BMI270_Arduino_Library.h"
#include "ai_inference.h"
#include "norm_params.h"
#include "rep_detector.h"
#include <Arduino.h>
#include <Wire.h>

BMI270 imu;

static const unsigned long SAMPLING_PERIOD_MS = 20;   // 50Hz
static unsigned long previousMillis = 0;

// ── Cấu hình per-bài (dễ tune): trục gyro + ngưỡng T + debounce ──
// Index theo class: 0 bicep, 1 idle, 2 lateral, 3 shoulder, 4 tricep
// Trục: gx=3, gy=4, gz=5  (bicep/lateral=gy, shoulder=gx, tricep=gz)
struct ExCfg { uint8_t axis; float T; uint32_t min_gap_ms; };
static const int IDLE_IDX = 1;
// T = min(T_trái, T_phải) suy từ data (analyze_axes.py) → tay yếu không miss.
static const ExCfg EXCFG[HAR_NUM_CLASSES] = {
    {4, 0.5f, 2000},  // [0] bicep_curl    gy  (trái yếu → T thấp)
    {0, 0.0f,    0},  // [1] idle (không dùng)
    {4, 0.8f,  850},  // [2] lateral_raise gy
    {3, 0.6f, 1910},  // [3] shoulder_press gx (tín hiệu yếu)
    {5, 1.1f,  970},  // [4] tricep_ext    gz
};
static inline bool isExercise(int c) {
    return c >= 0 && c < HAR_NUM_CLASSES && c != IDLE_IDX;
}

RepDetector detector;
int cur_ex    = -1;   // bài đang đếm (-1 = không)
int rep_count = 0;

// ── Ring buffer cho inference (RAW) ──
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
  Serial.println("✅ Classify + rep detector (nguong co dinh). Bat dau...");
}

void loop() {
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis < SAMPLING_PERIOD_MS) return;
  previousMillis = currentMillis;

  imu.getSensorData();
  const float raw6[6] = {imu.data.accelX, imu.data.accelY, imu.data.accelZ,
                         imu.data.gyroX,  imu.data.gyroY,  imu.data.gyroZ};

  // Z-score mỗi sample (dùng cho detector — đơn vị chuẩn hoá).
  float z6[6];
  for (int j = 0; j < HAR_NUM_AXES; j++)
    z6[j] = (raw6[j] - HAR_MEAN[j]) * HAR_INV_STD[j];

  // ── Đếm rep: chỉ khi đang trong 1 bài tập ──
  if (cur_ex >= 0 && detector.update(z6, currentMillis)) {
    rep_count++;
    Serial.printf("\xF0\x9F\x92\xAA REP! [%s] count: %d\n",
                  HAR_CLASS_NAMES[cur_ex], rep_count);
  }

  // ── Nạp ring buffer (RAW) cho inference ──
  for (int j = 0; j < HAR_NUM_AXES; j++) ring_buf[head_index][j] = raw6[j];
  head_index = (head_index + 1) % HAR_WINDOW_SIZE;
  bool buffer_full = (sample_count < HAR_WINDOW_SIZE)
                         ? (++sample_count == HAR_WINDOW_SIZE)
                         : true;
  if (!buffer_full || (++stride_counter < STRIDE)) return;
  stride_counter = 0;

  // Flatten → inference
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

  if (result.confidence > 0.80f)
    Serial.printf("\xF0\x9F\x8F\x8B %-15s %5.1f%%\n", result.class_name,
                  result.confidence * 100.0f);

  // ── Gate đơn giản: đổi bài / chốt set theo classifier ──
  if (result.confidence <= 0.80f) return;   // không chắc → bỏ qua

  if (isExercise(result.class_index)) {
    if (result.class_index != cur_ex) {     // bắt đầu bài mới
      if (cur_ex >= 0)
        Serial.printf("[SET] doi bai (%s: %d reps)\n",
                      HAR_CLASS_NAMES[cur_ex], rep_count);
      cur_ex    = result.class_index;
      rep_count = 0;
      const ExCfg &c = EXCFG[cur_ex];
      detector.configure(c.axis, c.T, c.min_gap_ms);
      detector.reset();
      Serial.printf("[SET] bat dau %s (truc gyro=%d)\n",
                    HAR_CLASS_NAMES[cur_ex], c.axis);
    }
  } else {                                   // idle → chốt set
    if (cur_ex >= 0) {
      Serial.printf("[SET] xong %s: %d reps\n", HAR_CLASS_NAMES[cur_ex],
                    rep_count);
      cur_ex    = -1;
      rep_count = 0;
    }
  }
}
#endif
