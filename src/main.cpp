#ifndef DATA_CAPTURE_MODE
// main.cpp — rebuild/rep-detection
// ═══════════════════════════════════════════════════════════════
// Classify (single-core) + REP DETECTOR PEAKDET (Billauer, delta).
//
// Gate đơn giản: classifier ra bài X (conf>0.80) → đếm rep của X (trục + delta
// của X); ra idle → chốt set (in tổng, reset). idle do CLASSIFIER loại bỏ,
// detector KHÔNG tự lọc idle (xem rep_detector.h / sim_peakdet.py).
//
// Detector = RepDetector (peakdet) — mirror 1:1 training/sim_peakdet.py, đã
// verify C++≡Python 100% trên toàn bộ CSV. Tune trục/delta ở EXCFG bên dưới
// (đồng bộ training/rep_config.py).
//
// Build BLE data-capture: bật cờ -D DATA_CAPTURE_MODE (platformio.ini) → file
// này tắt, src/data_capture.cpp (BLE NUS + SPIFFS) chạy thay.
// ═══════════════════════════════════════════════════════════════

#include "SparkFun_BMI270_Arduino_Library.h"
#include "ai_inference.h"
#include "norm_params.h"
#include "rep_detector.h"
#include <Arduino.h>
#include <Wire.h>

BMI270 imu;

static const unsigned long SAMPLING_PERIOD_MS = 20;   // 50Hz

// ── Cấu hình per-bài (đồng bộ training/rep_config.py) ──────────
// Index theo class: 0 bicep, 1 idle, 2 lateral, 3 shoulder, 4 tricep
// Trục: ax=0 ay=1 az=2 gx=3 gy=4 gz=5
//   bicep/lateral = gy(4) ; shoulder/tricep = ay(1)
struct ExCfg {
    uint8_t  axis;
    float    delta;
    uint32_t min_gap_ms;
    uint8_t  smooth_win;
    uint32_t reset_timeout_ms;
    float    min_abs;
};
static const int IDLE_IDX = 1;
static const ExCfg EXCFG[HAR_NUM_CLASSES] = {
    {4, 1.0f, 600, 5, 3000, 0.0f},  // [0] bicep_curl     gy
    {0, 0.0f,   0, 1,    0, 0.0f},  // [1] idle (không dùng)
    {4, 1.0f, 600, 5, 3000, 0.0f},  // [2] lateral_raise  gy
    {1, 0.5f, 600, 5, 3000, 0.0f},  // [3] shoulder_press ay (tín hiệu yếu)
    {1, 0.9f, 600, 5, 3000, 0.0f},  // [4] tricep_ext     ay
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

static unsigned long previousMillis = 0;

void setup() {
  Serial.begin(115200);
  delay(3000);

  Wire.begin(D4, D5);
  delay(100);

  if (imu.beginI2C(0x68) != BMI2_OK) {
    Serial.println("LOI: Khong the khoi tao BMI270!");
    while (1)
      ;
  }
  if (!harInit()) {
    Serial.println("LOI: Khoi tao AI (TFLite) that bai!");
    while (1)
      ;
  }
  harPrintModelInfo();
  Serial.println("Classify + rep detector (PEAKDET). Bat dau...");
}

void loop() {
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis < SAMPLING_PERIOD_MS) return;
  previousMillis = currentMillis;

  imu.getSensorData();
  const float raw6[6] = {imu.data.accelX, imu.data.accelY, imu.data.accelZ,
                         imu.data.gyroX,  imu.data.gyroY,  imu.data.gyroZ};

  // Z-score mỗi sample (dùng cho detector — đơn vị chuẩn hoá, khớp norm_params).
  float z6[6];
  for (int j = 0; j < HAR_NUM_AXES; j++)
    z6[j] = (raw6[j] - HAR_MEAN[j]) * HAR_INV_STD[j];

  // ── Đếm rep: chỉ khi đang trong 1 bài tập ──
  if (cur_ex >= 0 && detector.update(z6, currentMillis)) {
    rep_count++;
    Serial.printf("REP! [%s] count: %d\n", HAR_CLASS_NAMES[cur_ex], rep_count);
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
    Serial.printf("%-15s %5.1f%%\n", result.class_name,
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
      detector.configure(c.axis, c.delta, c.min_gap_ms, c.smooth_win,
                         c.reset_timeout_ms, c.min_abs);
      Serial.printf("[SET] bat dau %s (truc=%d, delta=%.2f)\n",
                    HAR_CLASS_NAMES[cur_ex], c.axis, c.delta);
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
