#ifndef DATA_CAPTURE_MODE
// main.cpp — rebuild/rep-detection
// ═══════════════════════════════════════════════════════════════
// Classify (single-core) + RepTracker (state machine + 3 detector PEAKDET).
//
// RepTracker (src/rep_tracker.h):
//   • IDLE→bài (conf>0.80): commit, +1 rep lead-in, khởi động 3 detector.
//   • ACTIVE: 3 detector always-warm đếm song song; switch bài bằng vote 3/4
//     trong ≤3 rep đầu (sửa sai khởi đầu, không mất rep); idle 3 window → chốt set.
//   • idle do CLASSIFIER loại; detector chỉ chạy khi ACTIVE.
//
// Detector PEAKDET verify C++≡Python 100% (training/sim_peakdet.py).
// Build BLE data-capture: -D DATA_CAPTURE_MODE (xem src/data_capture.cpp).
//
// Stage 2 — BLE GATT notify gói TUYỆT ĐỐI (src/ble_notify.h, doc Section 3):
//   notify mỗi inference cycle (~1.5s) + ngay khi đếm được rep.
// ═══════════════════════════════════════════════════════════════

#include "SparkFun_BMI270_Arduino_Library.h"
#include "ai_inference.h"
#include "ble_notify.h"
#include "norm_params.h"
#include "rep_tracker.h"
#include <Arduino.h>
#include <Wire.h>

BMI270 imu;

static const unsigned long SAMPLING_PERIOD_MS = 20;   // 50Hz

RepTracker tracker;
BleNotify  ble;

static uint8_t last_conf_pct = 0;   // confidence inference gần nhất (0-100)

// Gói tuyệt đối từ trạng thái tracker hiện tại (doc Section 3).
// IDLE: activity=idle, rep_count giữ kết quả set vừa chốt (sticky).
static BLEPacket_t makePacket(unsigned long now_ms) {
  BLEPacket_t p;
  bool active = tracker.committed() >= 0;
  p.activity    = active ? (uint8_t)tracker.committed() : 1;   // 1 = idle
  p.confidence  = last_conf_pct;
  int rep       = active ? tracker.repDisplay() : tracker.closedReps();
  p.rep_count   = rep > 255 ? 255 : (uint8_t)rep;
  p.set_no      = (uint8_t)tracker.setNo();
  p.timestamp_s = (uint16_t)(now_ms / 1000UL);
  return p;
}

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
  tracker.reset();
  ble.init();
  Serial.println("Classify + RepTracker + BLE notify. Bat dau...");
}

void loop() {
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis < SAMPLING_PERIOD_MS) return;
  previousMillis = currentMillis;

  imu.getSensorData();
  const float raw6[6] = {imu.data.accelX, imu.data.accelY, imu.data.accelZ,
                         imu.data.gyroX,  imu.data.gyroY,  imu.data.gyroZ};

  // Z-score mỗi sample (đơn vị chuẩn hoá, khớp norm_params) → detector.
  float z6[6];
  for (int j = 0; j < HAR_NUM_AXES; j++)
    z6[j] = (raw6[j] - HAR_MEAN[j]) * HAR_INV_STD[j];

  // ── Đếm rep (detector chạy bên trong tracker, chỉ khi ACTIVE) ──
  // z6 cho peakdet (bicep/lateral/tricep), raw6 cho world-vertical (shoulder).
  if (tracker.onSample(z6, raw6, currentMillis)) {
    if (tracker.committed() == 3)   // shoulder_press: in up/down ms để tune min_phase
      Serial.printf("REP! [%s] count: %d (up=%lums down=%lums)\n",
                    HAR_CLASS_NAMES[tracker.committed()], tracker.repDisplay(),
                    (unsigned long)tracker.lastShoulderUpMs(),
                    (unsigned long)tracker.lastShoulderDownMs());
    else
      Serial.printf("REP! [%s] count: %d (swing=%lums)\n",
                    HAR_CLASS_NAMES[tracker.committed()], tracker.repDisplay(),
                    (unsigned long)tracker.lastSwingMs());
    ble.send(makePacket(currentMillis));   // notify NGAY — UI nhảy rep tức thì
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

  // ── State machine: commit / switch / chốt set ──
  RepTracker::Event ev =
      tracker.onInference(result.class_index, result.confidence, currentMillis);
  switch (ev) {
    case RepTracker::COMMITTED:
      Serial.printf("[SET %d] bat dau %s (+1 rep lead-in)\n",
                    tracker.setNo(), HAR_CLASS_NAMES[tracker.committed()]);
      break;
    case RepTracker::SWITCHED:
      Serial.printf("[SET %d] sua sai -> %s (rep=%d)\n",
                    tracker.setNo(), HAR_CLASS_NAMES[tracker.committed()],
                    tracker.repDisplay());
      break;
    case RepTracker::SET_CLOSED:
      Serial.printf("[SET %d] xong: %d reps\n",
                    tracker.setNo(), tracker.closedReps());
      break;
    default:
      break;
  }

  // ── Notify định kỳ mỗi inference cycle (sau khi state machine cập nhật) ──
  last_conf_pct = (uint8_t)(result.confidence * 100.0f);
  ble.send(makePacket(currentMillis));
}
#endif
