#if !defined(DATA_CAPTURE_MODE) && !defined(CLASSIFY_ONLY_MODE)
#include "SparkFun_BMI270_Arduino_Library.h"
#include "ai_inference.h"
#include "exercise_config.h"
#include "norm_params.h"
#include "rep_counter.h"
#include <Arduino.h>
#include <Wire.h>

// Bật log gỡ lỗi detector: in mỗi lần bắn đỉnh (axis/envelope/T/dt) để tune.
// Đặt 0 khi chạy bình thường.
#define REP_DEBUG 1

// ── Warm-up khi khởi động ────────────────────────────────────────
// Gyro BMI270 có bias/độ trôi lúc mới boot → vài giây đầu trục gx/gy nhiễu
// → model đọc nhầm (thường ra shoulder_press). Khi thu data, record chỉ bắt đầu
// khi user bấm lệnh (đã đợi ổn định) nên transient boot KHÔNG nằm trong tập train.
// Giải pháp: firmware vẫn lấy mẫu (detector ấm) nhưng CHƯA phân loại/đếm cho tới
// khi sensor settle — phát hiện bằng "idle ổn định liên tiếp", hoặc timeout.
static constexpr int      ARM_IDLE_STREAK = 3;        // 3 inference idle liên tiếp (~4.5s) → arm
static constexpr uint32_t ARM_TIMEOUT_MS  = 15000UL;  // fallback: arm sau 15s

// ═══════════════════════════════════════════════════════════════
// KIẾN TRÚC DUAL-CORE (ESP32-S3)
// ───────────────────────────────────────────────────────────────
//  Core 1 (APP) — SamplerTask (50Hz chuẩn): đọc IMU → Z-score → 3×RepDetector
//      (gx/gy/gz, LUÔN ấm) → fire thì gửi PEAK(axis); mỗi STRIDE gửi WINDOW.
//      SỞ HỮU: imu/Wire, 3 RepDetector, ring buffer.
//  Core 0 (PRO) — InferenceTask: TFLite + RepTracker (vote chọn bài + đếm).
//      SỞ HỮU: interpreter, RepTracker.
//  Giao tiếp: 1 queue Sampler→Inference (PEAK/WINDOW, đúng thứ tự thời gian).
//      Không còn đẩy trục về Sampler — detector cố định đa trục nên bỏ portMUX.
// ═══════════════════════════════════════════════════════════════

BMI270 imu;

static constexpr unsigned long SAMPLING_PERIOD_MS = 20;      // 50Hz
static constexpr int           STRIDE = HAR_WINDOW_SIZE / 2;  // 75 mẫu

// ── Message Sampler → Inference (1 queue, đúng thứ tự thời gian) ──
enum class MsgType : uint8_t { PEAK = 0, WINDOW = 1 };
struct CoreMsg {
  MsgType  type;
  uint32_t ts_ms;
  uint8_t  axis;     // PEAK: trục detector đã fire
  float    env;      // PEAK: envelope (debug)
  float    T;        // PEAK: ngưỡng (debug)
  uint8_t  buf_idx;  // WINDOW: chỉ số ping-pong buffer
};
static QueueHandle_t q_msg = nullptr;

// ── Ping-pong flat buffers (Sampler ghi, Inference đọc) ──
static float flat_buf[2][HAR_WINDOW_SIZE][HAR_NUM_AXES];

// ═══════════════════════════════════════════════════════════════
// SAMPLER TASK  (Core 1 / APP) — 50Hz, 3 detector trục gyro cố định
// ═══════════════════════════════════════════════════════════════
static void samplerTask(void *arg) {
  // 3 detector cố định: gx(3)→shoulder, gy(4)→bicep/lateral, gz(5)→tricep.
  // LUÔN chạy & ấm → không còn "sai trục lúc đầu" / warm-up gap khi đổi bài.
  static RepDetector det_gx, det_gy, det_gz;
  static float ring[HAR_WINDOW_SIZE][HAR_NUM_AXES];
  RepDetector *dets[3] = {&det_gx, &det_gy, &det_gz};

  det_gx.configure(3);
  det_gy.configure(4);
  det_gz.configure(5);

  int     head       = 0;
  int     count      = 0;
  int     stride_ctr = 0;
  uint8_t pp         = 0;
  TickType_t last_wake = xTaskGetTickCount();
  const TickType_t period = pdMS_TO_TICKS(SAMPLING_PERIOD_MS);

  for (;;) {
    vTaskDelayUntil(&last_wake, period);   // nhịp 50Hz chính xác, không trôi
    uint32_t now = millis();

    imu.getSensorData();
    const float raw6[6] = {imu.data.accelX, imu.data.accelY, imu.data.accelZ,
                           imu.data.gyroX,  imu.data.gyroY,  imu.data.gyroZ};
    float norm6[6];
    for (int j = 0; j < HAR_NUM_AXES; j++)
      norm6[j] = (raw6[j] - HAR_MEAN[j]) * HAR_INV_STD[j];

    // 3 detector chạy tuần tự (cực rẻ ~vài chục ops mỗi cái) → fire thì gửi PEAK.
    for (int k = 0; k < 3; k++) {
      if (dets[k]->update(norm6)) {
        CoreMsg m{};
        m.type  = MsgType::PEAK;
        m.ts_ms = now;
        m.axis  = dets[k]->getAxis();
        m.env   = dets[k]->getEnvelope();
        m.T     = dets[k]->getThreshold();
        xQueueSend(q_msg, &m, 0);
      }
    }

    // Nạp RAW vào ring buffer (O(1)).
    for (int j = 0; j < HAR_NUM_AXES; j++) ring[head][j] = raw6[j];
    head = (head + 1) % HAR_WINDOW_SIZE;
    bool full = (count < HAR_WINDOW_SIZE) ? (++count == HAR_WINDOW_SIZE) : true;

    if (full && (++stride_ctr >= STRIDE)) {
      stride_ctr = 0;
      int older = HAR_WINDOW_SIZE - head;     // mẫu CŨ NHẤT: head..hết mảng
      memcpy(&flat_buf[pp][0][0], &ring[head][0],
             older * HAR_NUM_AXES * sizeof(float));
      if (head > 0)                            // mẫu MỚI NHẤT: 0..head-1
        memcpy(&flat_buf[pp][older][0], &ring[0][0],
               head * HAR_NUM_AXES * sizeof(float));

      CoreMsg m{};
      m.type    = MsgType::WINDOW;
      m.ts_ms   = now;
      m.buf_idx = pp;
      xQueueSend(q_msg, &m, 0);
      pp ^= 1;
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// INFERENCE TASK  (Core 0 / PRO) — TFLite + RepTracker
// ═══════════════════════════════════════════════════════════════
static void inferenceTask(void *arg) {
  RepTracker tracker;
  bool armed       = false;   // chưa "vũ trang" cho tới khi sensor settle
  int  idle_streak = 0;
  CoreMsg m;

  for (;;) {
    if (xQueueReceive(q_msg, &m, portMAX_DELAY) != pdTRUE) continue;

    // ── PEAK: detector bắn đỉnh → thử tính rep (gate trong tracker) ──
    if (m.type == MsgType::PEAK) {
      if (!armed) continue;   // chưa settle → bỏ qua
#if REP_DEBUG
      {
        static uint32_t dbg_last_peak = 0;
        Serial.printf("[PD] axis=%d env=%.2f T=%.2f dt=%lums\n", m.axis, m.env,
                      m.T, (unsigned long)(m.ts_ms - dbg_last_peak));
        dbg_last_peak = m.ts_ms;
      }
#endif
      if (tracker.onPeak(m.axis, m.ts_ms)) {
        int cc = tracker.committedClass();
        Serial.printf("\xF0\x9F\x92\xAA REP! [%s] | count: %d\n",
                      HAR_CLASS_NAMES[cc], tracker.repCount());
      }
      continue;
    }

    // ── WINDOW: chạy inference + cập nhật RepTracker ──
    HarResult result;
    HarStatus status = harRunInference(flat_buf[m.buf_idx], result);
    if (status != HarStatus::OK) {
      Serial.println("LOI Inference!");
      continue;
    }

    // ── Warm-up: chờ gyro settle (idle ổn định) trước khi phân loại/đếm ──
    if (!armed) {
      if (result.class_index == IDLE_CLASS_INDEX && result.confidence >= 0.80f)
        idle_streak++;
      else
        idle_streak = 0;
      if (idle_streak >= ARM_IDLE_STREAK || m.ts_ms >= ARM_TIMEOUT_MS) {
        armed = true;
        Serial.println("[INIT] Sensor da on dinh — bat dau nhan dien.");
      } else {
        Serial.printf("[INIT] warm-up... (%s %.0f%%)\n", result.class_name,
                      result.confidence * 100.0f);
        continue;
      }
    }

    if (result.confidence > 0.80f) {
      Serial.printf(
          "\xF0\x9F\x8F\x8B Hoat dong: %-15s | Tin cay: %5.1f%% | Tre: %2d ms\n",
          result.class_name, result.confidence * 100.0f,
          result.inference_time_ms);
    }

    tracker.onInference(result.class_index, result.confidence, m.ts_ms);

    if (tracker.committedChanged()) {
      if (tracker.endedClass() >= 0)
        Serial.printf("[SET] xong %s: %d reps\n",
                      HAR_CLASS_NAMES[tracker.endedClass()], tracker.endedCount());
      if (tracker.committedClass() >= 0)
        Serial.printf("[SET] bat dau %s (rep-classify = %d)\n",
                      HAR_CLASS_NAMES[tracker.committedClass()], tracker.repCount());
    }
  }
}

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

  q_msg = xQueueCreate(16, sizeof(CoreMsg));
  if (q_msg == nullptr) {
    Serial.println("❌ LỖI: Không tạo được queue!");
    while (1)
      ;
  }

  xTaskCreatePinnedToCore(inferenceTask, "infer", 12288, nullptr, 2, nullptr, 0);
  xTaskCreatePinnedToCore(samplerTask, "sampler", 4096, nullptr, 3, nullptr, 1);

  Serial.println("✅ Sẵn sàng (dual-core)! Sampler@Core1 50Hz, Inference@Core0...");
}

void loop() {
  vTaskDelay(portMAX_DELAY);   // mọi việc trong 2 task FreeRTOS
}
#endif
