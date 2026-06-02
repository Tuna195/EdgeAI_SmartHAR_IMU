#ifndef DATA_CAPTURE_MODE
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

BMI270 imu;

// Phase 1 — Rep Counter objects
RepDetector  peak_detector;
StateMachine state_machine;
RepValidator rep_validator;
int prev_counting_class = -1; // theo dõi counting-class để bỏ pending khi streak vỡ
int last_infer_class = IDLE_CLASS_INDEX; // class của inference gần nhất (để dừng đếm khi idle)

const unsigned long SAMPLING_PERIOD_MS = 20; // 50Hz
unsigned long previousMillis = 0;

// TRUE RING BUFFER — O(1) time complexity per sample
// Lưu dữ liệu RAW (KHÔNG còn EMA filter) — khớp domain với model (train trên raw).
float ring_buf[HAR_WINDOW_SIZE][HAR_NUM_AXES];
int head_index =
    0; // Trỏ tới vị trí ghi mẫu tiếp theo (cũng là mẫu cũ nhất trong cửa sổ)
int sample_count =
    0; // Đếm số mẫu để check buffer_full (Dừng đếm ở HAR_WINDOW_SIZE)
int stride_counter = 0; // Đếm riêng để xử lý Stride (Tránh Overflow integer)

const int STRIDE = HAR_WINDOW_SIZE / 2; // 75 mẫu (Stride)

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
  Serial.println("✅ Sẵn sàng nhận diện 5 bài tập! Bắt đầu lấy mẫu 50Hz...");
}

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= SAMPLING_PERIOD_MS) {
    previousMillis = currentMillis;

    imu.getSensorData();

    // 1. Đọc IMU RAW (KHÔNG còn EMA filter — model train trên raw, detector cũng)
    float ax = imu.data.accelX;
    float ay = imu.data.accelY;
    float az = imu.data.accelZ;
    float gx = imu.data.gyroX;
    float gy = imu.data.gyroY;
    float gz = imu.data.gyroZ;

    // [Phase 1] Z-score (z = (x - mean) * inv_std) + cập nhật detector (50Hz)
    // Detector LUÔN chạy để baseline khử-DC luôn ấm; chỉ ĐẾM khi có counting-class.
    const float raw6[6] = {ax, ay, az, gx, gy, gz};
    float norm6[6];
    for (int j = 0; j < HAR_NUM_AXES; j++)
      norm6[j] = (raw6[j] - HAR_MEAN[j]) * HAR_INV_STD[j];

    bool peak = peak_detector.update(norm6);

#if REP_DEBUG
    if (peak) {
      static uint32_t dbg_last_peak = 0;
      Serial.printf("[PD] axis=%d env=%.2f T=%.2f dt=%lums\n",
                    peak_detector.getAxis(), peak_detector.getEnvelope(),
                    peak_detector.getThreshold(),
                    (unsigned long)(currentMillis - dbg_last_peak));
      dbg_last_peak = currentMillis;
    }
#endif

    int cc = state_machine.getCountingClass();
    HarState st = state_machine.getState();
    bool counting = (cc >= 0);
    // Chỉ đếm "live" khi đang ACTIVE VÀ inference gần nhất KHÔNG phải idle.
    // → dừng đếm NGAY khi user dừng (frame idle tới), KHÔNG đếm xuyên TRANSITIONING.
    bool active = (st == HarState::ACTIVE) && (last_infer_class != IDLE_CLASS_INDEX);

    if (peak && counting) {
      const ExerciseConfig *cfg = getExerciseConfig(cc);
      uint32_t min_gap = cfg ? cfg->min_gap_ms : 800;
      if (rep_validator.onPeak(currentMillis, active, counting, min_gap)) {
        Serial.printf("\xF0\x9F\x92\xAA REP! [%s] | count: %d%s\n",
                      HAR_CLASS_NAMES[cc], rep_validator.getRepCount(),
                      active ? "" : " (pending)");
      }
    }

    // 2. Nạp RAW vào True Ring Buffer (xoay vòng con trỏ, O(1))
    ring_buf[head_index][0] = ax;
    ring_buf[head_index][1] = ay;
    ring_buf[head_index][2] = az;
    ring_buf[head_index][3] = gx;
    ring_buf[head_index][4] = gy;
    ring_buf[head_index][5] = gz;

    head_index = (head_index + 1) % HAR_WINDOW_SIZE;

    bool buffer_full = (sample_count < HAR_WINDOW_SIZE)
                           ? (++sample_count == HAR_WINDOW_SIZE)
                           : true;

    // 3. Suy luận AI mỗi khi thu đủ Window Size và qua đủ ngưỡng Stride
    if (buffer_full && (++stride_counter >= STRIDE)) {
      stride_counter = 0;

      // Flatten Ring Buffer thành mảng tuyến tính
      // Sử dụng static để tránh cấp phát 3.6KB trên Stack mỗi 1.5s
      static float flat_buf[HAR_WINDOW_SIZE][HAR_NUM_AXES];

      // Mảnh 1: Từ head_index đến hết mảng ring_buf (Đây là các mẫu CŨ NHẤT)
      int older = HAR_WINDOW_SIZE - head_index;
      memcpy(&flat_buf[0][0], &ring_buf[head_index][0],
             older * HAR_NUM_AXES * sizeof(float));

      // Mảnh 2: Từ đầu mảng ring_buf đến head_index-1 (Đây là các mẫu MỚI NHẤT)
      if (head_index > 0) {
        memcpy(&flat_buf[older][0], &ring_buf[0][0],
               head_index * HAR_NUM_AXES * sizeof(float));
      }

      HarResult result;
      // Suy luận trên mảng đã được trải phẳng
      HarStatus status = harRunInference(flat_buf, result);

      if (status == HarStatus::OK) {
        last_infer_class = result.class_index;  // cập nhật class mới nhất (gate dừng-khi-idle)

        // In inference nếu tự tin > 80%
        if (result.confidence > 0.80f) {
          Serial.printf(
              "\xF0\x9F\x8F\x8B Hoat dong: %-15s | Tin cay: %5.1f%% | Tre: %2d ms\n",
              result.class_name, result.confidence * 100.0f,
              result.inference_time_ms);
        }

        // [Phase 1] Cập nhật State Machine
        state_machine.update(result.class_index, result.confidence, currentMillis);

        // Cấu hình detector theo counting-class mới nhất (chỉ reset nếu ĐỔI trục
        // → giữ baseline ấm trong cùng 1 set). Bật đếm sớm ngay khi có streak.
        int ncc = state_machine.getCountingClass();
        const ExerciseConfig *ncfg = getExerciseConfig(ncc);
        if (ncfg)
          peak_detector.configure(ncfg->best_axis);

        bool nactive = (state_machine.getState() == HarState::ACTIVE ||
                        state_machine.getState() == HarState::TRANSITIONING);

        // Streak vỡ trước khi ACTIVE (đổi class / về idle) → bỏ pending tích luỹ.
        if (!nactive && ncc != prev_counting_class)
          rep_validator.resetPending();

        if (state_machine.stateChanged()) {
          HarState s = state_machine.getState();
          if (s == HarState::ACTIVE) {
            // Xác nhận ACTIVE → gộp pending (rep đã đếm trong lúc chờ) vào tổng.
            rep_validator.commitPending();
            Serial.printf("[STATE] ACTIVE  <- %s | reps so far: %d\n",
                          HAR_CLASS_NAMES[state_machine.getActiveClass()],
                          rep_validator.getRepCount());
          } else if (s == HarState::IDLE) {
            Serial.printf("[STATE] IDLE    (set xong: %d reps)\n",
                          rep_validator.getRepCount());
            rep_validator.reset();
          } else {
            Serial.println("[STATE] TRANSITIONING...");
          }
        }

        prev_counting_class = ncc;
      } else {
        Serial.println("LOI Inference!");
      }
    }
  }
}
#endif
