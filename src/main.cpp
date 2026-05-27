#ifndef DATA_CAPTURE_MODE
#include "SparkFun_BMI270_Arduino_Library.h"
#include "ai_inference.h"
#include "dsp_algorithm.h"
#include "norm_params.h"
#include <Arduino.h>
#include <Wire.h>

BMI270 imu;
EMAFilter filter_ax, filter_ay, filter_az, filter_gx, filter_gy, filter_gz;

const unsigned long SAMPLING_PERIOD_MS = 20; // 50Hz
unsigned long previousMillis = 0;

// TRUE RING BUFFER — O(1) time complexity per sample
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

    // 1. Áp dụng bộ lọc EMA
    float ax = filter_ax.filter(imu.data.accelX);
    float ay = filter_ay.filter(imu.data.accelY);
    float az = filter_az.filter(imu.data.accelZ);
    float gx = filter_gx.filter(imu.data.gyroX);
    float gy = filter_gy.filter(imu.data.gyroY);
    float gz = filter_gz.filter(imu.data.gyroZ);

    // 2. Nạp vào True Ring Buffer (xoay vòng con trỏ, O(1))
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
        // Chỉ in ra nếu tự tin > 80%
        if (result.confidence > 0.80f) {
          Serial.printf(
              "🏋️ Hoạt động: %-15s | Tin cậy: %5.1f%% | Trễ: %2d ms\n",
              result.class_name, result.confidence * 100.0f,
              result.inference_time_ms);
        }
      } else {
        Serial.println("❌ Lỗi Inference!");
      }
    }
  }
}
#endif