#include <Arduino.h>
#include <Wire.h>
#include "SparkFun_BMI270_Arduino_Library.h"
#include "dsp_algorithm.h"
#include "ai_inference.h"

BMI270 imu;

EMAFilter filter_ax, filter_ay, filter_az, filter_gx, filter_gy, filter_gz;
CircularBuffer data_buffer;
PeakDetector peak_detector(0.6f);
float ai_matrix[100][6];
DominantAxisSelector axis_selector;

// --- CAU HINH ---
const unsigned long SAMPLING_PERIOD_MS = 20; // 20ms = 50Hz
unsigned long previousMillis = 0;
int rep_count = 0;
int good_count = 0;
int bad_count = 0;

const float MIN_CONFIDENCE = 0.50f;           // Bo qua neu AI khong chac chan
const unsigned long MIN_REP_DURATION_MS = 200; // 1 nhip phai dai it nhat 0.5 giay

void setup() {
  Serial.begin(115200);
  unsigned long t = millis();
  while (!Serial && (millis() - t < 3000));
  
  Wire.begin(D4, D5);
  delay(100); 

  if (imu.beginI2C(0x68) != BMI2_OK) {
      Serial.println("LOI: Khong the khoi tao BMI270.");
      while(1);
  }
  
  ai_init();
  Serial.println("===========================");
  Serial.println("  EDGE AI REHAB TRACKER");
  Serial.println("  AI da san sang!");
  Serial.println("  Bat dau tap luyen...");
  Serial.println("===========================");
}

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= SAMPLING_PERIOD_MS) {
      previousMillis = currentMillis;
      imu.getSensorData();
      IMUData filtered_data;
      filtered_data.timestamp = currentMillis;
      filtered_data.ax = filter_ax.filter(imu.data.accelX);
      filtered_data.ay = filter_ay.filter(imu.data.accelY);
      filtered_data.az = filter_az.filter(imu.data.accelZ);
      filtered_data.gx = filter_gx.filter(imu.data.gyroX);
      filtered_data.gy = filter_gy.filter(imu.data.gyroY);
      filtered_data.gz = filter_gz.filter(imu.data.gyroZ);
      
      data_buffer.push(filtered_data);

      unsigned long start_time = 0, end_time = 0;
      float dominant_axis = axis_selector.update(filtered_data.ax, filtered_data.ay, filtered_data.az);
      bool rep_detected = peak_detector.processSample(dominant_axis, currentMillis, start_time, end_time);

      if(rep_detected){
        // Bo qua nhip qua ngan (nhieu/rung)
        unsigned long duration = end_time - start_time;
        if(duration < MIN_REP_DURATION_MS){
          return; // Bo qua, khong phai nhip that
        }

        Resampler::resample(&data_buffer, start_time, end_time, ai_matrix);
        float confidence = 0;
        int result = ai_predict(ai_matrix, &confidence);
        int pct = (int)(confidence * 100);
        
        // Bo qua neu AI khong chac chan
        if(confidence < MIN_CONFIDENCE){
          Serial.print("(?) Khong ro - "); Serial.print(pct); Serial.println("%");
          return;
        }

        if(result == 1) {
          rep_count++;
          good_count++;
          Serial.print("Rep #"); Serial.print(rep_count);
          Serial.print(" >>> GOOD FORM! ("); Serial.print(pct); Serial.println("%)");
        } else if(result == 2) {
          rep_count++;
          bad_count++;
          Serial.print("Rep #"); Serial.print(rep_count);
          Serial.print(" >>> BAD FORM! ("); Serial.print(pct); Serial.println("%)");
        } else {
          Serial.print("(Khong phai tap - "); Serial.print(pct); Serial.println("%)");
        }
        
        Serial.print("   [Good: "); Serial.print(good_count);
        Serial.print(" | Bad: "); Serial.print(bad_count);
        Serial.println("]");
      }
  }
}

