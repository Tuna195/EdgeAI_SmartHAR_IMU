# EdgeAI Smart Rehab: IMU Signal Processing & Bicep Curl Tracking

Dự án phát triển hệ thống nhúng thông minh (Edge AI) trên vi điều khiển ESP32, ứng dụng trong theo dõi và phục hồi chức năng bằng cảm biến IMU (Gia tốc kế).

## 📌 Tính năng nổi bật
* **Thu thập dữ liệu Real-time:** Đọc dữ liệu từ cảm biến IMU qua ESP32.
* **DSP Pipeline (Xử lý tín hiệu số):** * Áp dụng Bộ lọc thông thấp (Low-pass Butterworth) triệt tiêu nhiễu tần số cao.
  * Thuật toán **State Machine (Máy trạng thái)** kết hợp Hysteresis để dò đỉnh/đáy, miễn nhiễm với trôi đường cơ sở và chiều đeo cảm biến.
* **Deep Learning (Đang phát triển):** Chuẩn bị trích xuất Sliding Window để huấn luyện mạng 1D-CNN phân loại động tác đúng/sai.

## 🛠 Công nghệ sử dụng
* **Phần cứng:** ESP32-S3, Cảm biến IMU (MPU6050/ICM20948...)
* **Embedded:** C/C++, PlatformIO framework.
* **Data Science:** Python, Pandas, Scipy (Signal Processing), Jupyter Notebook.

## 🚀 Cách chạy thử DSP Pipeline
1. Clone repository này về máy.
2. Mở file `filter.ipynb` bằng Jupyter Notebook hoặc VS Code.
3. Chạy các Cell để quan sát thuật toán DSP dọn dẹp dữ liệu `bicep_curl_1.csv` và đếm số nhịp tập.

---
*Đang phát triển bởi Hoàng Minh Tuấn - [Năm 2026]*
