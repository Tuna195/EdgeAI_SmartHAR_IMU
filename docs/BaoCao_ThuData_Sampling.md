# Báo cáo: Tính toàn vẹn lấy mẫu khi thu dữ liệu (sampling integrity)

**Ngày:** 2026-06-03
**Phạm vi:** Kiểm tra hiện tượng "thi thoảng miss ghi một vài samples" khi thu CSV (50 Hz).

---

## 1. Kết luận nhanh

- **Bộ dữ liệu hiện tại (51 file CSV) SẠCH — không mất mẫu nào.** Toàn bộ khoảng cách timestamp giữa 2 mẫu liên tiếp đều đúng **20 ms** (0/51 file có gap khác 20 ms).
- Dữ liệu train được thu bằng **đường Serial** (`src/main.cpp` cũ): timestamp là `millis()` thật → nếu có rớt mẫu sẽ hiện ra thành gap > 20 ms. Thực tế **không có gap nào** → loop chạy kịp 50 Hz, không nghẽn.
- **Tuy nhiên đường BLE** (`src/data_capture.cpp`) có **lỗi tiềm ẩn** khiến mẫu bị rớt mà **KHÔNG nhìn thấy được** (timestamp vẫn đều tăm tắp). Đường này hiện không dùng để thu bộ data train, nhưng cần sửa trước khi dùng lại.

Bằng chứng (script kiểm tra):
```
--- 0/51 file co gap khac 20ms ---
bicep_curl_01:   rows=2768  span=55340ms  diff: toàn bộ = 20ms  thiếu=0 mẫu
shoulder_press_01: rows=2765 span=55280ms diff: toàn bộ = 20ms  thiếu=0 mẫu
tricep_ext_01:   rows=3623  span=72440ms  diff: toàn bộ = 20ms  thiếu=0 mẫu
```

---

## 2. Phân tích hai đường thu dữ liệu

### 2a. Đường Serial — `src/main.cpp` (bản thu cũ)

```cpp
if (currentMillis - previousMillis >= SAMPLING_PERIOD_MS) {
    previousMillis = currentMillis;          // (A) gán = thời điểm thật
    ...
    Serial.print(currentMillis); ...         // (B) timestamp = millis() thật
}
```

- **(A) `previousMillis = currentMillis`** (gán, không cộng dồn): nếu một vòng loop bị trễ (vd `Serial.print` nghẽn khi buffer USB-CDC đầy), mốc kế được dời theo thời điểm trễ → **không cố bù lại** mẫu đã lỡ. Hệ quả: tần số có thể trôi xuống dưới 50 Hz, nhưng…
- **(B) timestamp = `millis()` thật**: …nên nếu rớt mẫu, khoảng cách sẽ hiện thành **> 20 ms** trong file → **rớt mẫu LỘ RA, kiểm tra được**.
- Thực tế quét 51 file: **0 gap** → đường này hoạt động tốt với pipeline hiện tại (chỉ đọc IMU + in Serial, tải nhẹ).

### 2b. Đường BLE — `src/data_capture.cpp` (ghi SPIFFS qua BLE)

```cpp
if (currentMillis - previousMillis < SAMPLING_PERIOD_MS) return;
previousMillis += SAMPLING_PERIOD_MS;        // (A') cộng dồn cố định
...
unsigned long timestamp = sampleCount * SAMPLING_PERIOD_MS;  // (B') timestamp TỔNG HỢP
dataFile.print(timestamp); ...
sampleCount++;
```

Hai vấn đề kết hợp tạo ra **mất mẫu ẩn**:

1. **(B') timestamp tổng hợp `sampleCount * 20`**: mỗi dòng ghi ra luôn cách nhau đúng 20 ms *theo công thức*, **bất kể** thực tế đã trôi bao lâu. → Nếu rớt mẫu, file **vẫn nhìn đều tăm tắp** — không thể phát hiện bằng cách xem timestamp (đây chính là lý do "thi thoảng miss mà không biết").
2. **Chỉ xử lý 1 mẫu / vòng loop + (A') cộng dồn**: khi ghi SPIFFS (flash) hoặc xử lý BLE làm 1 vòng loop kéo dài > 20 ms (thậm chí > 40 ms), code **chỉ đọc/ghi đúng 1 mẫu** rồi `previousMillis += 20`. Nó **không đọc bù** các mẫu lẽ ra rơi vào khoảng trễ đó → các mẫu này **biến mất vĩnh viễn**, trong khi `sampleCount` chỉ tăng 1.

→ Kết quả: file CSV có **ít dòng hơn thời gian thực**, nhưng timestamp vẫn 0,20,40,… hoàn hảo. Mất mẫu **hoàn toàn vô hình**.

> Độ trễ ghi SPIFFS trên ESP32-S3 có thể lên hàng chục ms khi flush/erase block — đủ để nuốt 1–vài mẫu mỗi lần. Tải BLE (notify) cũng góp thêm jitter.

---

## 3. Tác động

- **Lên bộ data train hiện tại:** KHÔNG. Data thu bằng đường Serial (2a), đã xác minh sạch.
- **Lên rep-counter / model:** không, vì cùng nguồn data sạch.
- **Rủi ro tương lai:** nếu quay lại thu data qua BLE (2b) mà chưa sửa, dữ liệu có thể thiếu mẫu mà không ai biết → méo tần số thật, lệch nhãn theo thời gian, ảnh hưởng cả train lẫn rep period.

---

## 4. Khuyến nghị (khi dùng lại đường BLE)

1. **Ghi timestamp THẬT** thay vì tổng hợp: `dataFile.print(currentMillis)` (hoặc lưu `micros()`), để mọi rớt mẫu lộ ra và kiểm tra được — giống đường Serial.
2. **Bù mẫu khi trễ** (catch-up): khi `currentMillis - previousMillis >= 2*PERIOD`, đọc/ghi đủ số mẫu đã lỡ (lặp), hoặc tối thiểu **đếm số mẫu bị bỏ** và ghi cảnh báo.
3. **Tách I/O nặng khỏi vòng lấy mẫu**: đọc IMU vào ring-buffer ở nhịp 50 Hz cố định (timer/`vTaskDelayUntil`), còn ghi SPIFFS/gửi BLE làm ở task/độ ưu tiên khác → flush flash không chặn nhịp lấy mẫu.
4. **Kiểm tra hậu kỳ**: luôn chạy script quét gap timestamp sau mỗi buổi thu (đã có sẵn logic; xem mục 1).

---

## 5. Ghi chú deploy liên quan

- Firmware vận hành (`src/main.cpp`) hiện là **classify + rep-detector PEAKDET**, dùng `currentMillis` thật cho cả nhịp lấy mẫu lẫn timestamp của detector → an toàn về mặt timing.
- Đường BLE data-capture (`src/data_capture.cpp`) được **giữ nguyên** sau cờ `-D DATA_CAPTURE_MODE` để tái dùng sau; các điểm cần sửa ở mục 4 áp dụng khi kích hoạt lại đường này.
