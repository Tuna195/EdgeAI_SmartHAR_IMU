#pragma once
// ble_notify.h — BLE GATT notify gói TUYỆT ĐỐI (Stage 2, doc Section 3)
// ─────────────────────────────────────────────────────────────
// Gói 6 bytes [activity, confidence, rep_count, set_no, timestamp_s].
// TUYỆT ĐỐI (không phải event flag): mất gói BLE không mất dữ liệu —
// gói sau tự mang trạng thái đúng, Bluefy chỉ render giá trị mới nhất.
//
// Cadence (main.cpp): mỗi inference cycle (~1.5s) + ngay khi rep_count
// đổi (onSample trả true) → UI nhảy tức thì, không đợi hết window.
//
// Protocol phía Bluefy:
//   • set_no tăng        → set mới bắt đầu (tắt rest timer)
//   • activity về idle   → set vừa chốt; rep của set = rep_count
//     (rest timer: rest_start = now - 4500ms, bù trễ 3 window idle)
// ─────────────────────────────────────────────────────────────

#include <BLE2902.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <cstdint>

// NUS UUIDs — cùng họ với data_capture.cpp, Bluefy connect bằng service này
#define BLE_NOTIFY_SERVICE_UUID "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define BLE_NOTIFY_TX_UUID      "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

typedef struct __attribute__((packed)) {
  uint8_t  activity;     // class index: 0=bicep 1=idle 2=lateral 3=shoulder 4=tricep
  uint8_t  confidence;   // 0-100 (float x 100)
  uint8_t  rep_count;    // ĐẾM DỒN rep của set hiện tại (tuyệt đối)
  uint8_t  set_no;       // số thứ tự set (tăng khi commit set mới)
  uint16_t timestamp_s;  // giây từ lúc bật máy (little-endian, max ~18h)
} BLEPacket_t;
static_assert(sizeof(BLEPacket_t) == 6, "BLEPacket_t phai dung 6 bytes");

class BleNotify {
public:
  void init(const char* dev_name = "EdgeAI_Tracker") {
    BLEDevice::init(dev_name);
    // Có anten ngoài (u.FL) → P3 (0dBm) đủ tầm cho thiết bị đeo, tiết kiệm pin.
    // (Khi chưa cắm anten phải dùng P9 để bù sóng yếu — giờ không cần.)
    BLEDevice::setPower(ESP_PWR_LVL_P3);
    server_ = BLEDevice::createServer();
    server_->setCallbacks(new SrvCb(this));

    BLEService* svc = server_->createService(BLE_NOTIFY_SERVICE_UUID);
    tx_ = svc->createCharacteristic(BLE_NOTIFY_TX_UUID,
                                    BLECharacteristic::PROPERTY_NOTIFY);
    tx_->addDescriptor(new BLE2902());
    svc->start();

    // PHẢI quảng bá service UUID: Web Bluetooth filter theo services chỉ
    // thấy thiết bị có UUID trong gói advertising. scan response để tên
    // đầy đủ không bị cắt (adv packet chỉ 31 bytes).
    BLEAdvertising* adv = server_->getAdvertising();
    adv->addServiceUUID(BLE_NOTIFY_SERVICE_UUID);
    adv->setScanResponse(true);
    adv->start();
  }

  bool connected() const { return connected_; }

  void send(const BLEPacket_t& p) {
    if (!connected_ || !tx_) return;
    tx_->setValue((uint8_t*)&p, sizeof(p));
    tx_->notify();
  }

private:
  // Callback CỰC NHẸ (chạy trên BTC task stack nhỏ — bài học từ data_capture)
  class SrvCb : public BLEServerCallbacks {
  public:
    explicit SrvCb(BleNotify* o) : o_(o) {}
    void onConnect(BLEServer*) override { o_->connected_ = true; }
    void onDisconnect(BLEServer* s) override {
      o_->connected_ = false;
      s->getAdvertising()->start();        // cho phép reconnect
    }
  private:
    BleNotify* o_;
  };

  BLEServer*         server_    = nullptr;
  BLECharacteristic* tx_        = nullptr;
  volatile bool      connected_ = false;
};
