#ifdef DATA_CAPTURE_MODE
#include "SparkFun_BMI270_Arduino_Library.h"
#include "dsp_algorithm.h"
#include <Arduino.h>
#include <BLE2902.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <SPIFFS.h>

BMI270 imu;
EMAFilter filter_ax, filter_ay, filter_az, filter_gx, filter_gy, filter_gz;
const int LED_PIN = LED_BUILTIN;

// NUS UUIDs
#define SERVICE_UUID "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define CHARACTERISTIC_UUID_RX "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
#define CHARACTERISTIC_UUID_TX "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
void handleCommand(char cmd);
void doDump();
// BLE Globals
BLEServer *pServer = nullptr;
BLECharacteristic *pTxCharacteristic = nullptr;
bool deviceConnected = false;

void bleSend(const char *msg) {
  if (deviceConnected && pTxCharacteristic) {
    pTxCharacteristic->setValue((uint8_t *)msg, strlen(msg));
    pTxCharacteristic->notify();
  }
}

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *pServer) {
    deviceConnected = true;
    Serial.println("BLE: Phone da ket noi!");
  }
  void onDisconnect(BLEServer *pServer) {
    deviceConnected = false;
    Serial.println("BLE: Phone da ngat!");
    pServer->getAdvertising()->start();
  }
};

// Command queue: BLE callback CHỈ lưu 1 ký tự, loop() sẽ xử lý
volatile char pendingCommand = 0;

class MyCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pCharacteristic) {
    // Giữ callback CỰC NHẸ — không gọi String, bleSend, hay SPIFFS
    // để tránh tràn BTC_TASK stack (~3.5KB)
    std::string val = pCharacteristic->getValue();
    if (val.length() > 0) {
      pendingCommand = val[0];  // Chỉ lưu ký tự lệnh
    }
  }
};

// Activity Labels
const char *ACTIVITY_NAMES[] = {"idle", "walking", "running", "bicep_curl",
                                "lateral_raise", "pushup", "shoulder_press"};
const int NUM_ACTIVITIES = 7;

int currentActivity =
    0; // 0=idle, 1=walk, 2=run, 3=curl, 4=lat, 5=pushup, 6=shoulder
int sessionCount[7] = {0, 0, 0, 0, 0, 0, 0};

bool isRecording = false;
File dataFile;
unsigned long previousMillis = 0;
const unsigned long SAMPLING_PERIOD_MS = 20;
unsigned long sampleCount = 0;

String makeFileName() {
  sessionCount[currentActivity]++;
  char buf[32];
  sprintf(buf, "/%s_%02d.csv", ACTIVITY_NAMES[currentActivity],
          sessionCount[currentActivity]);
  return String(buf);
}

void listFiles() {
  File root = SPIFFS.open("/");
  File file = root.openNextFile();
  while (file) {
    String msg = String(file.name()) + " (" + String(file.size()) +
                 " bytes)\n" + " Total Bytes: " + String(SPIFFS.totalBytes()) +
                 " - Used: " + String(SPIFFS.usedBytes());
    bleSend(msg.c_str());
    file = root.openNextFile();
  }
}

// handleCommand() giờ LUÔN chạy từ loop() (loopTask ~8KB stack)
// => An toàn gọi bleSend(), SPIFFS, String, v.v.
void handleCommand(char c) {
  Serial.print("CMD: ");
  Serial.println(c);

  if (c >= '1' && c <= '7') {
    currentActivity = c - '1';
    String msg = ">> Activity: " + String(ACTIVITY_NAMES[currentActivity]);
    bleSend(msg.c_str());
  } else if (c == 's' || c == 'S') {
    if (isRecording)
      return;

    String fileName = makeFileName();
    dataFile = SPIFFS.open(fileName, FILE_WRITE);
    if (!dataFile) {
      bleSend("SPIFFS open dataFile failed");
      return;
    }
    dataFile.println("timestamp,ax,ay,az,gx,gy,gz");
    previousMillis = millis();
    isRecording = true;
    sampleCount = 0;
    String msg = ">>> REC: " + fileName + "\n";
    bleSend(msg.c_str());
  } else if (c == 'p' || c == 'P') {
    if (dataFile) {
      dataFile.close();
      isRecording = false;
      digitalWrite(LED_PIN, LOW);
      String msg = ">>> STOPPED - " + String(sampleCount) + " samples\n";
      bleSend(msg.c_str());
    } else {
      bleSend("Not recording");
    }
  } else if (c == 'l' || c == 'L') {
    listFiles();
  } else if (c == 'd' || c == 'D') {
    bleSend(">>> DUMPING...");
    doDump();
  } else if (c == 'f' || c == 'F') {
    if (isRecording) {
      bleSend("Dang record! Stop truoc khi format.");
      return;
    }
    bleSend(">>> FORMATTING SPIFFS...");
    SPIFFS.format();
    String msg = "Done! Free: " + String(SPIFFS.totalBytes()) + " bytes\n";
    bleSend(msg.c_str());
  }
}

// Hàm dump chạy trên loopTask (stack lớn ~8KB) — an toàn cho SPIFFS + Serial
void doDump() {
  // In thông tin SPIFFS để debug
  Serial.printf("SPIFFS: %u / %u bytes used\n", SPIFFS.usedBytes(), SPIFFS.totalBytes());
  Serial.println("Bat dau Dump Data...");

  File root = SPIFFS.open("/");
  File file = root.openNextFile();
  int fileIndex = 0;

  while (file) {
    size_t fileSize = file.size();
    fileIndex++;
    Serial.printf("[%d] %s (%u bytes)\n", fileIndex, file.name(), fileSize);

    // Header riêng 1 dòng để Python parse dễ dàng
    Serial.print("===File:");
    Serial.print(file.name());
    Serial.println("===");
    Serial.flush();

    // Đọc theo chunk, với safety counter chống loop vô hạn (SPIFFS corrupted)
    uint8_t buf[256];
    size_t totalRead = 0;
    while (file.available()) {
      int bytesRead = file.read(buf, sizeof(buf));
      if (bytesRead <= 0) break;  // Lỗi đọc -> thoát
      Serial.write(buf, bytesRead);
      totalRead += bytesRead;
      // Safety: nếu đọc vượt quá file size -> SPIFFS có vấn đề, thoát
      if (totalRead > fileSize + 256) {
        Serial.println("\n[!] SPIFFS ERROR: read > fileSize, skip file");
        break;
      }
      delay(1);
    }

    Serial.println();
    Serial.println("===END===");
    Serial.flush();
    delay(10);
    file = root.openNextFile();
  }
  Serial.println("Hoan thanh Dump Data");
  bleSend(">>> DUMP COMPLETE!\n");
}

void loop() {
  // Xử lý command từ BLE — chạy trên loopTask (stack lớn, an toàn)
  if (pendingCommand != 0) {
    char cmd = pendingCommand;
    pendingCommand = 0;
    handleCommand(cmd);
    return;
  }

  if (!isRecording)
    return;
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis < SAMPLING_PERIOD_MS)
    return;
  previousMillis += SAMPLING_PERIOD_MS;

  imu.getSensorData();

  float ax = filter_ax.filter(imu.data.accelX);
  float ay = filter_ay.filter(imu.data.accelY);
  float az = filter_az.filter(imu.data.accelZ);
  float gx = filter_gx.filter(imu.data.gyroX);
  float gy = filter_gy.filter(imu.data.gyroY);
  float gz = filter_gz.filter(imu.data.gyroZ);

  unsigned long timestamp = sampleCount * SAMPLING_PERIOD_MS;
  dataFile.print(timestamp);
  dataFile.print(",");
  dataFile.print(ax, 4);
  dataFile.print(",");
  dataFile.print(ay, 4);
  dataFile.print(",");
  dataFile.print(az, 4);
  dataFile.print(",");
  dataFile.print(gx, 4);
  dataFile.print(",");
  dataFile.print(gy, 4);
  dataFile.print(",");
  dataFile.println(gz, 4);
  sampleCount++;
  if (sampleCount % 50 == 0) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }
}
void initBLE() {
  BLEDevice::init("EdgeAI_Tracker");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  pTxCharacteristic = pService->createCharacteristic(
      CHARACTERISTIC_UUID_TX, BLECharacteristic::PROPERTY_NOTIFY);
  pTxCharacteristic->addDescriptor(new BLE2902());

  BLECharacteristic *pRxCharacteristic = pService->createCharacteristic(
      CHARACTERISTIC_UUID_RX, BLECharacteristic::PROPERTY_WRITE);
  pRxCharacteristic->setCallbacks(new MyCallbacks());

  pService->start();
  pServer->getAdvertising()->start();

  Serial.println("BLE da khoi tao, dang cho ket noi...");
}
void setup() {
  Serial.begin(115200);
  delay(3000); // Chờ Serial khởi động
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW); // Tắt đèn LED ban đầu
  // Khởi tạo IMU (I2C)
  Wire.begin(D4, D5);
  delay(100);
  if (imu.beginI2C(0x68) != BMI2_OK) {
    Serial.println("Loi: Khong tim thay IMU!");
    while (1)
      ; // Treo máy nếu lỗi phần cứng
  }
  // Khởi tạo SPIFFS
  if (!SPIFFS.begin(true)) {
    Serial.println("Loi: Mount SPIFFS that bai!");
    while (1)
      ;
  }
  // Khởi tạo Bluetooth
  initBLE();
  Serial.println("==========================================");
  Serial.println("   DATA CAPTURE MODE — EdgeAI Tracker");
  Serial.println("==========================================");
}
#endif