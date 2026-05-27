import serial
import os

# CAU HINH: Thay doi ten cong COM cua ban vao day (VD: COM3, /dev/ttyUSB0)
PORT = 'COM8' 
BAUDRATE = 115200
DATA_DIR = '../training/data'  # Luu vao training/data/ de phuc vu training pipeline

def main():
    try:
        # Khởi tạo kết nối Serial
        ser = serial.Serial(PORT, BAUDRATE, timeout=1)
        print(f"[*] Dang lang nghe tren {PORT}... Hay gui lenh 'd' tu dien thoai cua ban.")
        
        current_file = None
        is_writing = False
        file_count = 0
        
        # Đảm bảo thư mục gốc tồn tại
        os.makedirs(DATA_DIR, exist_ok=True)

        while True:
            # Đọc 1 dòng từ Serial và decode từ bytes sang string
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if not line:
                continue

            # --- TRẠNG THÁI 1: Bắt đầu một file mới ---
            # Firmware đã dùng Serial.println() nên header nằm trên dòng riêng:
            # "===File:/walking_01.csv==="
            if line.startswith("===File:") and line.endswith("==="):
                # 1. Trích xuất tên file
                raw_filename = line[len("===File:"):-len("===")]

                # 2. Xóa dấu '/' đầu tiên do SPIFFS thêm vào (VD: /walking_01.csv -> walking_01.csv)
                filename = raw_filename.lstrip('/')

                # 3. Lấy tên thư mục từ tên file (VD: "walking_01.csv" -> "walking")
                folder_name = filename.split('_')[0]

                # 4. Tạo thư mục nếu chưa có
                folder_path = os.path.join(DATA_DIR, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                # 5. Mở file để chuẩn bị ghi
                file_path = os.path.join(folder_path, filename)
                current_file = open(file_path, 'w', encoding='utf-8')
                is_writing = True
                file_count += 1
                print(f"[+] ({file_count}) Dang ghi: {file_path}")
                
            # --- TRẠNG THÁI 2: Kết thúc file ---
            elif line == "===END===":
                if current_file:
                    current_file.close()
                    current_file = None
                is_writing = False
                print(f"    -> Done.")
                
            # --- TRẠNG THÁI 3: Dừng toàn bộ chương trình ---
            elif "Hoan thanh Dump Data" in line:
                print(f"\n[*] DUMP HOAN TAT! Da luu {file_count} files vao '{os.path.abspath(DATA_DIR)}'")
                break
                
            # --- TRẠNG THÁI 4: Ghi nội dung CSV ---
            else:
                if is_writing and current_file:
                    current_file.write(line + "\n")
                else:
                    # In các dòng log khác từ ESP32 (VD: "Bat dau Dump Data...")
                    print(f"  ESP32: {line}")

    except serial.SerialException as e:
        print(f"[!] Loi Serial: {e}")
        print("    -> Kiem tra lai cong COM va cap USB.")
    except KeyboardInterrupt:
        print(f"\n[*] Ngat boi nguoi dung. Da luu {file_count} files.")
    except Exception as e:
        print(f"[!] Loi: {e}")
    finally:
        if 'current_file' in locals() and current_file:
            current_file.close()
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("[*] Da dong cong Serial.")

if __name__ == "__main__":
    main()
