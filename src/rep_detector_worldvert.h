#pragma once
// rep_detector_worldvert.h — WORLD-VERTICAL rep detector (CHỈ cho shoulder_press)
// ─────────────────────────────────────────────────────────────
// Mirror ý tưởng training/sim_worldvert.py + ĐẾM THEO DURATION (anti-nhún).
//
// shoulder_press là chuyển động TỊNH TIẾN (lên/xuống), không xoay → không trục
// accel đơn lẻ nào tin được. Pipeline (CAUSAL, dùng RAW: accel g, gyro deg/s):
//   1) g_hat = complementary filter (gyro xoay nhanh + accel kéo chậm)
//   2) a_up = -dot(a - g_hat, ĝ)   = gia tốc đứng THẬT, CÓ DẤU (+ lên, − xuống)
//   3) v_up = leaky_integrate(a_up) = vận tốc đứng có dấu
//   4) ĐẾM THEO DURATION: pha LÊN (v>0) đủ dài RỒI pha XUỐNG (v<0) đủ dài = 1 rep.
//
// VÌ SAO DÙNG DẤU (không phải |v|): một cú đẩy lên = MỘT pha dương liền mạch dù
// nâng nặng/giật cục; còn |v| sẽ vỡ thành nhiều cụm giật → đếm nhầm. Dấu của v
// (lên/xuống) bền với giật cục.
// VÌ SAO DURATION (không phải amplitude/travel): nhún nhanh có vận tốc cao & tích
// đủ quãng đường, nhưng MỖI PHA quá ngắn (~0.15s). Không thể giả pha đi-lên LIỀN
// MẠCH ~1s bằng cách lắc → muốn pha dài phải đi thật. Định nghĩa DƯƠNG (whitelist).
//   • thr        : deadband chống rung quanh 0 khi đổi pha.
//   • min_phase  : nửa nhịp tối thiểu (ms). Rep thật ~1–1.5s; nhún <0.5s → ~450ms.
//   • min_gap    : debounce rep↔rep.
// ─────────────────────────────────────────────────────────────

#include <cstdint>
#include <math.h>

class WorldVertDetector {
public:
    void configure(float thr, uint32_t min_gap_ms, float alpha = 0.02f,
                   float leak = 0.985f, uint32_t min_up_ms = 900,
                   uint32_t min_down_ms = 350) {
        thr_         = thr;
        min_gap_ms_  = min_gap_ms;
        alpha_       = alpha;
        leak_        = leak;
        min_up_ms_   = min_up_ms;     // gate CHÍNH chống nhún (pha lên phải đủ dài)
        min_down_ms_ = min_down_ms;   // chỉ cần có pha xuống thật (không siết)
        reset();
    }

    void reset() {
        inited_ = false;
        gx_ = gy_ = gz_ = 0.0f;
        v_ = 0.0f;
        phase_ = 0; phase_start_ = 0;
        up_dur_ = 0; up_ok_ = false; counted_ = false;
        have_last_ = false; last_ms_ = 0;
        last_up_ms_ = 0; last_down_ms_ = 0;
    }

    // raw6 = ax,ay,az (g)  +  gx,gy,gz (deg/s). Trả về true khi đếm 1 rep.
    bool update(const float raw6[6], uint32_t now_ms) {
        const float DT      = 0.02f;          // 50Hz
        const float DEG2RAD = 0.01745329252f;

        float ax = raw6[0], ay = raw6[1], az = raw6[2];
        float wx = raw6[3] * DEG2RAD, wy = raw6[4] * DEG2RAD, wz = raw6[5] * DEG2RAD;

        if (!inited_) { gx_ = ax; gy_ = ay; gz_ = az; inited_ = true; }

        // 1) gyro xoay g_hat: dg/dt = -(w × g)
        float cx = wy * gz_ - wz * gy_;
        float cy = wz * gx_ - wx * gz_;
        float cz = wx * gy_ - wy * gx_;
        gx_ -= cx * DT; gy_ -= cy * DT; gz_ -= cz * DT;
        // 2) accel kéo nhẹ g_hat về
        gx_ = (1.0f - alpha_) * gx_ + alpha_ * ax;
        gy_ = (1.0f - alpha_) * gy_ + alpha_ * ay;
        gz_ = (1.0f - alpha_) * gz_ + alpha_ * az;

        // a_up = -dot(a - g_hat, ĝ)   (+ = đi lên, − = đi xuống)
        float norm = sqrtf(gx_ * gx_ + gy_ * gy_ + gz_ * gz_) + 1e-9f;
        float ux = gx_ / norm, uy = gy_ / norm, uz = gz_ / norm;
        float lx = ax - gx_, ly = ay - gy_, lz = az - gz_;
        float a_up = -(lx * ux + ly * uy + lz * uz);

        // 3) leaky integrate → v_up (có dấu)
        v_ = leak_ * v_ + a_up * DT;

        // 4) ĐẾM theo DURATION nửa nhịp (deadband ±thr chống rung khi đổi pha)
        bool rep = false;
        if (v_ > thr_) {                       // ── vào pha LÊN ──
            if (phase_ != 1) { phase_ = 1; phase_start_ = now_ms; }
        } else if (v_ < -thr_) {               // ── vào pha XUỐNG ──
            if (phase_ != -1) {                // vừa chuyển LÊN→XUỐNG
                up_dur_ = (phase_ == 1) ? (now_ms - phase_start_) : 0;
                up_ok_  = (up_dur_ >= min_up_ms_);   // pha lên đủ dài? (gate chính)
                phase_ = -1; phase_start_ = now_ms; counted_ = false;
            }
            if (up_ok_ && !counted_) {         // xuống đủ lâu SAU khi lên đủ lâu → rep
                uint32_t down_dur = now_ms - phase_start_;
                if (down_dur >= min_down_ms_) {
                    bool gap_ok = !have_last_ || (now_ms - last_ms_) >= min_gap_ms_;
                    if (gap_ok) {
                        last_ms_ = now_ms; have_last_ = true;
                        last_up_ms_ = up_dur_; last_down_ms_ = down_dur;
                        rep = true; counted_ = true;
                    }
                }
            }
        }
        // |v|<=thr: giữ nguyên pha (đi qua điểm quay đầu); duration tính từ phase_start_
        return rep;
    }

    // debug getters (in ra để tune min_phase)
    float    velocity()   const { return v_; }
    uint32_t lastUpMs()   const { return last_up_ms_; }
    uint32_t lastDownMs() const { return last_down_ms_; }

private:
    // config
    float    thr_          = 0.012f;
    uint32_t min_gap_ms_   = 500;
    float    alpha_        = 0.02f;
    float    leak_         = 0.985f;
    uint32_t min_up_ms_    = 900;
    uint32_t min_down_ms_  = 350;

    // state
    bool     inited_ = false;
    float    gx_ = 0, gy_ = 0, gz_ = 0;        // g_hat
    float    v_ = 0;                            // v_up (có dấu)
    int8_t   phase_ = 0;                        // 0 chưa, +1 lên, -1 xuống
    uint32_t phase_start_ = 0;
    uint32_t up_dur_ = 0;
    bool     up_ok_ = false;
    bool     counted_ = false;
    bool     have_last_ = false;
    uint32_t last_ms_ = 0;
    uint32_t last_up_ms_ = 0, last_down_ms_ = 0;
};
