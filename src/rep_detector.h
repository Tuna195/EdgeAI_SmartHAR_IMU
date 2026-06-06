#pragma once
// rep_detector.h — PEAKDET (Billauer, delta) streaming rep counter
// ─────────────────────────────────────────────────────────────
// Mirror 1:1 training/sim_peakdet.py  (smooth_causal -> peakdet -> phase-lock count).
//
// Ý tưởng: bám đáy chạy (mn) và đỉnh chạy (mx). Khi tín hiệu RỜI cực đó xa hơn
// `delta` → chốt cực đó. 1 valley + 1 peak = 1 rep (khoá pha: cực đầu định E1,
// chạm cực ĐỐI E2 = 1 rep). delta = biên độ swing tối thiểu (z-units).
//   • BẤT BIẾN với offset DC (ngưỡng tương đối) → bền với cách đeo/trôi, không
//     miss nhịp đầu (khác Schmitt ±T tuyệt đối cũ).
//   • smooth_win: MA causal làm mượt chống gai trước peakdet.
//   • min_gap_ms: debounce thời gian giữa 2 rep.
//   • reset_timeout_ms: không có cực nào trong khoảng này → reset pha (về idle).
//   • min_abs: sàn biên độ tuyệt đối (0 = TẮT). Knob phụ lọc FP biên độ nhỏ.
//
// Đơn vị thời gian = ms (now_ms từ millis()). Detector chỉ chạy khi đang trong
// 1 bài tập (classifier gate ở main.cpp); idle do CLASSIFIER loại, không phải đây.
// ─────────────────────────────────────────────────────────────

#include <cstdint>

class RepDetector {
public:
    // axis: 0..5 (ax,ay,az,gx,gy,gz). delta/min_abs: z-units.
    void configure(uint8_t axis, float delta, uint32_t min_gap_ms,
                   uint8_t smooth_win, uint32_t reset_timeout_ms,
                   float min_abs = 0.0f, uint32_t min_swing_ms = 0) {
        axis_             = axis;
        delta_            = delta;
        min_gap_ms_       = min_gap_ms;
        smooth_win_       = smooth_win < 1 ? 1
                          : (smooth_win > MAX_SMOOTH ? MAX_SMOOTH : smooth_win);
        reset_timeout_ms_ = reset_timeout_ms;
        min_abs_          = min_abs;
        min_swing_ms_     = min_swing_ms;   // 0 = tắt. Chống nhún nhanh (anti-cheat).
        reset();
    }

    void reset() {
        sm_head_ = 0; sm_count_ = 0; sm_sum_ = 0.0f;       // smoothing
        mn_ = 1e30f; mx_ = -1e30f; mn_ms_ = 0; mx_ms_ = 0; // peakdet
        phase_ = 0;                                        // 0 chưa định, +1 chờ peak, -1 chờ valley
        e1_type_ = 0; saw_e1_ = false;                     // counter
        have_last_rep_ = false; last_rep_ms_ = 0;
        have_last_ev_  = false; last_ev_ms_  = 0;
        last_swing_ms_ = 0;
    }

    // z6[6] = 6 trục đã Z-score. Trả về true khi đếm được 1 rep.
    bool update(const float z6[6], uint32_t now_ms) {
        // ── 1) Làm mượt MA causal (ring buffer) ──────────────
        float v = z6[axis_];
        if (sm_count_ < smooth_win_) {
            sm_buf_[sm_count_] = v; sm_sum_ += v; sm_count_++;
        } else {
            sm_sum_ -= sm_buf_[sm_head_];
            sm_buf_[sm_head_] = v; sm_sum_ += v;
            sm_head_ = (sm_head_ + 1) % smooth_win_;
        }
        float x = sm_sum_ / (float)sm_count_;

        // ── 2) Peakdet streaming → event (0 không, +1 peak, -1 valley) ──
        if (x > mx_) { mx_ = x; mx_ms_ = now_ms; }
        if (x < mn_) { mn_ = x; mn_ms_ = now_ms; }

        if (phase_ == 0) {                       // chưa biết chiều → chờ rời 1 cực > delta
            if (x > mn_ + delta_)      { phase_ =  1; mx_ = x; mx_ms_ = now_ms; }
            else if (x < mx_ - delta_) { phase_ = -1; mn_ = x; mn_ms_ = now_ms; }
            return false;
        }

        int8_t   ev    = 0;
        uint32_t ev_ms = now_ms;
        if (phase_ == 1) {                       // đang chờ peak
            if (x < mx_ - delta_) {              // đã rời đỉnh đủ xa → chốt peak
                if (min_abs_ <= 0.0f || mx_ >= min_abs_) { ev = 1; ev_ms = mx_ms_; }
                mn_ = x; mn_ms_ = now_ms; phase_ = -1;
            }
        } else {                                 // đang chờ valley
            if (x > mn_ + delta_) {              // đã rời đáy đủ xa → chốt valley
                if (min_abs_ <= 0.0f || mn_ <= -min_abs_) { ev = -1; ev_ms = mn_ms_; }
                mx_ = x; mx_ms_ = now_ms; phase_ = 1;
            }
        }
        if (ev == 0) return false;

        // ── 3) Khoá pha + debounce (mirror count_reps) ───────
        if (have_last_ev_ && (ev_ms - last_ev_ms_) > reset_timeout_ms_) {
            e1_type_ = 0; saw_e1_ = false;       // nghỉ quá lâu → reset pha
        }
        uint32_t prev_ev_ms = last_ev_ms_;       // mốc cực TRƯỚC (cho min_swing)
        last_ev_ms_ = ev_ms; have_last_ev_ = true;

        if (e1_type_ == 0) { e1_type_ = ev; saw_e1_ = true; return false; }

        if (saw_e1_) {
            if (ev != e1_type_) {                // chạm cực ĐỐI → xong 1 chu kỳ
                saw_e1_ = false;
                uint32_t swing = ev_ms - prev_ev_ms;   // đáy→đỉnh kéo dài bao lâu
                bool gap_ok   = !have_last_rep_ || (ev_ms - last_rep_ms_) >= min_gap_ms_;
                bool swing_ok = (min_swing_ms_ == 0) || (swing >= min_swing_ms_);
                if (gap_ok && swing_ok) {        // đủ cách + đủ CHẬM → 1 REP
                    last_rep_ms_ = ev_ms; have_last_rep_ = true;
                    last_swing_ms_ = swing;
                    return true;
                }
            }
        } else {
            if (ev == e1_type_) saw_e1_ = true;  // quay lại E1 → mở rep kế
        }
        return false;
    }

    uint8_t  axis() const { return axis_; }
    uint32_t lastSwingMs() const { return last_swing_ms_; }   // debug/tune

private:
    static const int MAX_SMOOTH = 32;

    // config
    uint8_t  axis_             = 4;
    float    delta_            = 1.0f;
    uint32_t min_gap_ms_       = 600;
    uint8_t  smooth_win_       = 5;
    uint32_t reset_timeout_ms_ = 3000;
    float    min_abs_          = 0.0f;
    uint32_t min_swing_ms_     = 0;

    // smoothing
    float sm_buf_[MAX_SMOOTH];
    int   sm_head_  = 0;
    int   sm_count_ = 0;
    float sm_sum_   = 0.0f;

    // peakdet
    float    mn_ = 1e30f, mx_ = -1e30f;
    uint32_t mn_ms_ = 0,  mx_ms_ = 0;
    int8_t   phase_ = 0;

    // counter
    int8_t   e1_type_ = 0;
    bool     saw_e1_  = false;
    bool     have_last_rep_ = false;
    uint32_t last_rep_ms_   = 0;
    bool     have_last_ev_  = false;
    uint32_t last_ev_ms_    = 0;
    uint32_t last_swing_ms_ = 0;
};
