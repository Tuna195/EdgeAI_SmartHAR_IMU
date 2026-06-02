// rep_counter.cpp
// ─────────────────────────────────────────────────────────────
// Triển khai Phase 1 — Rep Counter Components (v2 — Adaptive)
// Thuật toán khớp 100% với training/rep_sim.py (AdaptiveDetector).
// ─────────────────────────────────────────────────────────────

#include "rep_counter.h"
#include <cmath>    // fabsf
#include <cstring>  // memset

// ═════════════════════════════════════════════════════════════
// 1. RepDetector — Adaptive Schmitt
// ═════════════════════════════════════════════════════════════
//
// Mỗi sample:
//   centered  = val - dc_ema          (khử DC chậm → loại trôi tư thế/gravity)
//   rectified = |centered|
//   envelope  = trung bình trượt 0.5s của rectified   (ước lượng biên độ)
//   T         = max(k * envelope, T_min)               (ngưỡng tự thích nghi)
//   Schmitt   : vượt +T rồi xuống dưới -T  ⇒  1 chu kỳ dao động = 1 rep
//
// Vì T tỉ lệ biên độ tức thời nên detector co giãn theo từng buổi/tay,
// đếm đúng 1 lần/rep kể cả khi vận động liên tục (envelope không tụt về 0).

RepDetector::RepDetector() {
    axis_ = 4;  // mặc định gy (chỉ là khởi tạo; configure() sẽ đặt lại)
    reset();
}

void RepDetector::configure(uint8_t axis) {
    if (axis == axis_) return;  // cùng trục → giữ baseline ấm, không reset
    axis_ = axis;
    reset();
}

void RepDetector::reset() {
    warmup_sum_ = 0.0f;
    dc_ema_   = 0.0f;
    ma_head_  = 0;
    ma_count_ = 0;
    ma_sum_   = 0.0f;
    envelope_ = 0.0f;
    env_peak_ = 0.0f;
    T_        = PD_T_MIN;
    above_    = false;
    memset(ma_buf_, 0, sizeof(ma_buf_));
    sm_head_  = 0;
    sm_count_ = 0;
    sm_sum_   = 0.0f;
    memset(sm_buf_, 0, sizeof(sm_buf_));
}

bool RepDetector::update(const float norm_axes[6]) {
    float val = norm_axes[axis_];

    // ── Bước 1: DC removal ───────────────────────────────────
    // 0.5s đầu (đang nạp MA): baseline = trung bình cộng dồn → settle NHANH,
    // tránh lệch do init bằng 1 mẫu giữa nhịp (gây miss/sai nhịp đầu).
    // Sau đó: slow EMA (τ≈3s) chỉ khử trôi tư thế.
    if (ma_count_ < PD_MA_WIN) {
        warmup_sum_ += val;
        dc_ema_      = warmup_sum_ / (float)(ma_count_ + 1);
    } else {
        dc_ema_ = (1.0f - PD_DC_ALPHA) * dc_ema_ + PD_DC_ALPHA * val;
    }
    float centered_raw = val - dc_ema_;

    // ── Bước 1b: Làm mượt centered (MA ngắn) → chống rung → 1 đỉnh/nhịp ──
    if (sm_count_ < PD_SMOOTH_WIN) {
        sm_buf_[sm_count_] = centered_raw;
        sm_sum_           += centered_raw;
        sm_count_++;
    } else {
        sm_sum_           -= sm_buf_[sm_head_];
        sm_buf_[sm_head_]  = centered_raw;
        sm_sum_           += centered_raw;
        sm_head_           = (sm_head_ + 1) % PD_SMOOTH_WIN;
    }
    float centered  = sm_sum_ / (float)sm_count_;
    float rectified = fabsf(centered);

    // ── Bước 2: Moving Average (circular) → envelope ─────────
    if (ma_count_ < PD_MA_WIN) {
        ma_buf_[ma_count_] = rectified;
        ma_sum_           += rectified;
        ma_count_++;
        if (ma_count_ < PD_MA_WIN) return false;  // buffer chưa đầy (0.5s đầu)
        envelope_ = ma_sum_ / PD_MA_WIN;
        ma_head_  = 0;
    } else {
        float oldest      = ma_buf_[ma_head_];
        ma_buf_[ma_head_] = rectified;
        ma_head_          = (ma_head_ + 1) % PD_MA_WIN;
        ma_sum_           = ma_sum_ - oldest + rectified;
        envelope_         = ma_sum_ / PD_MA_WIN;
    }

    // ── Bước 3: Ngưỡng thích nghi (peak-hold) ────────────────
    // T bám theo ĐỈNH biên độ (giữ cao suốt set), không tụt giữa 2 nhịp
    // → dao động nhỏ (wobble) giữa nhịp không vượt ngưỡng → hết đếm dư.
    if (envelope_ > env_peak_) env_peak_ = envelope_;
    else                       env_peak_ *= PD_PEAK_DECAY;
    T_ = PD_K_BAND * env_peak_;
    if (T_ < PD_T_MIN) T_ = PD_T_MIN;

    // ── Bước 4: Schmitt trigger trên tín hiệu đã khử DC ──────
    if (!above_ && centered > T_) {
        above_ = true;   // vào nửa trên của chu kỳ
        return false;
    }
    if (above_ && centered < -T_) {
        above_ = false;  // hoàn thành 1 chu kỳ
        return true;     // ← REP EVENT
    }
    return false;
}

// ═════════════════════════════════════════════════════════════
// 2. StateMachine
// ═════════════════════════════════════════════════════════════

StateMachine::StateMachine() { reset(); }

void StateMachine::reset() {
    state_          = HarState::IDLE;
    active_class_   = -1;
    state_changed_  = false;
    confirm_class_  = -1;
    confirm_streak_ = 0;
    exit_counter_   = 0;
    trans_start_ms_ = 0;
}

int StateMachine::getCountingClass() const {
    // ACTIVE và TRANSITIONING đều còn đang trong 1 set (TRANS có thể là false
    // alarm → quay lại ACTIVE) ⇒ vẫn đếm cho active_class_.
    if (state_ == HarState::ACTIVE ||
        state_ == HarState::TRANSITIONING) return active_class_;
    if (confirm_streak_ > 0)               return confirm_class_;  // đang gom streak
    return -1;
}

HarState StateMachine::update(int class_index, float confidence,
                               uint32_t timestamp_ms) {
    state_changed_ = false;

    switch (state_) {

    // ── IDLE: chờ N inference liên tiếp cùng class (non-idle) ─
    case HarState::IDLE:
        if (class_index != IDLE_CLASS_INDEX && confidence >= SM_MIN_CONF) {
            if (class_index == confirm_class_) {
                confirm_streak_++;
            } else {
                confirm_class_  = class_index;
                confirm_streak_ = 1;
            }
            if (confirm_streak_ >= SM_CONFIRM_N) {
                state_          = HarState::ACTIVE;
                active_class_   = confirm_class_;
                state_changed_  = true;
                exit_counter_   = 0;
                confirm_streak_ = 0;
            }
        } else {
            // idle hoặc confidence thấp → phá streak
            confirm_class_  = -1;
            confirm_streak_ = 0;
        }
        break;

    // ── ACTIVE: ổn định hoặc phát hiện exit ───────────────────
    case HarState::ACTIVE:
        if (class_index == active_class_ && confidence >= SM_MIN_CONF) {
            exit_counter_ = 0;  // inference ổn định, tiếp tục set
        } else {
            exit_counter_++;
            if (exit_counter_ >= SM_EXIT_M) {
                state_          = HarState::TRANSITIONING;
                trans_start_ms_ = timestamp_ms;
                state_changed_  = true;
            }
        }
        break;

    // ── TRANSITIONING: false alarm hoặc confirm IDLE ──────────
    case HarState::TRANSITIONING:
        if (class_index == active_class_ && confidence >= SM_MIN_CONF) {
            // False alarm — quay lại ACTIVE (người dùng chưa nghỉ thật)
            state_         = HarState::ACTIVE;
            state_changed_ = true;
            exit_counter_  = 0;
        } else if ((timestamp_ms - trans_start_ms_) >= SM_CONFIRM_IDLE_MS) {
            // Đã qua 3s mà không quay lại → xác nhận nghỉ
            state_          = HarState::IDLE;
            active_class_   = -1;
            state_changed_  = true;
            confirm_class_  = -1;
            confirm_streak_ = 0;
        }
        break;
    }

    return state_;
}

// ═════════════════════════════════════════════════════════════
// 3. RepValidator (debounce + retro-count)
// ═════════════════════════════════════════════════════════════

RepValidator::RepValidator() { reset(); }

void RepValidator::reset() {
    rep_count_   = 0;
    pending_     = 0;
    last_rep_ms_ = 0;
}

bool RepValidator::onPeak(uint32_t timestamp_ms, bool active, bool counting,
                          uint32_t min_gap_ms) {
    // Chỉ tính khi có một bài tập đang được counting (streak hoặc active).
    if (!counting) return false;

    // Debounce — tránh đếm nhiều rep từ 1 đỉnh nhiễu.
    if (last_rep_ms_ != 0 && (timestamp_ms - last_rep_ms_) < min_gap_ms) {
        return false;
    }
    last_rep_ms_ = timestamp_ms;

    if (active) {
        rep_count_++;   // đã ACTIVE → tính ngay
    } else {
        pending_++;     // đang chờ xác nhận → tạm giữ (sẽ commit khi ACTIVE)
    }
    return true;
}

void RepValidator::commitPending() {
    rep_count_ += pending_;
    pending_    = 0;
}

void RepValidator::resetPending() {
    pending_ = 0;
}
