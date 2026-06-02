// rep_counter.cpp
// ─────────────────────────────────────────────────────────────
// Phase 1 — Rep Counter (v3 — Multi-axis + Majority-Vote)
//   RepDetector : Adaptive Schmitt 1 trục (giữ nguyên thuật toán v2).
//   RepTracker  : vote chọn bài + +1-khi-classify + đếm peak trên trục bài đó.
// ─────────────────────────────────────────────────────────────

#include "rep_counter.h"
#include <cmath>    // fabsf
#include <cstring>  // memset

// ═════════════════════════════════════════════════════════════
// 1. RepDetector — Adaptive Schmitt (1 trục cố định)
// ═════════════════════════════════════════════════════════════
//   centered  = val - dc_ema          (khử DC chậm → loại trôi tư thế/gravity)
//   centered  = MA ngắn (làm mượt)     (chống rung tần số cao → 1 đỉnh/nhịp)
//   envelope  = MA 0.5s của |centered| (ước lượng biên độ)
//   env_peak  = peak-hold của envelope (giữ cao suốt set → chống đếm wobble)
//   T         = max(k * env_peak, T_min)
//   Schmitt   : vượt +T rồi xuống dưới -T  ⇒  1 chu kỳ dao động = 1 rep

RepDetector::RepDetector() {
    axis_ = 4;  // gy mặc định
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
    // 0.5s đầu: baseline = trung bình cộng dồn → settle nhanh; sau đó slow EMA.
    if (ma_count_ < PD_MA_WIN) {
        warmup_sum_ += val;
        dc_ema_      = warmup_sum_ / (float)(ma_count_ + 1);
    } else {
        dc_ema_ = (1.0f - PD_DC_ALPHA) * dc_ema_ + PD_DC_ALPHA * val;
    }
    float centered_raw = val - dc_ema_;

    // ── Bước 1b: Làm mượt centered (MA ngắn) ─────────────────
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
    if (envelope_ > env_peak_) env_peak_ = envelope_;
    else                       env_peak_ *= PD_PEAK_DECAY;
    T_ = PD_K_BAND * env_peak_;
    if (T_ < PD_T_MIN) T_ = PD_T_MIN;

    // ── Bước 4: Schmitt trigger ──────────────────────────────
    if (!above_ && centered > T_) {
        above_ = true;
        return false;
    }
    if (above_ && centered < -T_) {
        above_ = false;
        return true;     // ← REP EVENT (1 chu kỳ đầy đủ)
    }
    return false;
}

// ═════════════════════════════════════════════════════════════
// 2. RepTracker — vote chọn bài + đếm
// ═════════════════════════════════════════════════════════════

RepTracker::RepTracker() { reset(); }

void RepTracker::reset() {
    vote_head_  = 0;
    vote_count_ = 0;
    for (int i = 0; i < RT_VOTE_N; i++) vote_buf_[i] = IDLE_CLASS_INDEX;
    committed_   = -1;
    last_infer_  = IDLE_CLASS_INDEX;
    commit_ts_   = 0;
    last_rep_ms_ = 0;
    for (int i = 0; i < NUM_EXERCISE_CLASSES; i++) {
        rep_count_[i]      = 0;
        first_credited_[i] = false;
    }
    committed_changed_ = false;
    ended_class_       = -1;
    ended_count_       = 0;
}

int RepTracker::voteWinner(int &winner_count) const {
    int counts[NUM_EXERCISE_CLASSES] = {0};
    for (int i = 0; i < vote_count_; i++) {
        int c = vote_buf_[i];
        if (c >= 0 && c < NUM_EXERCISE_CLASSES) counts[c]++;
    }
    int best = IDLE_CLASS_INDEX, best_n = -1;
    for (int c = 0; c < NUM_EXERCISE_CLASSES; c++) {
        if (counts[c] > best_n) { best_n = counts[c]; best = c; }
    }
    winner_count = best_n;
    return best;
}

void RepTracker::endCommittedSet() {
    if (committed_ < 0) return;
    ended_class_ = committed_;
    ended_count_ = rep_count_[committed_];
    rep_count_[committed_]      = 0;
    first_credited_[committed_] = false;
    committed_ = -1;
}

void RepTracker::onInference(int class_index, float confidence, uint32_t ts_ms) {
    committed_changed_ = false;
    ended_class_       = -1;

    // conf thấp → coi như idle (no-exercise) cho cả vote lẫn cổng đếm.
    int v = (confidence >= RT_MIN_CONF) ? class_index : IDLE_CLASS_INDEX;
    last_infer_ = v;

    // Đẩy vào cửa sổ vote (ring).
    vote_buf_[vote_head_] = v;
    vote_head_ = (vote_head_ + 1) % RT_VOTE_N;
    if (vote_count_ < RT_VOTE_N) vote_count_++;

    int wcount = 0;
    int winner = voteWinner(wcount);
    if (wcount < RT_VOTE_M) return;   // chưa ai áp đảo → giữ nguyên

    if (winner == IDLE_CLASS_INDEX) {
        // Nghỉ áp đảo → kết thúc set hiện tại (nếu có).
        if (committed_ >= 0) {
            endCommittedSet();
            committed_changed_ = true;
        }
        return;
    }

    // winner là 1 bài tập.
    if (winner != committed_) {
        endCommittedSet();          // kết thúc bài cũ (nếu có) → ghi ended_*
        committed_ = winner;
        if (!first_credited_[winner]) {
            rep_count_[winner]      = 1;   // +1 "rep-classify" (cửa sổ 3s ≈ 1 rep)
            first_credited_[winner] = true;
        }
        commit_ts_   = ts_ms;
        last_rep_ms_ = ts_ms;        // chặn đếm đôi ngay sau khi chốt
        committed_changed_ = true;
    }
}

bool RepTracker::onPeak(uint8_t axis, uint32_t ts_ms) {
    if (committed_ < 0) return false;
    const ExerciseConfig *cfg = getExerciseConfig(committed_);
    if (!cfg) return false;

    if (axis != cfg->best_axis)   return false;  // sai trục bài đang đếm
    if (last_infer_ != committed_) return false;  // không chắc đang làm bài này
    if (ts_ms <= commit_ts_)       return false;  // chỉ tính sau khi chốt (bỏ peak buildup)

    // Debounce: peak dồn < min_gap (tập quá nhanh / nhiễu) → coi sai form, bỏ.
    if (last_rep_ms_ != 0 && (ts_ms - last_rep_ms_) < cfg->min_gap_ms) return false;

    last_rep_ms_ = ts_ms;
    rep_count_[committed_]++;
    return true;
}
