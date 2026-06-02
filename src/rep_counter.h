#pragma once
// rep_counter.h
// ─────────────────────────────────────────────────────────────
// Phase 1 — Rep Counter (v3 — Multi-axis + Majority-Vote)
//
//   RepDetector  : phát hiện 1 rep (= 1 chu kỳ dao động) @ 50Hz trên MỘT trục
//                  gyro cố định bằng Schmitt trigger NGƯỠNG TỰ THÍCH NGHI.
//                  → chạy 3 detector cố định (gx/gy/gz), LUÔN ấm, độc lập
//                    classifier → không còn "sai trục lúc đầu" / warm-up gap.
//
//   RepTracker   : thay cho StateMachine + RepValidator cũ.
//                  • Chọn bài bằng MAJORITY-VOTE trên N inference gần nhất
//                    (bài ≠ idle đạt ≥ M lần → chốt) → chống flip-flop nhầm class.
//                  • Lúc chốt bài X: +1 "rep-classify" (cửa sổ 3s tạo ra classify
//                    ≈ đã tập 1 rep), đánh dấu first_credited[X] để KHÔNG cộng lại.
//                  • Sau khi chốt: peak trên ĐÚNG trục của X (và đang chắc chắn
//                    làm X) → rep_count[X]++.
//                  • Ngắt giữa chừng = vote tiếp: idle đạt M → kết thúc set;
//                    bài Y đạt M → chuyển hẳn sang Y (set mới). Lẻ tẻ → giữ X.
//
//   Lưu vết theo MẢNG index-by-class: rep_count[] + first_credited[] → mỗi bài
//   nhớ trạng thái riêng khi đan xen. Tập quá nhanh (peak dồn < min_gap) coi như
//   sai form → KHÔNG tính (debounce loại).
//
// Thứ tự gọi trong main.cpp (dual-core):
//   SamplerTask  : mỗi sample → 3×RepDetector.update() → fire thì gửi PEAK(axis)
//   InferenceTask: mỗi inference → tracker.onInference(); mỗi PEAK → tracker.onPeak()
// ─────────────────────────────────────────────────────────────

#include <cstdint>
#include "exercise_config.h"

// ─── Detector tuning (đã kiểm chứng bằng training/rep_sim.py) ──
static constexpr int      PD_MA_WIN     = 25;       // Moving avg = 0.5s @ 50Hz
static constexpr float    PD_DC_ALPHA   = 0.0067f;  // Khử DC chậm: τ≈3s, cutoff≈0.05Hz
static constexpr float    PD_K_BAND     = 0.6f;     // Ngưỡng Schmitt T = k * env_peak
static constexpr float    PD_T_MIN      = 0.20f;    // Sàn tuyệt đối (z-units) chống chatter
static constexpr int      PD_SMOOTH_WIN = 9;        // MA ngắn làm mượt centered (~0.18s)
static constexpr float    PD_PEAK_DECAY = 0.995f;   // đỉnh-envelope giữ-chậm (peak-hold)

// ─── RepTracker tuning ────────────────────────────────────────
static constexpr int      RT_VOTE_N   = 5;      // cửa sổ vote = 5 inference gần nhất
static constexpr int      RT_VOTE_M   = 3;      // ≥3/5 cùng 1 bài → chốt (≈4.5s)
static constexpr float    RT_MIN_CONF = 0.80f;  // conf < ngưỡng → coi như idle (no-exercise)

// ─── 1. Rep Detector (Adaptive Schmitt, 1 trục cố định) ────────
class RepDetector {
public:
    RepDetector();

    // Đặt trục cố định cho detector. Chỉ reset state khi ĐỔI trục.
    void configure(uint8_t axis);

    void reset();

    // Cập nhật 1 sample đã Z-score. Trả về true khi hoàn thành 1 chu kỳ rep.
    // norm_axes[6]: ax, ay, az, gx, gy, gz (khớp norm_params.h)
    bool update(const float norm_axes[6]);

    float   getEnvelope()  const { return envelope_; }
    float   getThreshold() const { return T_; }
    uint8_t getAxis()      const { return axis_; }

private:
    uint8_t axis_;

    float   dc_ema_;
    float   warmup_sum_;

    float   sm_buf_[PD_SMOOTH_WIN];
    int     sm_head_;
    int     sm_count_;
    float   sm_sum_;

    float   ma_buf_[PD_MA_WIN];
    int     ma_head_;
    int     ma_count_;
    float   ma_sum_;
    float   envelope_;
    float   env_peak_;

    float   T_;
    bool    above_;
};

// ─── 2. Rep Tracker (vote chọn bài + đếm theo bài committed) ────
class RepTracker {
public:
    RepTracker();
    void reset();

    // Gọi sau MỖI inference (chỉ khi đã armed). Cập nhật vote + chốt/đổi/kết-thúc set.
    void onInference(int class_index, float confidence, uint32_t ts_ms);

    // Gọi khi MỘT detector bắn đỉnh (axis = trục bắn). Trả về true nếu tính 1 rep.
    bool onPeak(uint8_t axis, uint32_t ts_ms);

    int  committedClass() const { return committed_; }            // bài đang đếm, -1 nếu không
    int  repCount()       const { return committed_ >= 0 ? rep_count_[committed_] : 0; }

    // Cờ + thông tin để main.cpp in log khi set chuyển trạng thái.
    bool committedChanged() const { return committed_changed_; }
    int  endedClass()       const { return ended_class_; }   // set vừa kết thúc (-1 nếu không)
    int  endedCount()       const { return ended_count_; }

private:
    // Trả về class chiếm nhiều nhất trong cửa sổ vote; gán số lần vào winner_count.
    int voteWinner(int &winner_count) const;
    // Kết thúc set của bài đang committed (ghi ended_*, reset rep_count/flag).
    void endCommittedSet();

    int      vote_buf_[RT_VOTE_N];
    int      vote_head_;
    int      vote_count_;

    int      committed_;     // bài đang đếm (-1 = không)
    int      last_infer_;    // inference gần nhất (đã gate theo conf) → cổng onPeak

    int      rep_count_[NUM_EXERCISE_CLASSES];
    bool     first_credited_[NUM_EXERCISE_CLASSES];
    uint32_t commit_ts_;     // thời điểm chốt bài hiện tại
    uint32_t last_rep_ms_;   // debounce

    bool     committed_changed_;
    int      ended_class_;
    int      ended_count_;
};
