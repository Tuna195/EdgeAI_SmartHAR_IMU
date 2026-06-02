#pragma once
// rep_counter.h
// ─────────────────────────────────────────────────────────────
// Phase 1 — Rep Counter Components (v2 — Adaptive)
//
//   RepDetector   : phát hiện 1 rep (= 1 chu kỳ dao động) @ 50 Hz
//                   bằng Schmitt trigger với NGƯỠNG TỰ THÍCH NGHI
//                   theo biên độ tức thời (envelope). → co giãn theo
//                   từng buổi tập / từng tay, hết nhân đôi đỉnh.
//   StateMachine  : IDLE / ACTIVE / TRANSITIONING (cổng phân loại)
//   RepValidator  : debounce + retro-count (pending) để không mất
//                   rep trong lúc chờ State Machine xác nhận ACTIVE.
//
// Vì sao cần cổng phân loại: dữ liệu "idle" thực tế VẪN có nhiều cử
// động (biên độ ngang bài tập) → KHÔNG thể loại idle bằng biên độ,
// phải dựa vào classifier (model). Detector chỉ ĐẾM khi có một bài
// tập đang được xác nhận (counting-class >= 0).
//
// Thứ tự gọi trong main.cpp:
//   Mỗi sample (50Hz) →  detector.update()  (luôn chạy → baseline ấm)
//                      →  rep_validator.onPeak() nếu fire
//   Mỗi inference     →  state_machine.update()
//                      →  detector.configure(axis của counting-class)
//                      →  commit/reset pending theo chuyển trạng thái
// ─────────────────────────────────────────────────────────────

#include <cstdint>
#include "exercise_config.h"

// ─── Detector tuning (đã kiểm chứng bằng training/rep_sim.py) ──
static constexpr int      PD_MA_WIN    = 25;       // Moving avg = 0.5s @ 50Hz
static constexpr float    PD_DC_ALPHA  = 0.0067f;  // Khử DC chậm: τ≈3s, cutoff≈0.05Hz
                                                   // (đủ chậm để KHÔNG ăn dải rep 0.25–0.5Hz)
static constexpr float    PD_K_BAND    = 0.6f;     // Ngưỡng Schmitt T = k * envelope
static constexpr float    PD_T_MIN     = 0.20f;    // Sàn tuyệt đối (z-units) chống chatter
static constexpr int      PD_SMOOTH_WIN = 9;       // MA ngắn làm mượt centered (~0.18s)
                                                   // → chống rung tần số cao → 1 đỉnh/nhịp
static constexpr float    PD_PEAK_DECAY = 0.995f;  // đỉnh-envelope giữ-chậm: T bám biên độ
                                                   // nhịp, KHÔNG tụt giữa 2 nhịp (chống đếm wobble)

// ─── State Machine tuning ─────────────────────────────────────
static constexpr int      SM_CONFIRM_N       = 2;       // N inference cùng class → ACTIVE (~4.5s)
static constexpr int      SM_EXIT_M          = 3;       // M inference khác class → TRANSITIONING
static constexpr uint32_t SM_CONFIRM_IDLE_MS = 3000UL;  // 3s confirm → IDLE
static constexpr float    SM_MIN_CONF        = 0.80f;   // min confidence

// ─── State Machine States ──────────────────────────────────────
enum class HarState : uint8_t {
    IDLE          = 0,
    ACTIVE        = 1,
    TRANSITIONING = 2,
};

// ─── 1. Rep Detector (Adaptive Schmitt) ───────────────────────
class RepDetector {
public:
    RepDetector();

    // Đặt trục theo bài tập (gọi mỗi inference). Chỉ reset state khi ĐỔI trục
    // (giữ baseline ấm khi cùng trục → không mất rep giữa set).
    void configure(uint8_t axis);

    // Reset toàn bộ state nội bộ.
    void reset();

    // Cập nhật với 1 sample đã Z-score. Trả về true khi hoàn thành 1 chu kỳ rep.
    // norm_axes[6]: thứ tự ax, ay, az, gx, gy, gz (khớp norm_params.h)
    bool update(const float norm_axes[6]);

    float   getEnvelope() const { return envelope_; }
    float   getThreshold() const { return T_; }
    uint8_t getAxis()      const { return axis_; }

private:
    uint8_t axis_;

    // DC removal: slow EMA theo dõi baseline tín hiệu
    float   dc_ema_;
    float   warmup_sum_;   // tổng cộng dồn 0.5s đầu để seed baseline nhanh

    // MA ngắn làm mượt centered trước Schmitt (chống rung tần số cao)
    float   sm_buf_[PD_SMOOTH_WIN];
    int     sm_head_;
    int     sm_count_;
    float   sm_sum_;

    // Moving average trên rectified signal → ước lượng biên độ (envelope)
    float   ma_buf_[PD_MA_WIN];
    int     ma_head_;
    int     ma_count_;
    float   ma_sum_;
    float   envelope_;
    float   env_peak_;   // đỉnh-envelope giữ-chậm (peak hold + slow decay)

    // Ngưỡng thích nghi hiện tại & trạng thái Schmitt
    float   T_;
    bool    above_;   // tín hiệu (centered) đang ở nửa trên (đã vượt +T)
};

// ─── 2. State Machine ──────────────────────────────────────────
class StateMachine {
public:
    StateMachine();
    void reset();

    // Gọi sau mỗi lần inference. Cập nhật state, trả về state hiện tại.
    HarState update(int class_index, float confidence, uint32_t timestamp_ms);

    HarState getState()       const { return state_; }
    int      getActiveClass() const { return active_class_; }  // -1 khi không ACTIVE
    bool     stateChanged()   const { return state_changed_; }

    // Class đang được "đếm cho": ACTIVE → active_class_; đang gom streak → confirm_class_;
    // còn lại → -1. Dùng để cấu hình detector & bật đếm sớm (retro-count).
    int getCountingClass() const;

private:
    HarState state_;
    int      active_class_;
    bool     state_changed_;

    int      confirm_class_;   // class đang đếm streak (IDLE → ACTIVE)
    int      confirm_streak_;  // số inference liên tiếp cùng class
    int      exit_counter_;    // số inference liên tiếp khác class (ACTIVE → TRANS)

    uint32_t trans_start_ms_;  // thời điểm bắt đầu TRANSITIONING
};

// ─── 3. Rep Validator (debounce + retro-count) ─────────────────
class RepValidator {
public:
    RepValidator();

    // Reset về 0 mỗi đầu set (gọi khi vào IDLE/kết thúc set).
    void reset();

    // Gọi khi RepDetector fire.
    //   active   = state == ACTIVE
    //   counting = có counting-class >= 0 (đang gom streak hoặc active)
    //   min_gap_ms = debounce của bài tập hiện tại
    // Trả về true nếu được tính (rep thật hoặc pending) — để log.
    bool onPeak(uint32_t timestamp_ms, bool active, bool counting, uint32_t min_gap_ms);

    void commitPending();   // ACTIVE xác nhận → gộp pending vào rep_count
    void resetPending();    // streak vỡ trước khi ACTIVE → bỏ pending

    int getRepCount() const { return rep_count_; }
    int getPending()  const { return pending_; }

private:
    int      rep_count_;
    int      pending_;
    uint32_t last_rep_ms_;
};
