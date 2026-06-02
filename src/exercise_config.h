#pragma once
// exercise_config.h
// ─────────────────────────────────────────────────────────────
// Bảng cấu hình per-exercise cho Rep Detector (Adaptive Schmitt).
// Re-derive từ training/rep_sim.py trên tín hiệu Z-score RAW
// (khớp 100% với norm_params.h — firmware KHÔNG còn EMA front-end).
//
// best_axis: 0=ax 1=ay 2=az 3=gx 4=gy 5=gz
//   → chọn theo VARIANCE (trục gyro xoay theo giải phẫu), NHẤT QUÁN
//     giữa tay trái/phải. (Config cũ dùng ay/gy là sai trục.)
// min_gap_ms: debounce tối thiểu giữa 2 rep (≈0.5×chu kỳ rep).
//
// Detector KHÔNG còn dùng ngưỡng tuyệt đối — ngưỡng tự thích nghi
// theo biên độ tức thời (xem PD_K_BAND / PD_T_MIN trong rep_counter.h).
// ─────────────────────────────────────────────────────────────

#include <cstdint>

struct ExerciseConfig {
    const char* name;
    uint8_t     best_axis;    // index trục tốt nhất [0..5]
    uint16_t    min_gap_ms;   // debounce tối thiểu mỗi rep
};

// Indexed by HarResult.class_index [0..4]
// ┌────────────────┬─────┬───────┬──────────┬──────────────────┐
// │ Class          │ idx │ axis  │ min_gap  │ chu kỳ (rep_sim) │
// ├────────────────┼─────┼───────┼──────────┼──────────────────┤
// │ bicep_curl     │ 0   │ gy(4) │ 2000 ms  │ ~4.0s            │
// │ idle           │ 1   │  —    │   —      │ (không dùng)     │
// │ lateral_raise  │ 2   │ gy(4) │  850 ms  │ ~1.7s (½ chu kỳ) │
// │ shoulder_press │ 3   │ gx(3) │ 1910 ms  │ ~3.8s            │
// │ tricep_ext     │ 4   │ gz(5) │  970 ms  │ ~1.9s (½ chu kỳ) │
// └────────────────┴─────┴───────┴──────────┴──────────────────┘
static const ExerciseConfig EXERCISE_CONFIGS[5] = {
    {"bicep_curl",      4, 2000},  // [0]
    {"idle",            0,    0},  // [1] không dùng
    {"lateral_raise",   4,  850},  // [2]
    {"shoulder_press",  3, 1910},  // [3]
    {"tricep_ext",      5,  970},  // [4]
};

static constexpr int IDLE_CLASS_INDEX = 1;
static constexpr int NUM_EXERCISE_CLASSES = 5;

// Trả về nullptr nếu class_index không hợp lệ hoặc là idle.
inline const ExerciseConfig* getExerciseConfig(int class_index) {
    if (class_index < 0 || class_index >= NUM_EXERCISE_CLASSES) return nullptr;
    if (class_index == IDLE_CLASS_INDEX)                         return nullptr;
    return &EXERCISE_CONFIGS[class_index];
}
