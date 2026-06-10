#pragma once
// rep_tracker.h — Bộ não đếm rep: state machine + 3 detector + vote/switch
// ─────────────────────────────────────────────────────────────
// Gộp 2 ý tưởng:
//   (1) +1 rep ngay khi commit bài (rep lead-in mà classify mới nhận ra được).
//   (2) Khoá bài (stickiness) + cho SỬA SAI khởi đầu trong ≤3 rep đầu bằng
//       vote M/K, KHÔNG mất rep nhờ 3 detector always-warm tích luỹ từ set start.
//
// Mapping class→detector: bicep(0)→D0, tricep(4)→D1, lateral(2)→D2;
// shoulder(3) → world-vertical (wv_, raw). Trục/delta: xem configDetectors()
// (nguồn chân lý duy nhất). 3 bài xoay dùng ax (gy đếm DƯ real-time: xoay/giật
// nhẹ cũng sinh swing gy; ay chết, az double-count).
//
// Cadence: onSample() mỗi mẫu (50Hz) chạy detector; onInference() mỗi 1.5s
// chạy state machine. idle do CLASSIFIER loại (detector chỉ chạy khi ACTIVE).
// ─────────────────────────────────────────────────────────────

#include "rep_detector.h"
#include "rep_detector_worldvert.h"   // shoulder_press: world-vertical (raw)
#include <cstdint>

class RepTracker {
public:
    enum State : uint8_t { IDLE, ACTIVE };
    enum Event : uint8_t { NONE, COMMITTED, SWITCHED, SET_CLOSED };

    // ── Tham số tune ──
    static constexpr float    CONF_TH         = 0.80f; // ngưỡng tin cậy
    static constexpr int      VOTE_K          = 4;     // cửa sổ vote
    static constexpr int      VOTE_M          = 3;     // M/K để switch (3/4)
    static constexpr int      SWITCH_MAX_REPS = 3;     // chỉ switch khi rep_display ≤ đây
    static constexpr int      IDLE_CLOSE      = 3;     // số window idle liên tiếp → chốt set
    static constexpr int      REP_BASE_CREDIT = 1;     // +1 rep lead-in lúc commit

    void reset() {
        state_ = IDLE; committed_ = -1; set_no_ = 0;
        for (int i = 0; i < N_DET; i++) det_count_[i] = 0;
        wv_.reset(); wv_count_ = 0;
        voteClear(); idle_streak_ = 0; closed_reps_ = 0;
    }

    // Gọi MỖI MẪU (cần z6 cho peakdet + raw6 cho world-vertical shoulder).
    // Trả về true nếu rep_display của bài đang commit vừa +1.
    bool onSample(const float z6[6], const float raw6[6], uint32_t now_ms) {
        if (state_ != ACTIVE) return false;
        bool committed_fired = false;
        // 3 bài xoay (bicep/tricep/lateral) — always-warm.
        for (int i = 0; i < N_DET; i++) {
            if (dets_[i].update(z6, now_ms)) {
                det_count_[i]++;
                if (i == detOf(committed_)) committed_fired = true;
            }
        }
        // shoulder_press dùng WORLD-VERTICAL (raw), không phải peakdet ay.
        if (wv_.update(raw6, now_ms)) {
            wv_count_++;
            if (committed_ == SHOULDER_CLS) committed_fired = true;
        }
        return committed_fired;
    }

    // Gọi MỖI LẦN INFERENCE. Trả về Event (COMMITTED/SWITCHED/SET_CLOSED/NONE).
    Event onInference(int cls, float conf, uint32_t now_ms) {
        bool confident = conf > CONF_TH;

        if (state_ == IDLE) {
            if (confident && isExercise(cls)) { commit(cls, now_ms); return COMMITTED; }
            return NONE;
        }

        // ── ACTIVE ──
        if (confident && cls == IDLE_IDX) {            // idle → đếm streak chốt set
            if (++idle_streak_ >= IDLE_CLOSE) { closeSet(); return SET_CLOSED; }
            return NONE;
        }
        idle_streak_ = 0;                              // bất kỳ non-idle → reset streak idle

        if (!confident) { votePush(-1); return NONE; } // low-conf = phiếu trắng

        votePush(cls);
        if (cls != committed_ && repDisplay() <= SWITCH_MAX_REPS
                && votesFor(cls) >= VOTE_M) {
            switchTo(cls);                             // sửa sai khởi đầu
            return SWITCHED;
        }
        return NONE;
    }

    // ── Getters ──
    State   state()      const { return state_; }
    int     committed()  const { return committed_; }
    int     setNo()      const { return set_no_; }
    int     closedReps() const { return closed_reps_; }   // rep của set vừa chốt
    int repDisplay() const {
        if (committed_ < 0) return 0;
        if (committed_ == SHOULDER_CLS) return REP_BASE_CREDIT + wv_count_;
        return REP_BASE_CREDIT + det_count_[detOf(committed_)];
    }
    uint32_t lastShoulderUpMs()   const { return wv_.lastUpMs(); }    // debug/tune
    uint32_t lastShoulderDownMs() const { return wv_.lastDownMs(); }
    uint32_t lastSwingMs() const {                                   // bài xoay (peakdet)
        if (committed_ < 0 || committed_ == SHOULDER_CLS) return 0;
        return dets_[detOf(committed_)].lastSwingMs();
    }

private:
    // class index: 0 bicep, 1 idle, 2 lateral, 3 shoulder, 4 tricep
    static constexpr int IDLE_IDX     = 1;
    static constexpr int SHOULDER_CLS = 3;   // dùng world-vertical thay peakdet
    static int  detOf(int cls) {                       // -1 nếu idle/shoulder/không hợp lệ
        static const int8_t M[5] = {0, -1, 2, -1, 1}; // bicep→D0 tricep→D1 lateral→D2 (shoulder→wv_)
        return (cls >= 0 && cls < 5) ? M[cls] : -1;
    }
    static bool isExercise(int c) { return c >= 0 && c < 5 && c != IDLE_IDX; }

    void configDetectors() {
        // configure(axis, delta, min_gap_ms, smooth_win, reset_timeout_ms, min_abs[, min_swing_ms])
        // min_swing_ms=500: anti-nhún — swing đáy→đỉnh phải ≥500ms mới tính rep.
        dets_[0].configure(0, 2.0f, 600, 5, 3000, 0.0f, /*min_swing_ms*/ 500);   // bicep: ax
        dets_[1].configure(0, 1.1f, 600, 5, 3000, 0.0f, /*min_swing_ms*/ 500);   // tricep: ax
        dets_[2].configure(0, 1.3f, 600, 5, 3000, 0.0f, /*min_swing_ms*/ 500);   // lateral: ax
        for (int i = 0; i < N_DET; i++) det_count_[i] = 0;
        // shoulder_press: world-vertical (raw). Đếm theo DURATION nửa nhịp (anti-nhún):
        // pha lên đủ dài RỒI pha xuống đủ dài = 1 rep. min_phase=450ms loại nhún
        // nhanh (>~1Hz). thr=deadband đổi pha. Tune min_phase theo log up/down ms.
        wv_.configure(/*thr*/ 0.012f, /*min_gap_ms*/ 500, /*alpha*/ 0.02f,
                      /*leak*/ 0.985f, /*min_up_ms*/ 900, /*min_down_ms*/ 350);
        wv_count_ = 0;
    }

    void commit(int cls, uint32_t /*now_ms*/) {
        state_ = ACTIVE; committed_ = cls; set_no_++;
        configDetectors();                             // reset + warm 4 detector
        idle_streak_ = 0; voteClear();
    }
    void switchTo(int cls) {
        committed_ = cls;                              // đổi nhãn; det count giữ nguyên (đã tích luỹ)
        voteClear();                                   // cần bằng chứng mới cho lần switch kế
    }
    void closeSet() {
        closed_reps_ = repDisplay();
        state_ = IDLE; committed_ = -1; idle_streak_ = 0; voteClear();
    }

    // ── Vote ring (K phiếu gần nhất; -1 = trắng) ──
    void voteClear() { for (int i = 0; i < VOTE_K; i++) vote_[i] = -1; vote_head_ = 0; }
    void votePush(int cls) { vote_[vote_head_] = (int8_t)cls; vote_head_ = (vote_head_ + 1) % VOTE_K; }
    int  votesFor(int cls) const {
        int n = 0; for (int i = 0; i < VOTE_K; i++) if (vote_[i] == cls) n++; return n;
    }

    static constexpr int N_DET = 3;                    // bicep,tricep,lateral (shoulder = wv_)

    State        state_      = IDLE;
    int          committed_  = -1;
    int          set_no_     = 0;
    int          closed_reps_= 0;
    RepDetector       dets_[N_DET];
    int               det_count_[N_DET] = {0, 0, 0};
    WorldVertDetector wv_;                 // shoulder_press (raw, world-vertical)
    int               wv_count_ = 0;
    int8_t       vote_[VOTE_K] = {-1, -1, -1, -1};
    int          vote_head_  = 0;
    int          idle_streak_= 0;
};
