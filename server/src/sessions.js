// Lưu / đọc các buổi tập của user đang đăng nhập.
import express from 'express';
import { db } from './db.js';
import { requireAuth } from './auth.js';

export const sessionsRouter = express.Router();
sessionsRouter.use(requireAuth);   // mọi route ở đây đều cần token

// POST /api/sessions — nhận nguyên gói payload từ Bluefy (buildSessionPayload).
sessionsRouter.post('/', (req, res) => {
  const p = req.body || {};
  if (!p || typeof p !== 'object' || !Array.isArray(p.sets))
    return res.status(400).json({ error: 'Payload không hợp lệ (thiếu sets)' });

  const t = p.totals || {};
  const info = db.prepare(`
    INSERT INTO sessions (user_id, ended_at, duration_ms, total_sets, total_reps, active_ms, rest_ms, data)
    VALUES (@uid, @endedAt, @durationMs, @sets, @reps, @activeMs, @restMs, @data)
  `).run({
    uid:        req.user.uid,
    endedAt:    p.endedAt ?? null,
    durationMs: p.durationMs ?? null,
    sets:       t.sets ?? p.sets.length,
    reps:       t.reps ?? null,
    activeMs:   t.activeMs ?? null,
    restMs:     t.restMs ?? null,
    data:       JSON.stringify(p),
  });
  res.status(201).json({ id: info.lastInsertRowid });
});

// GET /api/sessions — danh sách tóm tắt (kèm rep theo từng bài), mới nhất trước.
sessionsRouter.get('/', (req, res) => {
  const rows = db.prepare(`
    SELECT id, ended_at, duration_ms, total_sets, total_reps, active_ms, rest_ms, data, created_at
    FROM sessions WHERE user_id = ? ORDER BY created_at DESC
  `).all(req.user.uid);
  const out = rows.map(r => {
    const byExercise = {};                         // { bicep_curl: 24, ... } cho biểu đồ chồng màu
    try {
      for (const s of (JSON.parse(r.data).sets || []))
        byExercise[s.exercise] = (byExercise[s.exercise] || 0) + (s.reps || 0);
    } catch {}
    const { data, ...rest } = r;                   // bỏ data thô khỏi list cho nhẹ
    return { ...rest, byExercise };
  });
  res.json(out);
});

// GET /api/sessions/:id — chi tiết 1 buổi (kèm mảng sets đã parse).
sessionsRouter.get('/:id', (req, res) => {
  const row = db.prepare('SELECT * FROM sessions WHERE id = ? AND user_id = ?')
                .get(req.params.id, req.user.uid);
  if (!row) return res.status(404).json({ error: 'Không tìm thấy buổi tập' });
  res.json({ ...row, data: JSON.parse(row.data) });
});

// DELETE /api/sessions/:id — xoá 1 buổi.
sessionsRouter.delete('/:id', (req, res) => {
  const info = db.prepare('DELETE FROM sessions WHERE id = ? AND user_id = ?')
                 .run(req.params.id, req.user.uid);
  if (info.changes === 0) return res.status(404).json({ error: 'Không tìm thấy buổi tập' });
  res.json({ ok: true });
});
