// Entry point — Express app: auth + sessions API.
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { authRouter } from './auth.js';
import { sessionsRouter } from './sessions.js';

const app = express();

// CORS: cho phép trang Bluefy (origin khác) gọi. CORS_ORIGIN rỗng = cho phép tất cả (dev).
const origins = (process.env.CORS_ORIGIN || '').split(',').map(s => s.trim()).filter(Boolean);
app.use(cors(origins.length ? { origin: origins } : {}));
app.use(express.json({ limit: '256kb' }));

app.get('/api/health', (req, res) => res.json({ ok: true, time: new Date().toISOString() }));
app.use('/api/auth', authRouter);
app.use('/api/sessions', sessionsRouter);

// 404 cho route không khớp
app.use((req, res) => res.status(404).json({ error: 'Not found' }));

// Bắt lỗi tập trung
app.use((err, req, res, _next) => {
  console.error(err);
  res.status(500).json({ error: 'Lỗi server' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`EdgeAI workout server: http://localhost:${PORT}`));
