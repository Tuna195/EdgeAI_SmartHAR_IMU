// Đăng ký / đăng nhập bằng email + mật khẩu, cấp JWT. Mật khẩu băm bằng bcrypt.
import express from 'express';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { db } from './db.js';

const SECRET  = process.env.JWT_SECRET || 'dev-insecure-secret-change-me';
const EXPIRES = '30d';

if (!process.env.JWT_SECRET)
  console.warn('⚠  JWT_SECRET chưa đặt — đang dùng secret mặc định KHÔNG an toàn. Đặt trong .env trước khi deploy.');

export const authRouter = express.Router();

function sign(user) {
  return jwt.sign({ uid: user.id, email: user.email }, SECRET, { expiresIn: EXPIRES });
}

// Middleware: bắt buộc có "Authorization: Bearer <token>" hợp lệ.
export function requireAuth(req, res, next) {
  const h = req.headers.authorization || '';
  const token = h.startsWith('Bearer ') ? h.slice(7) : null;
  if (!token) return res.status(401).json({ error: 'Thiếu token' });
  try {
    req.user = jwt.verify(token, SECRET);   // { uid, email, iat, exp }
    next();
  } catch {
    res.status(401).json({ error: 'Token không hợp lệ hoặc đã hết hạn' });
  }
}

authRouter.post('/register', (req, res) => {
  const { email, password } = req.body || {};
  if (!email || !password) return res.status(400).json({ error: 'Thiếu email hoặc mật khẩu' });
  if (String(password).length < 6) return res.status(400).json({ error: 'Mật khẩu tối thiểu 6 ký tự' });

  const hash = bcrypt.hashSync(String(password), 10);
  try {
    const info = db.prepare('INSERT INTO users (email, password_hash) VALUES (?, ?)')
                   .run(String(email).toLowerCase(), hash);
    const user = { id: info.lastInsertRowid, email: String(email).toLowerCase() };
    res.status(201).json({ token: sign(user), user });
  } catch (e) {
    if (String(e).includes('UNIQUE')) return res.status(409).json({ error: 'Email đã được đăng ký' });
    throw e;
  }
});

authRouter.post('/login', (req, res) => {
  const { email, password } = req.body || {};
  if (!email || !password) return res.status(400).json({ error: 'Thiếu email hoặc mật khẩu' });

  const user = db.prepare('SELECT * FROM users WHERE email = ?').get(String(email).toLowerCase());
  if (!user || !bcrypt.compareSync(String(password), user.password_hash))
    return res.status(401).json({ error: 'Sai email hoặc mật khẩu' });

  res.json({ token: sign(user), user: { id: user.id, email: user.email } });
});

// Thông tin user hiện tại (kiểm tra token còn sống).
authRouter.get('/me', requireAuth, (req, res) => {
  const u = db.prepare('SELECT id, email, created_at FROM users WHERE id = ?').get(req.user.uid);
  if (!u) return res.status(404).json({ error: 'User không tồn tại' });
  res.json(u);
});
