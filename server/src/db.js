// SQLite — dùng module built-in node:sqlite (Node 22+), không cần biên dịch native.
// API (prepare/run/get/all) gần giống better-sqlite3.
import { DatabaseSync } from 'node:sqlite';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DB_PATH = process.env.DB_PATH || path.join(__dirname, '..', 'data.db');

export const db = new DatabaseSync(DB_PATH);
db.exec('PRAGMA journal_mode = WAL;');   // ghi nhanh, đọc/ghi song song tốt hơn
db.exec('PRAGMA foreign_keys = ON;');

db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    ended_at    TEXT,
    duration_ms INTEGER,
    total_sets  INTEGER,
    total_reps  INTEGER,
    active_ms   INTEGER,
    rest_ms     INTEGER,
    data        TEXT NOT NULL,           -- nguyên gói JSON từ Bluefy (gồm mảng sets)
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
  );

  CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, created_at DESC);
`);
