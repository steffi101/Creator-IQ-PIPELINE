"""
Database Service - SQLite

Stores:
- Campaign performance data (from CSV uploads)
- Pipeline cycle results (for feedback loop)
- Hook scores across cycles (for comparison)
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional


DB_PATH = "./data/creatoriq.db"


def get_connection():
    os.makedirs("./data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS campaigns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            app_name TEXT NOT NULL,
            app_description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS performance_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id INTEGER,
            hook_style TEXT,
            format TEXT,
            cpi REAL,
            ctr REAL,
            completion_rate REAL,
            installs INTEGER,
            spend REAL,
            days_live INTEGER,
            audience TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
        );

        CREATE TABLE IF NOT EXISTS cycles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id INTEGER,
            cycle_number INTEGER NOT NULL,
            trend_output TEXT,
            hook_output TEXT,
            script_output TEXT,
            feedback_output TEXT,
            hook_scores TEXT,
            rag_docs_used TEXT,
            perf_data_used BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
        );

        CREATE TABLE IF NOT EXISTS hook_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_id INTEGER,
            hook_text TEXT,
            brevity REAL,
            specificity REAL,
            emotion REAL,
            engagement REAL,
            interrupt REAL,
            native_feel REAL,
            composite INTEGER,
            grade TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (cycle_id) REFERENCES cycles(id)
        );
    """)

    conn.commit()
    conn.close()


# ---- Campaign Operations ----

def create_campaign(app_name: str, app_description: str = "") -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO campaigns (app_name, app_description) VALUES (?, ?)",
        (app_name, app_description)
    )
    conn.commit()
    campaign_id = cursor.lastrowid
    conn.close()
    return campaign_id


def get_campaign(campaign_id: int) -> Optional[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM campaigns WHERE id = ?", (campaign_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# ---- Performance Data Operations ----

def insert_performance_data(campaign_id: int, rows: list[dict]):
    conn = get_connection()
    cursor = conn.cursor()
    for row in rows:
        cursor.execute("""
            INSERT INTO performance_data
            (campaign_id, hook_style, format, cpi, ctr, completion_rate, installs, spend, days_live, audience)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            campaign_id,
            row.get("hook_style", ""),
            row.get("format", ""),
            row.get("cpi", 0),
            row.get("ctr", 0),
            row.get("completion_rate", 0),
            row.get("installs", 0),
            row.get("spend", 0),
            row.get("days_live", 0),
            row.get("audience", ""),
        ))
    conn.commit()
    conn.close()


def get_performance_summary(campaign_id: int) -> dict:
    conn = get_connection()
    cursor = conn.cursor()

    # Overall stats
    cursor.execute("""
        SELECT
            COUNT(*) as total_creatives,
            AVG(cpi) as avg_cpi,
            AVG(ctr) as avg_ctr,
            AVG(completion_rate) as avg_completion,
            SUM(installs) as total_installs,
            SUM(spend) as total_spend,
            MIN(cpi) as best_cpi,
            MAX(cpi) as worst_cpi
        FROM performance_data WHERE campaign_id = ?
    """, (campaign_id,))
    overall = dict(cursor.fetchone())

    # By hook style
    cursor.execute("""
        SELECT
            hook_style,
            AVG(cpi) as avg_cpi,
            AVG(ctr) as avg_ctr,
            COUNT(*) as count,
            SUM(installs) as installs
        FROM performance_data
        WHERE campaign_id = ?
        GROUP BY hook_style
        ORDER BY avg_cpi ASC
    """, (campaign_id,))
    by_hook = [dict(r) for r in cursor.fetchall()]

    # By format
    cursor.execute("""
        SELECT
            format,
            AVG(cpi) as avg_cpi,
            AVG(ctr) as avg_ctr,
            COUNT(*) as count
        FROM performance_data
        WHERE campaign_id = ?
        GROUP BY format
        ORDER BY avg_cpi ASC
    """, (campaign_id,))
    by_format = [dict(r) for r in cursor.fetchall()]

    # By audience
    cursor.execute("""
        SELECT
            audience,
            AVG(cpi) as avg_cpi,
            COUNT(*) as count,
            SUM(installs) as installs
        FROM performance_data
        WHERE campaign_id = ?
        GROUP BY audience
        ORDER BY avg_cpi ASC
    """, (campaign_id,))
    by_audience = [dict(r) for r in cursor.fetchall()]

    # Best and worst individual creatives
    cursor.execute("""
        SELECT * FROM performance_data
        WHERE campaign_id = ? ORDER BY cpi ASC LIMIT 3
    """, (campaign_id,))
    best = [dict(r) for r in cursor.fetchall()]

    cursor.execute("""
        SELECT * FROM performance_data
        WHERE campaign_id = ? ORDER BY cpi DESC LIMIT 3
    """, (campaign_id,))
    worst = [dict(r) for r in cursor.fetchall()]

    conn.close()

    return {
        "overall": overall,
        "by_hook_style": by_hook,
        "by_format": by_format,
        "by_audience": by_audience,
        "best_performers": best,
        "worst_performers": worst,
    }


# ---- Cycle Operations ----

def save_cycle(campaign_id: int, cycle_number: int, outputs: dict, hook_scores: list, rag_docs: list, used_perf_data: bool) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO cycles
        (campaign_id, cycle_number, trend_output, hook_output, script_output, feedback_output, hook_scores, rag_docs_used, perf_data_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        campaign_id, cycle_number,
        outputs.get("trend", ""),
        outputs.get("hook", ""),
        outputs.get("script", ""),
        outputs.get("feedback", ""),
        json.dumps([s if isinstance(s, dict) else s for s in hook_scores]),
        json.dumps(rag_docs),
        used_perf_data,
    ))
    conn.commit()
    cycle_id = cursor.lastrowid

    # Save individual hook evaluations
    for score in hook_scores:
        s = score if isinstance(score, dict) else score.__dict__ if hasattr(score, '__dict__') else {}
        cursor.execute("""
            INSERT INTO hook_evaluations
            (cycle_id, hook_text, brevity, specificity, emotion, engagement, interrupt, native_feel, composite, grade)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cycle_id,
            s.get("text", ""),
            s.get("brevity", 0),
            s.get("specificity", 0),
            s.get("emotion", 0),
            s.get("engagement", 0),
            s.get("interrupt", 0),
            s.get("native", s.get("native_feel", 0)),
            s.get("composite", 0),
            s.get("grade", ""),
        ))

    conn.commit()
    conn.close()
    return cycle_id


def get_cycle_history(campaign_id: int) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM cycles WHERE campaign_id = ? ORDER BY cycle_number ASC
    """, (campaign_id,))
    cycles = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return cycles


def get_latest_cycle_feedback(campaign_id: int) -> Optional[str]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT feedback_output FROM cycles
        WHERE campaign_id = ?
        ORDER BY cycle_number DESC LIMIT 1
    """, (campaign_id,))
    row = cursor.fetchone()
    conn.close()
    return row["feedback_output"] if row else None


def get_cycle_comparison(campaign_id: int) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.cycle_number,
               AVG(h.composite) as avg_score,
               COUNT(h.id) as hook_count,
               SUM(CASE WHEN h.grade = 'A' THEN 1 ELSE 0 END) as a_count,
               SUM(CASE WHEN h.grade = 'B' THEN 1 ELSE 0 END) as b_count
        FROM cycles c
        JOIN hook_evaluations h ON h.cycle_id = c.id
        WHERE c.campaign_id = ?
        GROUP BY c.cycle_number
        ORDER BY c.cycle_number
    """, (campaign_id,))
    results = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return results


# Initialize on import
init_db()
