"""Output-delay staging semantics: approve backdates, discard deletes.

Replicates the SQL used by _backdate_staged_rows / handle_discard_staged
against a real sqlite DB shaped like a session database, and checks it
against the emit-time age gate from emit_new_entries.
"""

import sqlite3
from datetime import datetime, timedelta

import pytest

FMT = "%Y-%m-%d %H:%M:%S"
DELAY = 7


@pytest.fixture
def db(tmp_path):
    conn = sqlite3.connect(tmp_path / "session.db")
    conn.execute(
        "CREATE TABLE transcriptions (id INTEGER PRIMARY KEY, timestamp TEXT,"
        " text TEXT, start_time REAL, end_time REAL)"
    )
    now = datetime.now()
    conn.execute("INSERT INTO transcriptions (timestamp, text) VALUES (?, ?)",
                 ((now - timedelta(seconds=60)).strftime(FMT), "old published line"))
    conn.execute("INSERT INTO transcriptions (timestamp, text) VALUES (?, ?)",
                 (now.strftime(FMT), "staged line A"))
    conn.execute("INSERT INTO transcriptions (timestamp, text) VALUES (?, ?)",
                 (now.strftime(FMT), "staged line B"))
    conn.commit()
    yield conn
    conn.close()


def staged_ids(conn):
    """The emit-time gate: rows younger than DELAY are staged."""
    out = []
    for rid, ts in conn.execute("SELECT id, timestamp FROM transcriptions"):
        if (datetime.now() - datetime.strptime(ts, FMT)).total_seconds() < DELAY:
            out.append(rid)
    return out


def backdated_ts():
    return (datetime.now() - timedelta(seconds=DELAY + 1)).strftime(FMT)


def test_fresh_rows_are_staged(db):
    assert staged_ids(db) == [2, 3]


def test_approve_single_publishes_immediately(db):
    db.execute("UPDATE transcriptions SET timestamp = ? WHERE id = ?", (backdated_ts(), 2))
    db.commit()
    assert staged_ids(db) == [3]


def test_discard_deletes_row(db):
    db.execute("DELETE FROM transcriptions WHERE id = ?", (3,))
    db.commit()
    assert staged_ids(db) == [2]
    assert db.execute("SELECT COUNT(*) FROM transcriptions").fetchone()[0] == 2


def test_approve_all_touches_only_staged_rows(db):
    cutoff = (datetime.now() - timedelta(seconds=DELAY)).strftime(FMT)
    db.execute("UPDATE transcriptions SET timestamp = ? WHERE timestamp > ?", (backdated_ts(), cutoff))
    db.commit()
    assert staged_ids(db) == []
    old_ts = db.execute("SELECT timestamp FROM transcriptions WHERE id = 1").fetchone()[0]
    age = (datetime.now() - datetime.strptime(old_ts, FMT)).total_seconds()
    assert 55 < age < 70, "published row must keep its original timestamp"
