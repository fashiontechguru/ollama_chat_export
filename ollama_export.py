#!/usr/bin/env python3
r""" 
ollama_export.py

Purpose
-------
Export Ollama Desktop chats from the local SQLite database (db.sqlite) into
a structured folder tree:

  .\chat_export\YYYY_MM_DD\YYYY_MM_DD_chat_001.txt
  .\chat_export\YYYY_MM_DD\YYYY_MM_DD_chat_002.txt
  ...
  .\chat_export\YYYY_MM_DD\YYYY_MM_DD_chat_###_attachment_A.<ext>
  .\chat_export\YYYY_MM_DD\YYYY_MM_DD_chat_###_attachment_B.<ext>
  ...

Also writes:
  .\chat_export\index.csv         (master index of all known chats and file paths)
  .\chat_export\export_log.txt    (append-only operational log)

Safety gate (IMPORTANT)
-----------------------
This exporter will ONLY run when db.sqlite-wal is empty (0 bytes). If the WAL file
is non-empty, the script fails fast and prints guidance to close the Ollama GUI:

  - Close the Ollama Desktop GUI completely
  - Wait a few seconds
  - Re-run this script

Rationale: a non-empty WAL implies pending writes that may not be reflected in db.sqlite
and exports can be incomplete or inconsistent.

Usage
-----
Run from the Ollama folder (where db.sqlite lives), e.g.:

  python .\ollama_export.py

Include thinking blocks (can be large):

  python .\ollama_export.py --full

Notes
-----
- Date folders are based on chats.created_at, interpreted as UTC and converted to
  America/New_York for folder naming and age calculations.
- Overwrite rules:
    * Chats whose (New York) start date is today or yesterday: ALWAYS overwrite.
    * Chats started 2+ days ago: skip if transcript already exists.
- Filenames are sequential per date: YYYY_MM_DD_chat_001.txt, 002, etc.
  Stability across runs is maintained by index.csv mapping chat_id -> filename.
- Attachments are exported per chat using letter suffixes A, B, C... and preserve
  the original file extension if present.

Credit
-----
Created January 21st, 2026 by Charles Beckwith (@fashiontechguru), with assistance from ChatGPT.

"Why is there no export button?" - Charles Beckwith

"""

import argparse
import csv
import datetime as dt
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    # Python 3.9+ standard library timezone support
    from zoneinfo import ZoneInfo
except Exception as e:
    print("ERROR: Python zoneinfo is required (Python 3.9+).")
    print(f"DETAILS: {e}")
    sys.exit(2)


# -----------------------------
# Configuration constants
# -----------------------------

DB_FILENAME = "db.sqlite"
WAL_FILENAME = "db.sqlite-wal"

EXPORT_ROOT_FOLDER = "chat_export"
INDEX_CSV_NAME = "index.csv"
EXPORT_LOG_NAME = "export_log.txt"

# Timezone policy: interpret DB timestamps as UTC, convert to America/New_York
TZ_UTC = ZoneInfo("UTC")
TZ_NY = ZoneInfo("America/New_York")

# "Old chat" threshold in days for skip policy
OLD_CHAT_DAYS_THRESHOLD = 2

# Windows reserved device names (case-insensitive) that cannot be used as filenames
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ChatRow:
    chat_id: str
    title: str
    created_at_raw: str  # raw string from DB
    created_at_utc: dt.datetime
    created_at_ny: dt.datetime
    folder_date: str     # YYYY_MM_DD


@dataclass
class IndexRow:
    chat_id: str
    chat_title: str
    chat_created_at_utc: str
    chat_created_at_ny: str
    folder_date: str
    transcript_relpath: str
    exported_at_ny: str
    export_status: str
    message_count: int
    tool_call_count: int
    attachment_count: int
    used_full: int


# -----------------------------
# Utility helpers
# -----------------------------

def crlf_join(lines: List[str]) -> str:
    """
    Join lines using explicit Windows CRLF newlines.

    We explicitly emit CRLF to match your requirement, rather than relying on
    platform newline translation behavior.
    """
    return "\r\n".join(lines) + "\r\n"


def ensure_dir(path: str) -> None:
    """
    Ensure a directory exists. If it does not exist, create it.
    """
    os.makedirs(path, exist_ok=True)


def append_log_line(log_path: str, line: str) -> None:
    """
    Append one line to the export log using CRLF newlines.
    """
    # Use newline="" and explicit CRLF in content so Python does not normalize.
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        f.write(line + "\r\n")


def fail_fast_if_wal_nonempty(base_dir: str) -> None:
    """
    Safety gate: abort if db.sqlite-wal exists and is non-empty.

    This ensures exports are stable and not missing recent messages stuck in WAL.
    """
    wal_path = os.path.join(base_dir, WAL_FILENAME)
    if os.path.exists(wal_path):
        try:
            size = os.path.getsize(wal_path)
        except Exception as e:
            print("ERROR: Could not stat WAL file.")
            print(f"PATH: {wal_path}")
            print(f"DETAILS: {e}")
            sys.exit(2)

        if size > 0:
            print("ERROR: WAL file is non-empty. Export aborted for safety.")
            print(f"WAL PATH: {wal_path}")
            print(f"WAL SIZE: {size} bytes")
            print("")
            print("ACTION REQUIRED:")
            print("  1) Close the Ollama Desktop GUI completely.")
            print("  2) Wait a few seconds for it to flush/close the database.")
            print("  3) Re-run this exporter.")
            sys.exit(1)


def parse_db_timestamp_as_utc(ts: str) -> dt.datetime:
    """
    Parse a SQLite TIMESTAMP string as a timezone-aware UTC datetime.

    Supported forms (most common first):
      1) "YYYY-MM-DD HH:MM:SS"            (SQLite CURRENT_TIMESTAMP)
      2) "YYYY-MM-DD HH:MM:SS.SSS"        (fractional seconds)
      3) ISO-like: "YYYY-MM-DDTHH:MM:SS"  (optional fractional)
      4) ISO with offset: "...+HH:MM" or "...-HH:MM"
      5) ISO with Z: "...Z"

    Policy (matches your requirements):
    - If no timezone is present in the string, treat it as UTC.
    - If a timezone/offset is present, normalize to UTC.

    Failure mode:
    - Raises ValueError with an explicit message if the timestamp cannot be parsed.
    """
    ts = (ts or "").strip()
    if not ts:
        raise ValueError("Empty timestamp string")

    # 1) SQLite CURRENT_TIMESTAMP (UTC, no fractional seconds)
    try:
        naive = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return naive.replace(tzinfo=TZ_UTC)
    except Exception:
        pass

    # 2) Fractional seconds
    try:
        naive = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
        return naive.replace(tzinfo=TZ_UTC)
    except Exception:
        pass

    # 3/4/5) ISO-like
    # Python fromisoformat does not accept trailing "Z", so normalize to +00:00.
    iso = ts
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"

    try:
        parsed = dt.datetime.fromisoformat(iso)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=TZ_UTC)
        return parsed.astimezone(TZ_UTC)
    except Exception:
        raise ValueError(f"Unrecognized timestamp format: {ts!r}")


def safe_attachment_extension(original_filename: str, max_len: int = 24) -> str:
    """
    Build a safe extension string for exported attachment files.

    Goals:
    - Preserve multi-extensions deterministically when they appear.
      Example: "archive.tar.gz" -> ".tar.gz"
    - Keep it Windows-safe and ASCII-friendly.
    - Avoid pathological long extensions.

    Rules:
    - If the filename is empty or has no dot -> "" (no extension).
    - If it begins with a single dot and no other dots (e.g. ".env") -> "".
    - If multiple dots exist, preserve only the last 2 segments.
      Example: "thing.backup.final.jpeg" -> ".final.jpeg"
    - Each segment is sanitized using sanitize_filename_component().

    Returns:
    - "" or ".ext" or ".tar.gz"
    """
    name = (original_filename or "").strip()
    if not name:
        return ""

    base = os.path.basename(name)

    # ".env" should not become ".env" as an "extension" in our exporter naming
    if base.startswith(".") and base.count(".") == 1:
        return ""

    parts = base.split(".")
    if len(parts) < 2:
        return ""

    ext_parts = parts[1:]

    # Preserve last two segments if there are multiple
    if len(ext_parts) > 2:
        ext_parts = ext_parts[-2:]

    cleaned: List[str] = []
    for p in ext_parts:
        seg = sanitize_filename_component(p, max_len=12)
        seg = seg.strip("._")
        if seg:
            cleaned.append(seg)

    if not cleaned:
        return ""

    ext = "." + ".".join(cleaned)

    # Clamp final extension length (rare)
    if len(ext) > max_len:
        ext = ext[:max_len].rstrip(" .")

    return ext


def sanitize_filename_component(s: str, max_len: int = 120) -> str:
    """
    Sanitize a string to be safe as a Windows filename component.

    Your specific whitespace rule:
      - Convert runs of spaces/tabs to underscores:
          * 1-3 whitespace chars -> 1-3 underscores (same count)
          * 4+ whitespace chars  -> exactly 3 underscores

    Additional rules:
      - Replace invalid Windows filename characters with underscores.
      - Strip trailing spaces and trailing dots.
      - Avoid Windows reserved device names.
      - Enforce a max length to reduce path-length issues.

    Note: This sanitizer is used for attachment filename fragments (and could be
    reused elsewhere). Chat transcript filenames are numeric per your rule.
    """
    if s is None:
        s = ""

    # Normalize whitespace to a deterministic underscore pattern per your rule.
    # We treat spaces and tabs as whitespace runs to convert.
    def _ws_repl(m: re.Match) -> str:
        n = len(m.group(0))
        if n <= 3:
            return "_" * n
        return "___"

    s = s.strip()
    s = re.sub(r"[ \t]+", _ws_repl, s)

    # Replace invalid filename characters with underscore
    # Invalid in Windows: < > : " / \ | ? *
    s = re.sub(r'[<>:"/\\|?*]', "_", s)

    # Remove control characters (ASCII 0-31) which can cause filesystem issues
    s = "".join(ch if ord(ch) >= 32 else "_" for ch in s)

    # Strip trailing spaces and dots (Windows restriction)
    s = s.rstrip(" .")

    # Prevent empty result
    if not s:
        s = "unnamed"

    # Avoid reserved device names
    if s.upper() in WINDOWS_RESERVED_NAMES:
        s = "_" + s

    # Enforce maximum length
    if len(s) > max_len:
        s = s[:max_len].rstrip(" .")
        if not s:
            s = "unnamed"

    return s


def compute_chat_age_days(chat_date_ny: dt.date, today_ny: dt.date) -> int:
    """
    Compute age in days (today - chat_date) in America/New_York local date terms.
    """
    return (today_ny - chat_date_ny).days


def letter_suffix(n: int) -> str:
    """
    Convert a 0-based index to Excel-like letter suffix:
      0 -> A
      1 -> B
      ...
      25 -> Z
      26 -> AA
      27 -> AB
    """
    if n < 0:
        return "A"
    out = []
    x = n
    while True:
        x, r = divmod(x, 26)
        out.append(chr(ord("A") + r))
        if x == 0:
            break
        x -= 1
    return "".join(reversed(out))


# -----------------------------
# SQLite access helpers
# -----------------------------

def open_db_readonly(db_path: str) -> sqlite3.Connection:
    """
    Open SQLite database in read-only mode.

    We use URI mode to enforce read-only access where supported:
      file:db.sqlite?mode=ro

    If URI open fails, fall back to normal connect, but still do not write.
    """
    # First attempt: explicit read-only URI
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        return conn
    except Exception:
        # Fallback: normal open (still read-only by convention)
        conn = sqlite3.connect(db_path)
        return conn


def load_chats(conn: sqlite3.Connection) -> List[ChatRow]:
    """
    Load chats from the chats table.

    The schema shows:
      chats(id TEXT PRIMARY KEY, title TEXT, created_at TIMESTAMP, browser_state TEXT)

    We interpret created_at as UTC and convert to America/New_York.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, created_at
        FROM chats
        ORDER BY created_at ASC, id ASC
    """)
    rows = cur.fetchall()

    chats: List[ChatRow] = []
    for (chat_id, title, created_at_raw) in rows:
        # Defensive conversion for None
        title = title or ""
        created_at_raw = created_at_raw or ""

        created_at_utc = parse_db_timestamp_as_utc(created_at_raw)
        created_at_ny = created_at_utc.astimezone(TZ_NY)

        folder_date = created_at_ny.strftime("%Y_%m_%d")

        chats.append(ChatRow(
            chat_id=str(chat_id),
            title=str(title),
            created_at_raw=str(created_at_raw),
            created_at_utc=created_at_utc,
            created_at_ny=created_at_ny,
            folder_date=folder_date,
        ))

    return chats


def load_messages_for_chat(conn: sqlite3.Connection, chat_id: str) -> List[dict]:
    """
    Load all messages for a given chat_id, in chronological order.

    We keep fields needed for transcript formatting and tool call linkage.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT
            id,
            created_at,
            role,
            COALESCE(model_name, '') AS model_name,
            content,
            thinking,
            COALESCE(tool_result, '') AS tool_result
        FROM messages
        WHERE chat_id = ?
        ORDER BY created_at ASC, id ASC
    """, (chat_id,))
    rows = cur.fetchall()

    msgs: List[dict] = []
    for (mid, created_at_raw, role, model_name, content, thinking, tool_result) in rows:
        created_at_raw = created_at_raw or ""
        created_at_utc = parse_db_timestamp_as_utc(created_at_raw)
        created_at_ny = created_at_utc.astimezone(TZ_NY)

        msgs.append({
            "id": int(mid),
            "created_at_raw": str(created_at_raw),
            "created_at_utc": created_at_utc,
            "created_at_ny": created_at_ny,
            "role": str(role or ""),
            "model_name": str(model_name or ""),
            "content": str(content or ""),
            "thinking": str(thinking or ""),
            "tool_result": str(tool_result or ""),
        })

    return msgs


def load_tool_calls_for_message(conn: sqlite3.Connection, message_id: int) -> List[dict]:
    """
    DEPRECATED (kept for compatibility / debugging):

    The exporter now uses load_tool_calls_for_chat(chat_id) to bulk-load tool calls
    for an entire chat in a single query (faster, fewer DB round trips).

    This function remains available for ad-hoc debugging of a specific message_id.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT
            id,
            type,
            function_name,
            function_arguments,
            COALESCE(function_result, '') AS function_result
        FROM tool_calls
        WHERE message_id = ?
        ORDER BY id ASC
    """, (message_id,))
    rows = cur.fetchall()

    out: List[dict] = []
    for (tc_id, tc_type, fn_name, fn_args, fn_result) in rows:
        out.append({
            "id": int(tc_id),
            "type": str(tc_type or ""),
            "function_name": str(fn_name or ""),
            "function_arguments": str(fn_args or ""),
            "function_result": str(fn_result or ""),
        })
    return out


def load_attachments_for_chat(conn: sqlite3.Connection, chat_id: str) -> List[dict]:
    """
    Load all attachments for a given chat_id.

    Attachments table is keyed by message_id, so we join messages -> attachments.

    Schema:
      attachments(message_id, filename, data BLOB)

    We return a list ordered by message created_at and attachment id to keep exports stable.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT
            a.id,
            a.message_id,
            COALESCE(a.filename, '') AS filename,
            a.data
        FROM attachments a
        JOIN messages m ON m.id = a.message_id
        WHERE m.chat_id = ?
        ORDER BY m.created_at ASC, a.id ASC
    """, (chat_id,))
    rows = cur.fetchall()

    out: List[dict] = []
    for (att_id, msg_id, filename, blob) in rows:
        out.append({
            "id": int(att_id),
            "message_id": int(msg_id),
            "filename": str(filename or ""),
            "data": blob,  # bytes
        })
    return out


def cleanup_existing_chat_attachments(base_dir: str, folder_date: str, seq: int) -> int:
    """
    Remove previously exported attachment files for one chat sequence within a date folder.

    Why:
    - When overwriting today/yesterday exports, stale attachment files can remain.
      Example: yesterday the chat had A/B/C, today it only has A. We must delete old B/C
      so the folder reflects the DB state.

    Safety:
    - We only delete files in:
        chat_export\\{folder_date}\\
      that begin with:
        {folder_date}_chat_{seq:03d}_attachment_
    - We never delete directories.
    - Fail-soft: if a file cannot be removed, we do not crash the entire export.

    Returns:
    - Number of files deleted.
    """
    date_folder = os.path.join(base_dir, EXPORT_ROOT_FOLDER, folder_date)
    if not os.path.isdir(date_folder):
        return 0

    prefix = f"{folder_date}_chat_{seq:03d}_attachment_"
    deleted = 0

    try:
        for name in os.listdir(date_folder):
            if not name.startswith(prefix):
                continue

            abs_path = os.path.join(date_folder, name)
            if not os.path.isfile(abs_path):
                continue

            try:
                os.remove(abs_path)
                deleted += 1
            except Exception:
                # Intentionally ignore individual delete failures.
                # The export will still attempt to write current attachment files.
                pass
    except Exception:
        return deleted

    return deleted


def load_tool_calls_for_chat(conn: sqlite3.Connection, chat_id: str) -> Dict[int, List[dict]]:
    """
    Bulk-load all tool calls for an entire chat in a single SQL query.

    Why:
    - Your previous approach loaded tool calls per message (N queries per chat),
      which can be slow for large histories.
    - This function performs one query per chat and returns a mapping:
        { message_id: [tool_call_dict, ...], ... }
      where tool calls are in stable, deterministic order.

    Determinism:
    - We order by message created_at, then tool_call id.
    - Within each message_id, tool calls are returned in id order.

    Output format:
    - tool_calls_by_message_id[message_id] = [
          {
            "id": int,
            "type": str,
            "function_name": str,
            "function_arguments": str,
            "function_result": str,
          }, ...
      ]
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT
            tc.id,
            tc.message_id,
            COALESCE(tc.type, '') AS type,
            COALESCE(tc.function_name, '') AS function_name,
            COALESCE(tc.function_arguments, '') AS function_arguments,
            COALESCE(tc.function_result, '') AS function_result
        FROM tool_calls tc
        JOIN messages m ON m.id = tc.message_id
        WHERE m.chat_id = ?
        ORDER BY m.created_at ASC, tc.id ASC
    """, (chat_id,))
    rows = cur.fetchall()

    out: Dict[int, List[dict]] = {}
    for (tc_id, message_id, tc_type, fn_name, fn_args, fn_result) in rows:
        mid = int(message_id)
        out.setdefault(mid, []).append({
            "id": int(tc_id),
            "type": str(tc_type or ""),
            "function_name": str(fn_name or ""),
            "function_arguments": str(fn_args or ""),
            "function_result": str(fn_result or ""),
        })

    return out


# -----------------------------
# Index handling
# -----------------------------

def index_csv_path(base_dir: str) -> str:
    return os.path.join(base_dir, EXPORT_ROOT_FOLDER, INDEX_CSV_NAME)


def export_log_path(base_dir: str) -> str:
    return os.path.join(base_dir, EXPORT_ROOT_FOLDER, EXPORT_LOG_NAME)


def load_existing_index(path: str, log_path: Optional[str] = None) -> Dict[str, dict]:
    """
    Load existing index.csv if present.

    Returns:
    - dict: chat_id -> row dict

    Why this matters:
    - Your transcript filenames are numeric per date (chat_001, chat_002, ...).
      The index is the authoritative map that preserves stability across runs.

    Behavior:
    - If index.csv is missing -> returns {}.
    - If index.csv exists but cannot be parsed -> returns {} AND logs a warning line
      to export_log.txt (when log_path is provided).

    Important:
    - We intentionally do not crash if the index is corrupt, because you may still want
      the exporter to run. However, we fail-loud via logging, because corrupt index
      means chat numbers could be reassigned.
    """
    if not os.path.exists(path):
        return {}

    out: Dict[str, dict] = {}
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = (row.get("chat_id") or "").strip()
                if cid:
                    out[cid] = row
        return out

    except Exception as e:
        if log_path:
            append_log_line(
                log_path,
                f"INDEX LOAD WARNING: Failed to parse index.csv; proceeding without it. error={e}"
            )
        return {}


def write_index(path: str, rows: List[IndexRow]) -> None:
    """
    Write the master index.csv (fully rebuilt each run).

    We rebuild each run so the index reflects current export outcomes and stays clean.
    """
    ensure_dir(os.path.dirname(path))

    fieldnames = [
        "chat_id",
        "chat_title",
        "chat_created_at_utc",
        "chat_created_at_ny",
        "folder_date",
        "transcript_relpath",
        "exported_at_ny",
        "export_status",
        "message_count",
        "tool_call_count",
        "attachment_count",
        "used_full",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "chat_id": r.chat_id,
                "chat_title": r.chat_title,
                "chat_created_at_utc": r.chat_created_at_utc,
                "chat_created_at_ny": r.chat_created_at_ny,
                "folder_date": r.folder_date,
                "transcript_relpath": r.transcript_relpath,
                "exported_at_ny": r.exported_at_ny,
                "export_status": r.export_status,
                "message_count": r.message_count,
                "tool_call_count": r.tool_call_count,
                "attachment_count": r.attachment_count,
                "used_full": r.used_full,
            })


# -----------------------------
# Filename assignment policy
# -----------------------------

def parse_seq_from_transcript_name(name: str, folder_date: str) -> Optional[int]:
    """
    Parse a transcript filename like:
      YYYY_MM_DD_chat_001.txt
    and return the integer sequence 1..N.

    Returns None if pattern does not match.
    """
    # Strict pattern to avoid mis-parsing arbitrary files.
    # Example: 2026_01_21_chat_001.txt
    pat = rf"^{re.escape(folder_date)}_chat_(\d{{3}})\.txt$"
    m = re.match(pat, name, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def next_available_seq_for_date(existing_relpaths: List[str], folder_date: str) -> int:
    """
    Determine the next available chat sequence number for a given folder date.

    We examine existing transcript filenames in index mappings. This ensures we do not
    renumber existing chats and preserves stable mappings across runs.

    Returns an integer >= 1.
    """
    max_seq = 0
    for relpath in existing_relpaths:
        base = os.path.basename(relpath)
        seq = parse_seq_from_transcript_name(base, folder_date)
        if seq is not None and seq > max_seq:
            max_seq = seq
    return max_seq + 1


def assign_transcript_relpath(
    existing_index: Dict[str, dict],
    chat: ChatRow,
    assigned_seq_by_date: Dict[str, int],
) -> Tuple[str, int]:
    """
    Assign a relative transcript path for a chat.

    Rules:
    - If chat_id already exists in index.csv and transcript_relpath is present, reuse it.
    - Otherwise, assign the next available sequence number for that folder date.
      Sequence is per date folder: 001, 002, 003...

    We return (relative_path, seq_int).
    """
    prior = existing_index.get(chat.chat_id)
    if prior:
        rel = (prior.get("transcript_relpath") or "").strip()
        if rel:
            seq = parse_seq_from_transcript_name(os.path.basename(rel), chat.folder_date)
            return rel, (seq if seq is not None else 0)

    if chat.folder_date not in assigned_seq_by_date:
        assigned_seq_by_date[chat.folder_date] = 1

    seq = assigned_seq_by_date[chat.folder_date]
    assigned_seq_by_date[chat.folder_date] += 1

    filename = f"{chat.folder_date}_chat_{seq:03d}.txt"
    relpath = os.path.join(EXPORT_ROOT_FOLDER, chat.folder_date, filename)

    # Normalize to backslashes for Windows-friendly relative paths in index.csv
    relpath = relpath.replace("/", "\\")
    return relpath, seq


# -----------------------------
# Export formatting
# -----------------------------

def format_transcript(
    chat: ChatRow,
    messages: List[dict],
    tool_calls_by_message_id: Dict[int, List[dict]],
    include_thinking: bool,
    export_time_ny: dt.datetime,
    attachment_exports: List[dict],
) -> str:
    """
    Build transcript content for one chat.

    Design goals:
    - Human-readable, audit-friendly, deterministic ordering
    - Explicit separation between message blocks
    - Tool calls and tool results are clearly bounded
    - Thinking is optional (--full) because it can be large
    - Output uses CRLF newlines (via crlf_join)

    Inputs:
    - messages: list of message dicts (chronological)
    - tool_calls_by_message_id: mapping mid -> [tool calls]
    - attachment_exports: list of exported attachment descriptors for this chat

    Returns:
    - Full transcript as a single string ending in CRLF.
    """
    lines: List[str] = []

    # Header
    lines.append("=== CHAT EXPORT ===")
    lines.append(f"Title: {chat.title}")
    lines.append(f"Chat ID: {chat.chat_id}")
    lines.append(f"Chat created_at (UTC): {chat.created_at_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append(f"Chat created_at (NY):  {chat.created_at_ny.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append(f"Exported at (NY):      {export_time_ny.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append(f"Message count: {len(messages)}")

    # Attachments summary
    if attachment_exports:
        lines.append(f"Attachments exported: {len(attachment_exports)}")
        for a in attachment_exports:
            lines.append(f"  - {a.get('relpath','')}")
    else:
        lines.append("Attachments exported: 0")

    lines.append("")  # blank line

    # Separator used between messages
    msg_sep = "-" * 72

    for idx, msg in enumerate(messages):
        if idx > 0:
            lines.append(msg_sep)

        mid = int(msg.get("id") or 0)
        role = (msg.get("role") or "").upper().strip() or "UNKNOWN"
        ts_ny = msg["created_at_ny"].strftime("%Y-%m-%d %H:%M:%S %Z")

        lines.append(f"[{ts_ny}] {role}")

        model_name = (msg.get("model_name") or "").strip()
        if model_name:
            lines.append(f"model: {model_name}")

        # Content
        lines.append("")
        lines.append("CONTENT:")
        content = msg.get("content") or ""
        if content:
            lines.append(content)
        else:
            lines.append("(no content)")

        # Thinking (optional)
        if include_thinking:
            thinking = msg.get("thinking") or ""
            if thinking.strip():
                lines.append("")
                lines.append("THINKING (enabled by --full):")
                lines.append("<<< THINKING BEGIN >>>")
                lines.append(thinking)
                lines.append("<<< THINKING END >>>")

        # Tool result block when present
        tool_result = (msg.get("tool_result") or "").strip()
        if tool_result:
            lines.append("")
            lines.append("TOOL_RESULT:")
            lines.append("<<< TOOL_RESULT BEGIN >>>")
            lines.append(tool_result)
            lines.append("<<< TOOL_RESULT END >>>")

        # Tool calls for this message
        tcs = tool_calls_by_message_id.get(mid) or []
        if tcs:
            lines.append("")
            lines.append(f"TOOL_CALLS: {len(tcs)}")
            for j, tc in enumerate(tcs, start=1):
                lines.append("")
                lines.append(f"[Tool Call {j}]")
                lines.append(f"tool_call_id: {tc.get('id','')}")
                lines.append(f"type: {tc.get('type','')}")
                lines.append(f"function: {tc.get('function_name','')}")

                args = (tc.get("function_arguments") or "").strip()
                if args:
                    lines.append("arguments:")
                    lines.append("<<< ARGUMENTS BEGIN >>>")
                    lines.append(args)
                    lines.append("<<< ARGUMENTS END >>>")

                result = (tc.get("function_result") or "").strip()
                if result:
                    lines.append("result:")
                    lines.append("<<< RESULT BEGIN >>>")
                    lines.append(result)
                    lines.append("<<< RESULT END >>>")

        lines.append("")  # trailing blank line after each message block

    return crlf_join(lines)


# -----------------------------
# Attachment export
# -----------------------------

def export_attachments_for_chat(
    base_dir: str,
    chat: ChatRow,
    seq: int,
    attachments: List[dict],
) -> List[dict]:
    """
    Export attachments for a chat into the same date folder as the transcript.

    Filename policy (your requirement):
      [date]_chat_###_attachment_A (B, C, D...) + extension if present

    Extension handling:
    - Uses safe_attachment_extension() to preserve multi-extensions deterministically:
        "archive.tar.gz" -> ".tar.gz"

    Returns:
    - list of dicts describing exported attachments, in deterministic order:
        {
          "relpath": "chat_export\\YYYY_MM_DD\\YYYY_MM_DD_chat_###_attachment_A.ext",
          "message_id": <int>,
          "original_filename": <str>,
          "bytes": <int>,
        }
    """
    out: List[dict] = []
    if not attachments:
        return out

    date_folder = os.path.join(base_dir, EXPORT_ROOT_FOLDER, chat.folder_date)
    ensure_dir(date_folder)

    for i, att in enumerate(attachments):
        suffix = letter_suffix(i)  # A, B, C...
        orig = (att.get("filename") or "").strip()

        ext = safe_attachment_extension(orig)

        base_name = f"{chat.folder_date}_chat_{seq:03d}_attachment_{suffix}{ext}"

        data = att.get("data")
        if data is None:
            data = b""

        abs_path = os.path.join(date_folder, base_name)
        with open(abs_path, "wb") as f:
            f.write(data)

        relpath = os.path.join(EXPORT_ROOT_FOLDER, chat.folder_date, base_name).replace("/", "\\")
        out.append({
            "relpath": relpath,
            "message_id": int(att.get("message_id") or 0),
            "original_filename": orig,
            "bytes": len(data),
        })

    return out


# -----------------------------
# Main export routine
# -----------------------------

def main() -> int:
    """
    Main entry point.

    High-level flow:
      1) Parse CLI flags
      2) Enforce WAL safety gate (fail fast with guidance)
      3) Ensure chat_export folder exists
      4) Open db.sqlite read-only
      5) Load chats
      6) Load existing index.csv (to preserve stable numbering)
      7) Initialize per-date next sequence numbers from index
      8) For each chat:
           - Determine age in NY local date terms
           - Determine transcript relpath and seq (stable via index)
           - Apply overwrite/skip policy:
               * today/yesterday: always export (overwrite transcript + refresh attachments)
               * 2+ days old: skip only if transcript already exists
                 - If transcript is missing (even if old), export it (and attachments)
           - When exporting: cleanup stale attachments for this chat seq before writing new ones
           - Export transcript (CRLF, UTF-8) and attachments
      9) Rebuild index.csv from this run's outcomes
     10) Append summary lines to export_log.txt and print console summary
    """
    parser = argparse.ArgumentParser(
        description="Export Ollama Desktop chats from db.sqlite to chat_export folders."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include message thinking blocks in transcripts (can be large)."
    )
    args = parser.parse_args()
    include_thinking = bool(args.full)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Prepare export folder and log paths early.
    export_root_abs = os.path.join(base_dir, EXPORT_ROOT_FOLDER)
    ensure_dir(export_root_abs)

    idx_path = index_csv_path(base_dir)
    log_path = export_log_path(base_dir)

    # Log run start
    now_ny = dt.datetime.now(tz=TZ_NY)
    append_log_line(log_path, "----------------------------------------")
    append_log_line(log_path, f"RUN START (NY): {now_ny.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    append_log_line(log_path, f"SCRIPT: {os.path.join(base_dir, 'ollama_export.py')}")
    append_log_line(log_path, f"DB: {os.path.join(base_dir, DB_FILENAME)}")
    append_log_line(log_path, f"INCLUDE_THINKING (--full): {1 if include_thinking else 0}")

    # Safety gate: WAL must be 0 bytes (or absent)
    try:
        fail_fast_if_wal_nonempty(base_dir)
        append_log_line(log_path, "WAL CHECK: OK (empty or missing)")
    except SystemExit as e:
        append_log_line(log_path, "WAL CHECK: FAILED (non-empty). Close Ollama GUI and rerun.")
        append_log_line(log_path, f"RUN END (NY): {dt.datetime.now(tz=TZ_NY).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        return int(e.code)

    # Confirm DB exists
    db_path = os.path.join(base_dir, DB_FILENAME)
    if not os.path.exists(db_path):
        msg = f"ERROR: Database file not found: {db_path}"
        print(msg)
        append_log_line(log_path, msg)
        append_log_line(log_path, f"RUN END (NY): {dt.datetime.now(tz=TZ_NY).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        return 2

    conn = None
    try:
        # Open DB read-only
        try:
            conn = open_db_readonly(db_path)
        except Exception as e:
            msg = f"ERROR: Could not open database: {e}"
            print(msg)
            append_log_line(log_path, msg)
            append_log_line(log_path, f"RUN END (NY): {dt.datetime.now(tz=TZ_NY).strftime('%Y-%m-%d %H:%M:%S %Z')}")
            return 2

        # Load chats
        try:
            chats = load_chats(conn)
        except Exception as e:
            msg = f"ERROR: Failed to load chats: {e}"
            print(msg)
            append_log_line(log_path, msg)
            append_log_line(log_path, f"RUN END (NY): {dt.datetime.now(tz=TZ_NY).strftime('%Y-%m-%d %H:%M:%S %Z')}")
            return 2

        append_log_line(log_path, f"CHATS FOUND: {len(chats)}")

        # Load existing index.csv for stable numbering
        existing_index = load_existing_index(idx_path, log_path=log_path)
        if existing_index:
            append_log_line(log_path, f"INDEX LOAD: Found existing index entries: {len(existing_index)}")
        else:
            append_log_line(log_path, "INDEX LOAD: No existing index (or could not parse). Starting fresh mapping.")

        # Initialize per-date sequence counters from existing index mappings
        assigned_seq_by_date: Dict[str, int] = {}

        relpaths_by_date: Dict[str, List[str]] = {}
        for _, row in existing_index.items():
            folder_date = (row.get("folder_date") or "").strip()
            rel = (row.get("transcript_relpath") or "").strip()
            if folder_date and rel:
                relpaths_by_date.setdefault(folder_date, []).append(rel)

        for folder_date, rels in relpaths_by_date.items():
            assigned_seq_by_date[folder_date] = next_available_seq_for_date(rels, folder_date)

        # Prepare index rows for output (rebuilt each run)
        index_rows: List[IndexRow] = []

        exported_count = 0
        skipped_count = 0
        error_count = 0

        today_ny = dt.datetime.now(tz=TZ_NY).date()

        for chat in chats:
            chat_date_ny = chat.created_at_ny.date()
            age_days = compute_chat_age_days(chat_date_ny, today_ny)

            transcript_relpath, seq = assign_transcript_relpath(
                existing_index=existing_index,
                chat=chat,
                assigned_seq_by_date=assigned_seq_by_date,
            )
            transcript_abspath = os.path.join(base_dir, transcript_relpath)
            ensure_dir(os.path.dirname(transcript_abspath))

            # Policy:
            # - today/yesterday: always export (overwrite)
            # - age >= 2: skip only if transcript exists
            should_overwrite = (age_days in (0, 1))
            should_skip_if_exists = (age_days >= OLD_CHAT_DAYS_THRESHOLD)

            transcript_exists = os.path.exists(transcript_abspath)

            # Skip only if chat is "old" AND transcript already exists
            if should_skip_if_exists and transcript_exists:
                status = "skipped_existing_old"
                skipped_count += 1

                # Counts for index (do not touch files)
                msg_count = 0
                try:
                    messages = load_messages_for_chat(conn, chat.chat_id)
                    msg_count = len(messages)
                except Exception:
                    msg_count = 0

                tool_call_count = 0
                attachment_count = 0
                try:
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT COUNT(*)
                        FROM tool_calls tc
                        JOIN messages m ON m.id = tc.message_id
                        WHERE m.chat_id = ?
                    """, (chat.chat_id,))
                    tool_call_count = int(cur.fetchone()[0] or 0)
                except Exception:
                    tool_call_count = 0

                try:
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT COUNT(*)
                        FROM attachments a
                        JOIN messages m ON m.id = a.message_id
                        WHERE m.chat_id = ?
                    """, (chat.chat_id,))
                    attachment_count = int(cur.fetchone()[0] or 0)
                except Exception:
                    attachment_count = 0

                append_log_line(
                    log_path,
                    f"SKIP (OLD): {chat.folder_date} chat_id={chat.chat_id} age_days={age_days} "
                    f"path={transcript_relpath}"
                )

                index_rows.append(IndexRow(
                    chat_id=chat.chat_id,
                    chat_title=chat.title,
                    chat_created_at_utc=chat.created_at_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    chat_created_at_ny=chat.created_at_ny.strftime("%Y-%m-%d %H:%M:%S"),
                    folder_date=chat.folder_date,
                    transcript_relpath=transcript_relpath.replace("/", "\\"),
                    exported_at_ny=now_ny.strftime("%Y-%m-%d %H:%M:%S"),
                    export_status=status,
                    message_count=msg_count,
                    tool_call_count=tool_call_count,
                    attachment_count=attachment_count,
                    used_full=1 if include_thinking else 0,
                ))
                continue

            # Export path:
            # - today/yesterday: overwrite transcript + refresh attachments
            # - age >= 2 and transcript missing: export (this is the "old chat missing transcript" fix)
            pre_exists = transcript_exists

            try:
                # Load messages
                messages = load_messages_for_chat(conn, chat.chat_id)

                # Bulk-load tool calls for the chat (one query), then attach to transcript by message_id
                tool_calls_by_mid = load_tool_calls_for_chat(conn, chat.chat_id)
                total_tool_calls = sum(len(v) for v in tool_calls_by_mid.values())

                # Determine sequence number for attachments
                # - Normally seq is parsed from transcript filename or allocated for new chats
                # - If seq is 0 due to a prior malformed transcript name, use 1 as fallback
                seq_for_attachments = seq if seq and seq > 0 else 1

                # Cleanup stale attachment files before writing new attachment exports.
                # This keeps the folder consistent for overwrite runs and for "old transcript missing" exports.
                deleted = cleanup_existing_chat_attachments(
                    base_dir=base_dir,
                    folder_date=chat.folder_date,
                    seq=seq_for_attachments,
                )
                if deleted:
                    append_log_line(
                        log_path,
                        f"ATTACHMENT CLEANUP: {chat.folder_date} chat_id={chat.chat_id} "
                        f"seq={seq_for_attachments:03d} deleted={deleted}"
                    )

                # Export attachments for this chat
                attachments = load_attachments_for_chat(conn, chat.chat_id)
                attachment_exports = export_attachments_for_chat(
                    base_dir=base_dir,
                    chat=chat,
                    seq=seq_for_attachments,
                    attachments=attachments,
                )

                # Format transcript
                export_time_ny = dt.datetime.now(tz=TZ_NY)
                transcript_text = format_transcript(
                    chat=chat,
                    messages=messages,
                    tool_calls_by_message_id=tool_calls_by_mid,
                    include_thinking=include_thinking,
                    export_time_ny=export_time_ny,
                    attachment_exports=attachment_exports,
                )

                # Write transcript with UTF-8 encoding and explicit CRLF newlines.
                with open(transcript_abspath, "w", encoding="utf-8", newline="") as f:
                    f.write(transcript_text)

                exported_count += 1

                # Correct status detection:
                # - "exported_overwrite" only when policy says overwrite AND file existed before
                status = "exported_overwrite" if (should_overwrite and pre_exists) else "exported"

                append_log_line(
                    log_path,
                    f"EXPORT: {chat.folder_date} chat_id={chat.chat_id} age_days={age_days} "
                    f"overwrite={1 if should_overwrite else 0} pre_exists={1 if pre_exists else 0} "
                    f"path={transcript_relpath} messages={len(messages)} "
                    f"tool_calls={total_tool_calls} attachments={len(attachment_exports)}"
                )

                index_rows.append(IndexRow(
                    chat_id=chat.chat_id,
                    chat_title=chat.title,
                    chat_created_at_utc=chat.created_at_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    chat_created_at_ny=chat.created_at_ny.strftime("%Y-%m-%d %H:%M:%S"),
                    folder_date=chat.folder_date,
                    transcript_relpath=transcript_relpath.replace("/", "\\"),
                    exported_at_ny=export_time_ny.strftime("%Y-%m-%d %H:%M:%S"),
                    export_status=status,
                    message_count=len(messages),
                    tool_call_count=total_tool_calls,
                    attachment_count=len(attachment_exports),
                    used_full=1 if include_thinking else 0,
                ))

            except Exception as e:
                error_count += 1
                append_log_line(
                    log_path,
                    f"ERROR EXPORT: {chat.folder_date} chat_id={chat.chat_id} age_days={age_days} "
                    f"path={transcript_relpath} error={e}"
                )
                index_rows.append(IndexRow(
                    chat_id=chat.chat_id,
                    chat_title=chat.title,
                    chat_created_at_utc=chat.created_at_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    chat_created_at_ny=chat.created_at_ny.strftime("%Y-%m-%d %H:%M:%S"),
                    folder_date=chat.folder_date,
                    transcript_relpath=transcript_relpath.replace("/", "\\"),
                    exported_at_ny=dt.datetime.now(tz=TZ_NY).strftime("%Y-%m-%d %H:%M:%S"),
                    export_status="error",
                    message_count=0,
                    tool_call_count=0,
                    attachment_count=0,
                    used_full=1 if include_thinking else 0,
                ))
                continue

        # Write index.csv (rebuilt)
        try:
            write_index(idx_path, index_rows)
            append_log_line(
                log_path,
                f"INDEX WRITE: OK path={os.path.join(EXPORT_ROOT_FOLDER, INDEX_CSV_NAME)} rows={len(index_rows)}"
            )
        except Exception as e:
            append_log_line(log_path, f"INDEX WRITE: FAILED error={e}")
            error_count += 1

        end_ny = dt.datetime.now(tz=TZ_NY)
        append_log_line(log_path, f"RUN SUMMARY: exported={exported_count} skipped={skipped_count} errors={error_count}")
        append_log_line(log_path, f"RUN END (NY): {end_ny.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        print(f"Export complete. exported={exported_count} skipped={skipped_count} errors={error_count}")
        print(f"Output folder: {export_root_abs}")
        print(f"Index: {idx_path}")
        print(f"Log: {log_path}")

        return 0 if error_count == 0 else 1

    finally:
        # Always close DB connection to release file handles on Windows.
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
