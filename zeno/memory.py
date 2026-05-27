from pathlib import Path
from datetime import datetime


def get_memory_dir(config: dict) -> Path:
    p = Path(config["memory"]["path"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_file(filename: str, config: dict) -> str:
    """Read a memory file. Returns empty string if it does not exist."""
    path = get_memory_dir(config) / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def append_session_note(note: str, config: dict) -> Path:
    """Append a timestamped note to SESSION_NOTES.md. Returns the file path."""
    mem_dir = get_memory_dir(config)
    notes_file = mem_dir / config["memory"]["session_notes"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    is_new = not notes_file.exists()

    with open(notes_file, "a", encoding="utf-8") as f:
        if is_new:
            f.write("# Zeno Session Notes\n\n")
        f.write(f"### {timestamp}\n{note.strip()}\n\n")

    return notes_file


def save_project_summary(project_name: str, summary: str, config: dict) -> Path:
    """Append a project summary to PROJECT_SUMMARIES.md."""
    mem_dir = get_memory_dir(config)
    summaries_file = mem_dir / "PROJECT_SUMMARIES.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    is_new = not summaries_file.exists()

    with open(summaries_file, "a", encoding="utf-8") as f:
        if is_new:
            f.write("# Zeno Project Summaries\n\n")
        f.write(f"## {project_name} — {timestamp}\n\n{summary.strip()}\n\n---\n\n")

    return summaries_file
