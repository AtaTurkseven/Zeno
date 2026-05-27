import os
from pathlib import Path
from typing import List, Dict


def scan_project(project_path: str, config: dict) -> Dict:
    """
    Walk a project directory and return structured project data.
    Returns: root, name, tree, files[], project_type
    """
    root = Path(project_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project path not found: {project_path}")
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {project_path}")

    cfg = config["project"]
    include_exts = set(cfg["include_extensions"])
    exclude_dirs = set(cfg["exclude_dirs"])
    max_size = cfg["max_file_size_kb"] * 1024

    files: List[Dict] = []
    tree_lines: List[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place (affects os.walk recursion)
        dirnames[:] = sorted(d for d in dirnames if d not in exclude_dirs)

        rel_dir = Path(dirpath).relative_to(root)
        depth = len(rel_dir.parts)
        indent = "  " * depth

        if depth == 0:
            tree_lines.append(f"{root.name}/")
        else:
            tree_lines.append(f"{indent}{rel_dir.parts[-1]}/")

        for filename in sorted(filenames):
            filepath = Path(dirpath) / filename
            rel_path = filepath.relative_to(root)
            ext = filepath.suffix.lower()
            size = filepath.stat().st_size

            tree_lines.append(f"{indent}  {filename}")

            if ext in include_exts and size <= max_size:
                try:
                    content = filepath.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    content = f"[READ ERROR: {e}]"
                files.append({
                    "path": str(rel_path),
                    "name": filename,
                    "ext": ext,
                    "size": size,
                    "content": content,
                })

    return {
        "root": str(root),
        "name": root.name,
        "tree": "\n".join(tree_lines),
        "files": files,
        "project_type": _detect_project_type(files),
    }


def _detect_project_type(files: List[Dict]) -> str:
    exts = {f["ext"] for f in files}
    names = {f["name"].lower() for f in files}

    if ".ino" in exts:
        return "Arduino/ESP32"
    if "platformio.ini" in names:
        return "PlatformIO (ESP32/Arduino)"
    if "cargo.toml" in names:
        return "Rust/Embedded"
    if ".py" in exts and "requirements.txt" in names:
        return "Python Project"
    if ".py" in exts:
        return "Python Script(s)"
    if ".cpp" in exts or ".c" in exts:
        return "C/C++"
    if "package.json" in names:
        return "JavaScript/Node.js"
    if "cmakelists.txt" in names:
        return "CMake Project"
    return "Unknown"


def build_context_text(project: Dict, max_chars: int = 14000) -> str:
    """
    Build a text block summarizing project contents for LLM context.
    Prioritizes README > logs/errors > primary code files.
    """
    parts = [
        f"PROJECT NAME: {project['name']}",
        f"PROJECT TYPE: {project['project_type']}",
        f"PATH: {project['root']}",
        "",
        "=== FILE TREE ===",
        project["tree"],
        "",
        "=== FILE CONTENTS ===",
    ]
    header_text = "\n".join(parts)
    used_chars = len(header_text)

    # Priority order for file inclusion
    def priority_key(f: Dict) -> int:
        name = f["name"].lower()
        path = f["path"].lower()
        if "readme" in name:
            return 0
        if f["ext"] == ".log" or "error" in name or "debug" in name:
            return 1
        if f["ext"] == ".ino":
            return 2
        if f["ext"] == ".py":
            return 3
        if f["ext"] in (".cpp", ".c", ".h"):
            return 4
        if f["ext"] == ".md":
            return 5
        if f["ext"] in (".yaml", ".yml", ".toml", ".ini", ".cfg"):
            return 6
        return 7

    sorted_files = sorted(project["files"], key=priority_key)
    content_blocks = []

    for f in sorted_files:
        header = f"\n--- {f['path']} ---\n"
        body = f["content"]
        block = header + body

        if used_chars + len(block) > max_chars:
            remaining = max_chars - used_chars - len(header) - 20
            if remaining > 300:
                block = header + body[:remaining] + "\n[...TRUNCATED...]"
            else:
                break

        content_blocks.append(block)
        used_chars += len(block)

    return header_text + "".join(content_blocks)
