import re
from typing import Dict, List, Optional


def summarize_project(project: Dict) -> str:
    code_files = [f for f in project["files"] if f["ext"] in {".py", ".ino", ".c", ".cpp", ".h", ".rs"}]
    log_files = [f for f in project["files"] if f["ext"] == ".log"]
    note_files = [f for f in project["files"] if f["ext"] == ".md"]
    issues = detect_issues(project)

    lines = [
        f"Project: {project['name']}",
        f"Type: {project['project_type']}",
        f"Loaded files: {len(project['files'])}",
        f"Code files: {len(code_files)} | Logs: {len(log_files)} | Markdown: {len(note_files)}",
        "",
        "Key files:",
    ]
    for file_info in project["files"][:6]:
        lines.append(f"- {file_info['path']}")

    if issues:
        lines.append("")
        lines.append("Detected issues:")
        for issue in issues[:5]:
            lines.append(f"- {issue['title']}: {issue['summary']}")

    next_steps = suggest_next_steps(project, issues)
    if next_steps:
        lines.append("")
        lines.append("Next steps:")
        for step in next_steps:
            lines.append(f"- {step}")

    return "\n".join(lines)


def detect_issues(project: Dict) -> List[Dict]:
    issues: List[Dict] = []

    for file_info in project["files"]:
        content = file_info["content"]
        path = file_info["path"]
        lower = content.lower()

        if "iram_attr" in lower and "vtaskdelay(" in lower:
            issues.append({
                "title": "ISR contains vTaskDelay",
                "summary": f"{path} calls vTaskDelay inside an ISR, which can trip the watchdog.",
                "path": path,
            })

        if "vtaskdelay()" in lower and "isr" in lower:
            issues.append({
                "title": "ISR contains vTaskDelay",
                "summary": f"{path} documents vTaskDelay inside an ISR, which matches an interrupt watchdog reset pattern.",
                "path": path,
            })

        if "gurumeditationerror" in lower.replace(" ", "") or "guru meditation error" in lower:
            if "interrupt wdt timeout" in lower:
                issues.append({
                    "title": "Interrupt watchdog reset",
                    "summary": f"{path} shows an interrupt watchdog timeout, usually caused by ISR misuse or blocking work.",
                    "path": path,
                })

        if "ondisconnect" in lower and "startadvertising" not in lower and "bleservercallbacks" in lower:
            issues.append({
                "title": "BLE advertising not restarted",
                "summary": f"{path} handles BLE disconnect without restarting advertising, so reconnects may fail.",
                "path": path,
            })

        if "dmp init failed" in lower or "dmp initialization" in lower:
            issues.append({
                "title": "MPU6050 DMP init failure",
                "summary": f"{path} indicates DMP initialization is failing; project is likely running in raw sensor fallback mode.",
                "path": path,
            })

    return _sort_issues(_dedupe_issues(issues))


def summarize_logs(project: Dict) -> str:
    log_files = [f for f in project["files"] if f["ext"] == ".log"]
    if not log_files:
        return "No .log files loaded in this project."

    lines = []
    for file_info in log_files:
        lines.append(f"Log: {file_info['path']}")
        extracted = extract_error_lines(file_info["content"])
        if extracted:
            for entry in extracted[:10]:
                lines.append(f"- {entry}")
        else:
            lines.append("- No obvious error lines detected.")
        lines.append("")
    return "\n".join(lines).strip()


def inspect_file(project: Dict, query: str, max_lines: int = 120) -> str:
    file_info = find_file(project, query)
    if not file_info:
        return f"File not found: {query}"

    lines = file_info["content"].splitlines()
    excerpt = lines[:max_lines]
    numbered = [f"{idx + 1:>4}: {line}" for idx, line in enumerate(excerpt)]
    if len(lines) > max_lines:
        numbered.append(".... [TRUNCATED]")

    return f"File: {file_info['path']}\n\n" + "\n".join(numbered)


def answer_without_llm(question: str, project: Dict) -> Optional[str]:
    lower = question.lower()
    issues = detect_issues(project)

    if any(term in lower for term in ["summarize", "summary", "what does this project do"]):
        return summarize_project(project)

    if any(term in lower for term in ["log", "error", "stack trace", "guru meditation"]):
        return summarize_logs(project)

    if any(term in lower for term in ["issue", "issues", "dangerous", "risk"]):
        return format_issues(issues)

    if "ble" in lower and "disconnect" in lower:
        for issue in issues:
            if issue["title"] in {"Interrupt watchdog reset", "BLE advertising not restarted", "ISR contains vTaskDelay"}:
                return format_issue_answer(project, issues)

    if "crash" in lower or "watchdog" in lower or "firmware crashing" in lower:
        return format_issue_answer(project, issues)

    return None


def format_issues(issues: List[Dict]) -> str:
    if not issues:
        return "No obvious issues detected from static scan."

    lines = []
    for issue in issues:
        lines.append(f"- {issue['title']} [{issue['path']}]")
        lines.append(f"  {issue['summary']}")
    return "\n".join(lines)


def format_issue_answer(project: Dict, issues: List[Dict]) -> str:
    if not issues:
        return "No obvious crash cause detected from static scan or log files."

    primary = issues[0]
    lines = [
        f"Likely root cause: {primary['title']}",
        primary["summary"],
        "",
        "Other relevant findings:",
    ]
    for issue in issues[1:4]:
        lines.append(f"- {issue['title']}: {issue['summary']}")
    return "\n".join(lines)


def suggest_next_steps(project: Dict, issues: List[Dict]) -> List[str]:
    steps: List[str] = []
    titles = {issue["title"] for issue in issues}

    if "ISR contains vTaskDelay" in titles:
        steps.append("Remove vTaskDelay or any blocking RTOS call from ISR context.")
    if "BLE advertising not restarted" in titles:
        steps.append("Restart BLE advertising in the disconnect callback so clients can reconnect.")
    if "Interrupt watchdog reset" in titles:
        steps.append("Reproduce the reset while capturing serial logs after removing ISR blocking work.")
    if "MPU6050 DMP init failure" in titles:
        steps.append("Keep raw accel/gyro fallback for now and isolate DMP init as a separate hardware/I2C debug task.")

    if not steps:
        steps.append("Load a real project folder and verify answers against known bugs before expanding scope.")

    return steps


def find_file(project: Dict, query: str) -> Optional[Dict]:
    normalized = query.strip().lower().replace("\\", "/")
    for file_info in project["files"]:
        path = file_info["path"].lower().replace("\\", "/")
        name = file_info["name"].lower()
        if normalized == path or normalized == name:
            return file_info
    for file_info in project["files"]:
        path = file_info["path"].lower().replace("\\", "/")
        name = file_info["name"].lower()
        if normalized in path or normalized in name:
            return file_info
    return None


def extract_error_lines(content: str) -> List[str]:
    patterns = [
        r"^\[ERROR\].*$",
        r"^Guru Meditation Error:.*$",
        r"^Backtrace:.*$",
        r"^rst:.*$",
        r"^.*watchdog.*$",
    ]
    matches: List[str] = []
    for line in content.splitlines():
        for pattern in patterns:
            if re.search(pattern, line, flags=re.IGNORECASE):
                matches.append(line.strip())
                break
    return matches


def _dedupe_issues(issues: List[Dict]) -> List[Dict]:
    seen = set()
    unique: List[Dict] = []
    for issue in issues:
        key = issue["title"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)
    return unique


def _sort_issues(issues: List[Dict]) -> List[Dict]:
    priorities = {
        "ISR contains vTaskDelay": 0,
        "Interrupt watchdog reset": 1,
        "BLE advertising not restarted": 2,
        "MPU6050 DMP init failure": 3,
    }
    return sorted(issues, key=lambda issue: (priorities.get(issue["title"], 99), issue["path"]))