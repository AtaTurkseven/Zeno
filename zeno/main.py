"""
Zeno Workshop — Demo 001: Project-Aware Assistant
Entry point: python run.py [project_path]
"""
import sys
from pathlib import Path
from rich.prompt import Prompt
from rich.console import Console

from .config import load_config
from .scanner import scan_project, build_context_text
from .llm import query_ollama, check_ollama
from .memory import append_session_note, save_project_summary
from . import cards

console = Console()


def _load_project(path: str, config: dict):
    """Load and scan a project folder. Returns (project_dict, context_text) or raises."""
    project = scan_project(path, config)
    context = build_context_text(project)
    return project, context


def main():
    config = load_config()

    cards.print_banner()

    # Startup Ollama check
    ok, status_msg = check_ollama(config)
    if ok:
        cards.print_status(f"[green]{status_msg}[/green]")
    else:
        cards.print_status(f"[yellow]LLM unavailable: {status_msg}[/yellow]")
    console.print()

    # Resolve project path from CLI arg or prompt
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
    else:
        project_path = Prompt.ask(
            "[cyan]Project path[/cyan]",
            default=str(Path("test_project").resolve()),
        )

    # Initial project load
    project = None
    context_text = ""
    try:
        cards.print_status(f"Scanning: {project_path}")
        project, context_text = _load_project(project_path, config)
        cards.print_project_summary(project)
    except FileNotFoundError as e:
        cards.print_error(str(e))
        sys.exit(1)

    cards.print_help()

    # ── Main REPL ──────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]ZENO >[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if user_input.startswith(":"):
            parts = user_input.split(" ", 1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in (":quit", ":q", ":exit"):
                console.print("[dim]Session ended.[/dim]")
                break

            elif cmd == ":load":
                if not arg:
                    cards.print_error("Usage: :load <project_path>")
                    continue
                try:
                    cards.print_status(f"Loading: {arg}")
                    project, context_text = _load_project(arg, config)
                    cards.print_project_summary(project)
                except FileNotFoundError as e:
                    cards.print_error(str(e))

            elif cmd == ":tree":
                if project:
                    cards.print_card("FILE TREE", project["tree"], style="cyan")
                else:
                    cards.print_error("No project loaded. Use :load <path>")

            elif cmd == ":files":
                if project:
                    lines = [
                        f"  {f['path']:50s} {f['size'] // 1024:4d} KB"
                        for f in project["files"]
                    ]
                    cards.print_card("LOADED FILES", "\n".join(lines), style="cyan")
                else:
                    cards.print_error("No project loaded.")

            elif cmd == ":note":
                if not arg:
                    cards.print_error("Usage: :note <text>")
                    continue
                path = append_session_note(arg, config)
                cards.print_status(f"[green]Note saved → {path}[/green]")

            elif cmd == ":summarize":
                if not project:
                    cards.print_error("No project loaded.")
                    continue
                cards.print_status("[dim]Summarizing project...[/dim]")
                summary_prompt = (
                    "Summarize this project in 5-10 bullet points. Include: "
                    "what it does, key files, main dependencies, detected issues, "
                    "and suggested next steps. Be specific to this project's actual contents."
                )
                response = query_ollama(summary_prompt, context_text, config)
                cards.print_response(response)
                save_project_summary(project["name"], response, config)
                cards.print_status(f"[green]Summary saved → memory/PROJECT_SUMMARIES.md[/green]")

            elif cmd == ":status":
                ok, msg = check_ollama(config)
                style = "green" if ok else "red"
                cards.print_card("OLLAMA STATUS", msg, style=style)

            elif cmd == ":clear":
                console.clear()
                cards.print_banner()

            elif cmd == ":help":
                cards.print_help()

            else:
                cards.print_error(f"Unknown command: {cmd}  —  type :help")

        # ── LLM Query ─────────────────────────────────────────────────────────
        else:
            if not project:
                cards.print_error("No project loaded. Use :load <path>")
                continue

            cards.print_status("[dim]Thinking...[/dim]")
            response = query_ollama(user_input, context_text, config)
            cards.print_response(response)
