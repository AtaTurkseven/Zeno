from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.text import Text
from rich import box

console = Console()

BANNER = """\
[bold cyan] ______
|___  / [/bold cyan][bold white]eno[/bold white]
[bold cyan]   / / [/bold cyan][bold white]Workshop[/bold white]
[bold cyan]  / /  [/bold cyan][dim]Project-Aware Assistant[/dim]
[bold cyan] / /__ [/bold cyan][dim]Demo 001[/dim]
[bold cyan]/_____|[/bold cyan]
"""


def print_banner():
    console.print()
    console.print(BANNER)
    console.rule(style="dim cyan")
    console.print()


def print_card(title: str, content: str, style: str = "cyan"):
    """Generic HUD card."""
    console.print(Panel(
        content.strip(),
        title=f"[bold {style}] {title} [/]",
        border_style=style,
        padding=(1, 2),
    ))
    console.print()


def print_project_summary(project: dict):
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), expand=False)
    t.add_column("key", style="dim cyan", min_width=10)
    t.add_column("val", style="white")
    t.add_row("Name", project["name"])
    t.add_row("Type", project["project_type"])
    t.add_row("Files", str(len(project["files"])))
    t.add_row("Path", project["root"])
    console.print(Panel(t, title="[bold green] PROJECT LOADED [/]", border_style="green", padding=(0, 1)))
    console.print()


def print_response(response: str):
    console.print(Panel(
        response.strip(),
        title="[bold blue] ZENO [/]",
        border_style="blue",
        padding=(1, 2),
    ))
    console.print()


def print_error(message: str):
    console.print(Panel(
        f"[red]{message.strip()}[/]",
        title="[bold red] ERROR [/]",
        border_style="red",
        padding=(0, 1),
    ))
    console.print()


def print_status(message: str):
    console.print(f"  [dim cyan]>[/dim cyan] {message}")


def print_help():
    help_text = (
        "[cyan]Commands:[/cyan]\n"
        "  [white]<question>[/white]         Ask anything about the loaded project\n"
        "  [white]:load <path>[/white]       Load a new project folder\n"
        "  [white]:tree[/white]              Show project file tree\n"
        "  [white]:files[/white]             List all loaded files\n"
        "  [white]:inspect <file>[/white]    Show a file excerpt with line numbers\n"
        "  [white]:issues[/white]            Run deterministic issue detection\n"
        "  [white]:logs[/white]              Show extracted error lines from log files\n"
        "  [white]:localsummary[/white]      Build a local summary without using the LLM\n"
        "  [white]:note <text>[/white]       Save a note to memory/SESSION_NOTES.md\n"
        "  [white]:save[/white]              Save the last response to SESSION_NOTES.md\n"
        "  [white]:capture[/white]           Save the last Q&A to memory/INTERACTIONS.md\n"
        "  [white]:closeout a|b|c|d|e|f[/white] Save a structured session closeout\n"
        "  [white]:summarize[/white]         Ask Zeno to summarize the project and save it\n"
        "  [white]:status[/white]            Check Ollama connection and model\n"
        "  [white]:clear[/white]             Clear screen\n"
        "  [white]:help[/white]              Show this help\n"
        "  [white]:quit[/white]  or  Ctrl-C  Exit"
    )
    console.print(Panel(help_text, title="[bold] HELP [/]", border_style="dim", padding=(1, 2)))
    console.print()
