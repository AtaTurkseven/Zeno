import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from .analyzer import answer_without_llm, detect_issues, summarize_project
from .config import load_config
from .llm import check_ollama, query_ollama
from .memory import append_session_note, save_interaction
from .scanner import build_context_text, scan_project


def build_hud_state(project_path: str, question: str = "", config: dict | None = None) -> dict:
    config = config or load_config()
    project = scan_project(project_path, config)
    context_text = build_context_text(project)
    issues = detect_issues(project)
    ok, status_msg = check_ollama(config)

    last_answer = summarize_project(project)
    if question.strip():
        fallback = answer_without_llm(question, project)
        if ok:
            response = query_ollama(question, context_text, config)
            if response.startswith("[") and fallback:
                last_answer = response + "\n\nFallback analysis:\n" + fallback
            else:
                last_answer = response
        elif fallback:
            last_answer = fallback
        else:
            last_answer = status_msg

    issue_lines = [f"- {issue['title']}: {issue['summary']}" for issue in issues[:5]]
    return {
        "project": project,
        "project_path": str(Path(project_path).resolve()),
        "ollama_status": status_msg,
        "context_text": context_text,
        "last_answer": last_answer,
        "issues_text": "\n".join(issue_lines) if issue_lines else "No obvious issues detected.",
    }


class ZenoHUD:
    def __init__(self, project_path: str | None = None):
        self.config = load_config()
        self.project_path = project_path or str(Path("test_project").resolve())
        self.state = None
        self.last_query = ""

        self.root = tk.Tk()
        self.root.title("Zeno Workshop HUD")
        self.root.geometry("1180x760")
        self.root.configure(bg="#101418")

        self._build_layout()
        self.refresh_project()

    def _build_layout(self):
        top = tk.Frame(self.root, bg="#101418")
        top.pack(fill="x", padx=14, pady=12)

        self.path_var = tk.StringVar(value=self.project_path)
        path_entry = tk.Entry(top, textvariable=self.path_var, bg="#182028", fg="#f0f4f8", insertbackground="#f0f4f8")
        path_entry.pack(side="left", fill="x", expand=True)

        tk.Button(top, text="Browse", command=self.browse_project, bg="#2f8f83", fg="white").pack(side="left", padx=8)
        tk.Button(top, text="Reload", command=self.refresh_project, bg="#3b4d61", fg="white").pack(side="left")

        middle = tk.Frame(self.root, bg="#101418")
        middle.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        middle.columnconfigure(0, weight=2)
        middle.columnconfigure(1, weight=3)
        middle.rowconfigure(0, weight=1)
        middle.rowconfigure(1, weight=1)

        self.project_card = self._make_card(middle, "Project Status", 0, 0)
        self.issues_card = self._make_card(middle, "Detected Issues", 1, 0)
        self.answer_card = self._make_card(middle, "Last Answer", 0, 1, rowspan=2)

        bottom = tk.Frame(self.root, bg="#101418")
        bottom.pack(fill="x", padx=14, pady=(0, 14))

        self.query_var = tk.StringVar()
        query_entry = tk.Entry(bottom, textvariable=self.query_var, bg="#182028", fg="#f0f4f8", insertbackground="#f0f4f8")
        query_entry.pack(side="left", fill="x", expand=True)
        query_entry.bind("<Return>", lambda _event: self.ask_question())

        tk.Button(bottom, text="Ask", command=self.ask_question, bg="#2f8f83", fg="white").pack(side="left", padx=8)
        tk.Button(bottom, text="Save Note", command=self.save_note, bg="#7a5c2e", fg="white").pack(side="left", padx=4)
        tk.Button(bottom, text="Capture Q&A", command=self.capture_interaction, bg="#5c3f75", fg="white").pack(side="left")

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var, anchor="w", bg="#101418", fg="#98a6b3").pack(fill="x", padx=16, pady=(0, 10))

    def _make_card(self, parent, title: str, row: int, column: int, rowspan: int = 1):
        frame = tk.LabelFrame(
            parent,
            text=title,
            bg="#182028",
            fg="#e7edf2",
            bd=1,
            labelanchor="n",
            padx=8,
            pady=8,
        )
        frame.grid(row=row, column=column, rowspan=rowspan, sticky="nsew", padx=8, pady=8)
        text = ScrolledText(frame, wrap="word", bg="#182028", fg="#f0f4f8", insertbackground="#f0f4f8", relief="flat")
        text.pack(fill="both", expand=True)
        text.configure(state="disabled")
        return text

    def _set_text(self, widget: ScrolledText, content: str):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content.strip())
        widget.configure(state="disabled")

    def browse_project(self):
        selected = filedialog.askdirectory(initialdir=self.path_var.get() or str(Path.cwd()))
        if selected:
            self.path_var.set(selected)
            self.refresh_project()

    def refresh_project(self):
        try:
            self.status_var.set("Loading project...")
            self.state = build_hud_state(self.path_var.get(), config=self.config)
            self.project_path = self.state["project_path"]
            self.path_var.set(self.project_path)
            self._render_state()
            self.status_var.set("Project loaded")
        except Exception as exc:
            messagebox.showerror("Zeno HUD", f"Failed to load project:\n{exc}")
            self.status_var.set("Project load failed")

    def ask_question(self):
        question = self.query_var.get().strip()
        if not question:
            return
        self.last_query = question
        self.status_var.set("Thinking...")
        threading.Thread(target=self._ask_question_worker, args=(question,), daemon=True).start()

    def _ask_question_worker(self, question: str):
        try:
            self.state = build_hud_state(self.path_var.get(), question=question, config=self.config)
            self.root.after(0, self._render_state)
            self.root.after(0, lambda: self.status_var.set("Answer ready"))
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Zeno HUD", f"Question failed:\n{exc}"))
            self.root.after(0, lambda: self.status_var.set("Question failed"))

    def save_note(self):
        if not self.state:
            messagebox.showwarning("Zeno HUD", "Load a project first.")
            return
        path = append_session_note(self.state["last_answer"], self.config)
        self.status_var.set(f"Saved note to {path.name}")

    def capture_interaction(self):
        if not self.state or not self.last_query:
            messagebox.showwarning("Zeno HUD", "Ask a question first.")
            return
        path = save_interaction(self.state["project"]["name"], self.last_query, self.state["last_answer"], self.config)
        self.status_var.set(f"Captured interaction to {path.name}")

    def _render_state(self):
        if not self.state:
            return
        project = self.state["project"]
        status_text = (
            f"Name: {project['name']}\n"
            f"Type: {project['project_type']}\n"
            f"Files: {len(project['files'])}\n"
            f"Path: {self.state['project_path']}\n\n"
            f"Ollama: {self.state['ollama_status']}"
        )
        self._set_text(self.project_card, status_text)
        self._set_text(self.issues_card, self.state["issues_text"])
        self._set_text(self.answer_card, self.state["last_answer"])

    def run(self):
        self.root.mainloop()


def run_hud(project_path: str | None = None):
    app = ZenoHUD(project_path=project_path)
    app.run()