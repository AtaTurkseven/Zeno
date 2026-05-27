import requests
from typing import Tuple

SYSTEM_PROMPT = """\
You are Zeno Workshop Assistant — a technical AI assistant for a maker and embedded systems engineer.

You have been given the full contents of a local project folder. Your job:
- Answer technical questions about the project directly and specifically
- Diagnose bugs, compilation errors, hardware faults, and logic issues
- Reference actual file names, function names, and line content when relevant
- Suggest concrete next steps — not generic advice
- Flag dangerous code (race conditions, memory leaks, blocking calls in ISRs, etc.)

Rules:
- Be direct and concise. No motivational fluff.
- If you don't know, say so. Do not hallucinate function names or library behavior.
- Prefer structured output: numbered steps, bullet points, code blocks.
- If a fix requires hardware changes, say so explicitly.
- Treat the user as a competent engineer who wants real answers.

The full project context is included below.
"""


def query_ollama(prompt: str, context: str, config: dict) -> str:
    """Send a prompt + project context to Ollama and return the response text."""
    llm_cfg = config["llm"]
    base_url = llm_cfg["base_url"].rstrip("/")
    model = llm_cfg["model"]
    timeout = int(llm_cfg.get("timeout", 90))

    full_prompt = f"{SYSTEM_PROMPT}\n\n{context}\n\n--- USER QUERY ---\n{prompt}"

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 1024,
                    "top_p": 0.9,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        response_text = data.get("response", "").strip()
        if not response_text:
            return "[EMPTY RESPONSE] Model returned no text."
        return response_text

    except requests.exceptions.ConnectionError:
        return (
            f"[CONNECTION ERROR] Cannot reach Ollama at {base_url}.\n"
            "Fix:\n"
            "  1. Run: ollama serve\n"
            f"  2. Pull model: ollama pull {model}\n"
            "  3. Check config.yaml → llm.base_url"
        )
    except requests.exceptions.Timeout:
        return (
            f"[TIMEOUT] Ollama did not respond within {timeout}s.\n"
            "Fix: increase llm.timeout in config.yaml, or use a smaller model."
        )
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            try:
                ollama_msg = e.response.json().get("error", "model not found")
            except Exception:
                ollama_msg = "model not found"
            return (
                f"[MODEL NOT FOUND] {ollama_msg}\n"
                f"Fix: ollama pull {model}\n"
                f"Or change 'llm.model' in config.yaml to a model you have."
            )
        return f"[HTTP ERROR] Ollama returned: {e}"
    except Exception as e:
        return f"[ERROR] LLM query failed: {type(e).__name__}: {e}"


def check_ollama(config: dict) -> Tuple[bool, str]:
    """Check if Ollama is running and the configured model is available."""
    llm_cfg = config["llm"]
    base_url = llm_cfg["base_url"].rstrip("/")
    model = llm_cfg["model"]

    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        tags_data = r.json()
        available_models = [
            m["name"].split(":")[0]
            for m in tags_data.get("models", [])
        ]
        full_names = [m["name"] for m in tags_data.get("models", [])]

        model_found = (
            model in available_models
            or any(model in name for name in full_names)
        )

        if model_found:
            return True, f"Ollama OK | model: {model} | url: {base_url}"
        else:
            available_str = ", ".join(available_models) if available_models else "none"
            return False, (
                f"Ollama running at {base_url}, but model '{model}' not found.\n"
                f"Available: {available_str}\n"
                f"Fix: ollama pull {model}"
            )

    except requests.exceptions.ConnectionError:
        return False, (
            f"Ollama not running at {base_url}.\n"
            "Fix: ollama serve"
        )
    except Exception as e:
        return False, f"Ollama check error: {type(e).__name__}: {e}"
