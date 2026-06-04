"""DeepSeek chat integration — supports both streaming and non-streaming responses.

When EVIR evaluation context is provided, the chat AI becomes aware
of the latest evaluation results and can discuss them in context.
)import json
import os
from typing import Any, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_API_KEY = os.getenv("DEEPSEEK_API_KEY")


def chat_stream(
    messages: list[dict],
    eval_context: Optional[dict[str, Any]] = None,
):
    """Stream chat responses one chunk at a time for SSE."""

    system_msg = _build_system_prompt(eval_context)
    chat_msgs = []
    if system_msg:
        chat_msgs.append({"role": "system", "content": system_msg})
    chat_msgs.extend(messages)

    payload = {
        "model": DEFAULT_MODEL,
        "messages": chat_msgs,
        "temperature": 0.7,
        "stream": True,
    }

    req = Request(
        url=f"{DEFAULT_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {DEFAULT_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    current_chunk = ""
    try:
        with urlopen(req, timeout=120) as response:
            for line in response:
                decoded = line.decode("utf-8")
                if not decoded.startswith("data: "):
                    continue
                decoded = decoded[6]  # strip "data: "
                if decoded.strip() == "[DONE]":
                    break
                try:
                    chunk_data = json.loads(decoded)
                    delta = chunk_data.get("choices", [])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        current_chunk += content
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
    except HTTPError as e:
        yield f"[Error: {e}]"
        return
    except Exception as e:
        yield f"[Error: {str(e)}]"


def chat_sync(
    messages: list[dict],
    eval_context: Optional[dict[str, Any]] = None,
) -> str:
    """Send a synchronous chat request and return the full response."""

    system_msg = _build_system_prompt(eval_context)
    chat_msgs = []
    if system_msg:
        chat_msgs.append({"role": "system", "content": system_msg})
    chat_msgs.extend(messages)

    payload = {
        "model": DEFAULT_MODEL,
        "messages": chat_msgs,
        "temperature": 0.7,
        "stream": False,
    }

    req = Request(
        url=f"{DEFAULT_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {DEFAULT_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=120) as response:
            data = json.loads(response.read())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Error: {str(e)}]"


def _build_system_prompt(eval_context: Optional[dict] = None) -> str:
    """Build the system prompt with optional evaluation context."""

    base_msg = "You are an expert financial advisor and investment analyst."
    if not eval_context:
        return base_msg

    ctx = eval_context
    return (
        f"{base_msg} \n\n"
        f"YOU HAVE JUST COMPLETED an investment evaluation\n"
        f"for $({ctx.get('ticker', '')=}). Here's what you concluded:\n"
        f"- Recommendation: ${ctx.get('recommendation', '')}\n"
        f"- Confidence: ${ctx.get('confidence', '')}\n"
        f"- Risk Level: ${ctx.get('risk_level', '')}\n"
        f"- Rationale: ${ctx.get('decision_rationale', '')}\n"
        "\n"
        "You can discuss this evaluation with the user. If they ask"
        "for details about the analysis, you should be able to explain the"
        "reasoning behind the decision."
    )
