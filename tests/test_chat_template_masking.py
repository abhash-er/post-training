"""Tests for the chat-template `{% generation %}` marker detector.

These guard against the silent-no-op failure mode where TRL's
`assistant_only_loss=True` looks like it's masking but actually isn't,
because the chat template lacks `{% generation %}…{% endgeneration %}`
markers around the assistant content emission.

We test the detector itself (cheap, no deps) plus per-template
regression guards on the registered chat templates.

Marker *presence* is necessary but not sufficient: markers spliced around
the wrong span mask the wrong tokens while still passing the detector.
The span-level tests at the bottom render templates through transformers'
own tracker and assert on which text actually lands in the loss.
"""

from __future__ import annotations

import pytest

from post_training.chat_templates.registry import (
    get_chat_template,
    has_generation_markers,
)

# ── detector unit tests ────────────────────────────────────────────────


def test_detects_plain_markers() -> None:
    assert has_generation_markers("before {% generation %} body {% endgeneration %} after")


def test_detects_whitespace_stripped_markers() -> None:
    assert has_generation_markers("before {%- generation -%} body {%- endgeneration -%} after")


def test_detects_asymmetric_strip_markers() -> None:
    # Mixed strip directions also valid Jinja2.
    assert has_generation_markers("{%- generation %} body {% endgeneration -%}")


def test_rejects_missing_close() -> None:
    assert not has_generation_markers("{% generation %} body, never closes")


def test_rejects_missing_open() -> None:
    assert not has_generation_markers("body without an opener {% endgeneration %}")


def test_rejects_empty_or_none() -> None:
    assert not has_generation_markers("")
    assert not has_generation_markers(None)


# ── registry-level regression guards ───────────────────────────────────


def test_olmo3_instruct_sft_template_has_markers() -> None:
    """The Instruct-SFT template is the one our production SFT config uses.
    If anyone strips `{% generation %}` markers from
    `olmo3-instruct-sft.jinja`, SFT silently regresses to full-sequence loss.
    """
    assert has_generation_markers(get_chat_template("olmo3-instruct-sft"))


def test_olmo3_think_sft_template_has_markers() -> None:
    """The Think-SFT template is the one to use when reproducing
    AllenAI's Olmo-3-7B-Think-SFT recipe via TRL.  Same masking story as
    Instruct-SFT — if `{% generation %}` markers go missing, SFT
    regresses to full-sequence loss.
    """
    assert has_generation_markers(get_chat_template("olmo3-think-sft"))


def test_olmo3_template_lacks_markers() -> None:
    """The legacy `olmo3` template (a re-formatted copy of
    Olmo-3-7B-Think-SFT) does not have `{% generation %}` markers.
    Documented here so the runtime guard's behaviour is explicit:
    starting an SFT run with `chat_template: olmo3` will raise — use
    `olmo3-instruct-sft` or `olmo3-think-sft` instead, depending on
    which checkpoint you are reproducing.
    """
    assert not has_generation_markers(get_chat_template("olmo3"))


def test_qwen3_template_has_markers() -> None:
    """The qwen3 template drives `assistant_only_loss=True` for Qwen3 SFT
    runs.  Same story as the OLMo templates — no markers, no masking.
    """
    assert has_generation_markers(get_chat_template("qwen3"))


# ── span-level masking tests ───────────────────────────────────────────
#
# These render through transformers' generation tracker — the same code
# path `return_assistant_tokens_mask=True` (and therefore TRL's
# `assistant_only_loss=True`) uses — and assert on the exact substrings
# that end up in the loss.  Character spans, not token spans, so no
# tokenizer download is needed.


def _assistant_spans(template_name: str, messages: list[dict]) -> list[str]:
    """Return the substrings of the rendered conversation that are in the loss."""
    # transformers-internal: if this import moves, the public equivalent is
    # `tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)`,
    # which additionally requires a tokenizer.
    chat_template_utils = pytest.importorskip("transformers.utils.chat_template_utils")

    compiled = chat_template_utils._compile_jinja_template(get_chat_template(template_name))
    rendered, indices = chat_template_utils._render_with_assistant_indices(
        compiled, messages, None, None, False
    )
    return [rendered[start:end] for start, end in indices]


def _think(reasoning: str, answer: str) -> str:
    return f"<think>\n{reasoning}\n</think>\n\n{answer}"


def test_qwen3_excludes_history_assistant_turns() -> None:
    """Qwen3 strips `<think>` from assistant turns at or before the last
    user query, so those turns render as a bare answer with no reasoning
    trace.  Training on them teaches the model to answer directly at
    exactly the position where it always opens a `<think>` block at
    inference — a train/inference mismatch that trains against the
    reasoning behaviour.  Only the final exchange belongs in the loss.
    """
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": _think("R1", "A1")},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": _think("R2", "A2")},
        {"role": "user", "content": "Q3"},
        {"role": "assistant", "content": _think("R3", "A3")},
    ]

    assert _assistant_spans("qwen3", messages) == [_think("R3", "A3") + "<|im_end|>"]


def test_qwen3_keeps_every_assistant_turn_after_last_user_query() -> None:
    """`ns.last_query_index` tracks the last *real* user message, not the
    last message.  A multi-step tool exchange therefore has several
    assistant turns after it, and all of them keep their reasoning trace
    when rendered — so all of them belong in the loss.
    """
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "<think>\nCall the tool.\n</think>\n\n",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Berlin"}},
                }
            ],
        },
        {"role": "tool", "content": '{"temp_c": 18}'},
        {"role": "assistant", "content": "18C in Berlin."},
    ]

    spans = _assistant_spans("qwen3", messages)
    assert len(spans) == 2
    assert spans[0].startswith("<think>\nCall the tool.\n</think>")
    assert "<tool_call>" in spans[0]
    assert spans[1].endswith("18C in Berlin.<|im_end|>")


def test_qwen3_excludes_prompt_header_and_turn_separator() -> None:
    """The `<|im_start|>assistant\\n` header is prompt, and the `\\n` after
    `<|im_end|>` is a turn separator — neither is something the model
    generates, so both stay outside the span.
    """
    messages = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": _think("R", "A")},
    ]

    (span,) = _assistant_spans("qwen3", messages)
    assert not span.startswith("<|im_start|>")
    assert span.endswith("<|im_end|>")


_EMPTY_THINK = "<think>\n\n</think>\n\n"

# A Dolci-Instruct-SFT tool-use row, in Dolci's own schema: tool signatures in
# the system message's `functions`, Python-style calls in the assistant's
# `function_calls`, tool results under role `environment`, multi-turn.
_DOLCI_TOOL_ROW = [
    {"role": "system", "content": "You can call functions.", "functions": '[{"name": "f"}]'},
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": None, "function_calls": "f(x=1)\ng(y=2)"},
    {"role": "environment", "content": '{"result": 1}\n{"result": 2}'},
    {"role": "assistant", "content": "A1"},
    {"role": "user", "content": "Q2"},
    {"role": "assistant", "content": None, "function_calls": "f(x=3)"},
    {"role": "environment", "content": '{"result": 3}'},
    {"role": "assistant", "content": "A2"},
]


def test_qwen3_dolci_tool_row_trains_every_assistant_turn() -> None:
    """Dolci tool rows follow AllenAI's OLMo-3 Instruct-SFT recipe: every
    assistant turn is in the loss, call-only turns included, each rendered
    as it was generated (empty think block, then the answer or the calls).
    """
    spans = _assistant_spans("qwen3", _DOLCI_TOOL_ROW)

    assert spans == [
        _EMPTY_THINK + "<function_calls>f(x=1)\ng(y=2)</function_calls><|im_end|>",
        _EMPTY_THINK + "A1<|im_end|>",
        _EMPTY_THINK + "<function_calls>f(x=3)</function_calls><|im_end|>",
        _EMPTY_THINK + "A2<|im_end|>",
    ]


def test_qwen3_dolci_tool_row_renders_functions_and_environment_as_prompt() -> None:
    """The signatures and the tool results are rendered (as in
    olmo3-instruct-sft) but stay outside the loss."""
    chat_template_utils = pytest.importorskip("transformers.utils.chat_template_utils")
    compiled = chat_template_utils._compile_jinja_template(get_chat_template("qwen3"))
    rendered, _ = chat_template_utils._render_with_assistant_indices(
        compiled, _DOLCI_TOOL_ROW, None, None, False
    )
    spans = "".join(_assistant_spans("qwen3", _DOLCI_TOOL_ROW))

    assert rendered.startswith(
        '<|im_start|>system\nYou can call functions. <functions>[{"name": "f"}]</functions><|im_end|>\n'
    )
    assert '<|im_start|>environment\n{"result": 1}\n{"result": 2}<|im_end|>\n' in rendered
    assert "<functions>" not in spans
    assert '{"result"' not in spans
    assert "Q1" not in spans and "Q2" not in spans


def test_qwen3_null_tool_fields_keep_final_turn_only() -> None:
    """A mixed parquet gives every row the Dolci fields, as None when unused.
    Such a row is not a tool row: only the final exchange stays in the loss."""
    messages = [
        {"role": "user", "content": "Q1", "function_calls": None, "functions": None},
        {"role": "assistant", "content": "A1", "function_calls": None, "functions": None},
        {"role": "user", "content": "Q2", "function_calls": None, "functions": None},
        {"role": "assistant", "content": "A2", "function_calls": None, "functions": None},
    ]

    assert _assistant_spans("qwen3", messages) == [_EMPTY_THINK + "A2<|im_end|>"]


def test_olmo3_instruct_sft_masks_all_assistant_turns() -> None:
    """Counterpoint to the qwen3 behaviour above, pinned deliberately: the
    OLMo templates do not strip anything from history, so every assistant
    turn renders exactly as generated and every one belongs in the loss.
    The qwen3 exclusion is a Qwen-specific consequence of think-stripping,
    not a house rule about multi-turn data.
    """
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]

    spans = _assistant_spans("olmo3-instruct-sft", messages)
    assert len(spans) == 2
    assert spans[0].startswith("A1")
    assert spans[1].startswith("A2")
