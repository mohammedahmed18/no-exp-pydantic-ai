from __future__ import annotations as _annotations

from pydantic_ai.messages import TextPart, ThinkingPart

START_THINK_TAG = "<think>"
END_THINK_TAG = "</think>"


def split_content_into_text_and_thinking(content: str) -> list[ThinkingPart | TextPart]:
    """Split a string into text and thinking parts.

    Some models don't return the thinking part as a separate part, but rather as a tag in the content.
    This function splits the content into text and thinking parts.

    We use the `<think>` tag because that's how Groq uses it in the `raw` format, so instead of using `<Thinking>` or
    something else, we just match the tag to make it easier for other models that don't support the `ThinkingPart`.
    """
    parts: list[ThinkingPart | TextPart] = []
    pos = 0
    len_start = len(START_THINK_TAG)
    len_end = len(END_THINK_TAG)
    content_len = len(content)

    while pos < content_len:
        start_index = content.find(START_THINK_TAG, pos)
        if start_index == -1:
            # The rest is plain text
            if pos < content_len:
                parts.append(TextPart(content=content[pos:]))
            break
        # Add text before <think> as TextPart, if any
        if start_index > pos:
            parts.append(TextPart(content=content[pos:start_index]))
        # Move after <think> tag
        start_think_content = start_index + len_start
        end_index = content.find(END_THINK_TAG, start_think_content)
        if end_index == -1:
            # No closing tag, treat the rest as plain text
            parts.append(TextPart(content=content[start_think_content:]))
            break
        # Add the thinking part
        parts.append(ThinkingPart(content=content[start_think_content:end_index]))
        pos = end_index + len_end

    return parts
