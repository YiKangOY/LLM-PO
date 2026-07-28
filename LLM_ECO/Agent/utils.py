def get_optimization_type_from_tcl_command(tcl_command: str) -> str:
    """Extract optimization type from executed TCL command"""
    if not tcl_command:
        return 'unknown'

    tcl_command_lower = tcl_command.lower()
    if 'opt_timing' in tcl_command_lower:
        return 'timing'
    elif 'opt_power' in tcl_command_lower:
        return 'power'
    elif 'opt_area' in tcl_command_lower:
        return 'area'
    else:
        return 'unknown'
from enum import Enum
class ECOType(Enum):
    """ECO fix types"""
    TIMING = "timing"
    POWER = "power"
    AREA = "area"
def extract_content_from_llm_response(response_content) -> str:
    """
    Extract text content from LLM response.

    Args:
        response_content: Can be either:
            - str: Direct text content
            - list: List of content items (reasoning mode) where text is at [0]['text']

    Returns:
        str: Extracted text content
    """
    if isinstance(response_content, str):
        return response_content
    # List format (reasoning mode) - extract text from first item
    return response_content[0]['text']


def escape_unquoted_quotes_in_json(json_text: str) -> str:
    """Escape double quotes that appear inside JSON strings without escaping."""
    in_string = False
    escaped = False
    result = []
    text_len = len(json_text)
    for idx, ch in enumerate(json_text):
        if not in_string:
            if ch == '"':
                in_string = True
            result.append(ch)
            continue

        if escaped:
            result.append(ch)
            escaped = False
            continue

        if ch == '\\':
            escaped = True
            result.append(ch)
            continue

        if ch == '"':
            next_idx = idx + 1
            while next_idx < text_len and json_text[next_idx].isspace():
                next_idx += 1
            if next_idx < text_len and json_text[next_idx] in [',', '}', ']', ':']:
                in_string = False
                result.append(ch)
            else:
                result.append('\\')
                result.append('"')
            continue

        result.append(ch)

    return "".join(result)


def remove_trailing_json_commas(json_text: str) -> str:
    """Remove trailing commas before JSON object or array endings."""
    in_string = False
    escaped = False
    result = []
    text_len = len(json_text)
    for idx, ch in enumerate(json_text):
        if not in_string:
            if ch == '"':
                in_string = True
            if ch == ',':
                next_idx = idx + 1
                while next_idx < text_len and json_text[next_idx].isspace():
                    next_idx += 1
                if next_idx < text_len and json_text[next_idx] in ['}', ']']:
                    continue
            result.append(ch)
            continue

        if escaped:
            result.append(ch)
            escaped = False
            continue

        if ch == '\\':
            escaped = True
            result.append(ch)
            continue

        if ch == '"':
            in_string = False

        result.append(ch)

    return "".join(result)


def extract_json_from_thinking_response(content: str) -> str:
    """Extract JSON content from LLM response that may contain <think>...</think> tags"""
    import re

    # Remove thinking tags and their content
    cleaned_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    cleaned_content = cleaned_content.replace("\u201c", "\"").replace("\u201d", "\"")
    cleaned_content = cleaned_content.replace("\u2018", "'").replace("\u2019", "'")

    # Try to extract JSON from the cleaned content
    # Look for content within ```json...``` blocks first
    json_match = re.search(r'```json\s*(.*?)\s*```', cleaned_content, re.DOTALL)
    if json_match:
        json_text = json_match.group(1).strip()
        return remove_trailing_json_commas(escape_unquoted_quotes_in_json(json_text))

    # If no code block, try to find JSON-like content
    # Look for content that starts with { and ends with }
    json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
    if json_match:
        json_text = json_match.group(0).strip()
        return remove_trailing_json_commas(escape_unquoted_quotes_in_json(json_text))

    # Return cleaned content if no specific JSON pattern found
    json_text = cleaned_content.strip()
    return remove_trailing_json_commas(escape_unquoted_quotes_in_json(json_text))
