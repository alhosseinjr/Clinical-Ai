"""
Shared LLM output parser.
[... keep your existing docstring ...]
"""

import ast
import json
import logging
import re
from typing import Any

# Type aliases for clearer annotations
JSONDict = dict[str, Any]
JSONArray = list[Any]
ParsedJSON = JSONDict | JSONArray

# Module-level constants
_MAX_LOG_CHARS = 500
_MAX_NESTING = 256

_SMART_QUOTES = {
    "\u201c": '"',  # "
    "\u201d": '"',  # "
    "\u2018": "'",  # '
    "\u2019": "'",  # '
}

_DELIMITER_PAIRS = {
    "}": "{",
    "]": "[",
}

# Safer regex: only strip fences that appear on their own lines
_MARKDOWN_RE = re.compile(
    r"^\s*```[^\n]*\n|\n?```\s*$",
    re.MULTILINE,
)

_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")

logger = logging.getLogger(__name__)

# Translation table for smart quote normalization
_QUOTE_TRANSLATION_TABLE = str.maketrans(_SMART_QUOTES)


def _normalize_smart_quotes(text: str) -> str:
    """
    Normalize smart/curly quotes to straight quotes.
    
    WARNING: This intentionally mutates string contents to recover malformed JSON
    emitted by some LLMs. This is a lossy operation.
    """
    return text.translate(_QUOTE_TRANSLATION_TABLE)


def _normalize_json_keys(text: str) -> str:
    """Fix common key naming errors from small models."""
    # Fix parentheses in keys
    text = re.sub(r'\("([^"]+)"\)', r'"\1"', text)
    
    # Fix specific key variations
    replacements = {
        '"notable Flags"': '"notable_flags"',
        '"notable flags"': '"notable_flags"',
        '"c citations"': '"citations"',
        '"recommendation"': '"recommendations"',
        '"mentioned_condition"': '"mentioned_conditions"',
        '"mentioned_medication"': '"mentioned_medications"',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences that appear on their own lines."""
    return _MARKDOWN_RE.sub("", text).strip()


def _try_parse(text: str, allow_arrays: bool = False) -> ParsedJSON | None:
    """Attempt to parse JSON text directly. Returns None on failure."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        if allow_arrays and isinstance(data, list):
            return data
        return None
    except json.JSONDecodeError:
        return None


def _failure(default: JSONDict, raw_output: str) -> JSONDict:
    """Build failure response with raw output preserved."""
    return {
        **default,
        "raw_output": raw_output,
    }


def _extract_json_block(text: str, allow_arrays: bool = False) -> str | None:
    """
    Extract the first complete JSON object (or array) from text using stack-based
    delimiter tracking.
    """
    # Find start of object or array
    obj_start = text.find("{")
    arr_start = text.find("[") if allow_arrays else -1
    
    # Determine which comes first
    if obj_start < 0 and arr_start < 0:
        return None
    
    if obj_start < 0:
        start = arr_start
    elif arr_start < 0:
        start = obj_start
    else:
        start = min(obj_start, arr_start)
    
    # Stack-based tracking for nested delimiters
    stack = []
    in_string = False
    escape_next = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        # Escape handling only matters inside strings
        if in_string and char == "\\":
            escape_next = True
            continue
        
        if char == '"':
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char in "{[":
            stack.append(char)
            # Protect against excessive nesting
            if len(stack) > _MAX_NESTING:
                logger.warning("JSON nesting depth exceeded maximum (%d)", _MAX_NESTING)
                return None
        elif char in "}]":
            if not stack:
                # Unmatched closing delimiter
                return None
            
            opening = stack.pop()
            
            # Validate that delimiters match correctly
            if opening != _DELIMITER_PAIRS[char]:
                return None
            
            # If we just closed the outermost delimiter, we're done
            if not stack:
                return text[start:i+1]
    
    return None


def _try_extract_and_parse(text: str, allow_arrays: bool = False) -> ParsedJSON | None:
    """
    Attempt to extract JSON block and parse it, with trailing comma repair.
    """
    # Attempt extraction
    extracted = _extract_json_block(text, allow_arrays)
    if not extracted:
        return None
    
    # Normalize keys before parsing
    extracted = _normalize_json_keys(extracted)
    
    # Attempt direct parse
    if data := _try_parse(extracted, allow_arrays):
        logger.debug("Recovered JSON via extraction")
        return data
    
    # Attempt repair of trailing commas
    repaired = _TRAILING_COMMA_RE.sub(r"\1", extracted)
    if data := _try_parse(repaired, allow_arrays):
        logger.debug("Recovered JSON via trailing comma repair")
        return data
    
    return None


def parse_llm_json(
    raw_output: str, 
    default: JSONDict | None = None,
    allow_arrays: bool = False
) -> ParsedJSON:
    """
    Robust JSON parser for small LLM outputs.
    [Keep your existing docstring]
    """
    if default is None:
        default = {"parse_error": "could not parse model output as JSON"}

    if not isinstance(raw_output, str) or not raw_output.strip():
        logger.debug("Empty or non-string input to parse_llm_json")
        return _failure(default, raw_output)

    text = raw_output.strip()
    
    # Strip UTF-8 BOM
    text = text.lstrip("\ufeff")
    
    # Attempt 1: Direct parse
    if data := _try_parse(text, allow_arrays):
        return data
    
    # Attempt 2: Extract JSON block and parse
    if data := _try_extract_and_parse(text, allow_arrays):
        return data
    
    # Attempt 3: Normalize smart quotes
    text = _normalize_smart_quotes(text)
    if data := _try_parse(text, allow_arrays):
        logger.debug("Recovered JSON after smart quote normalization")
        return data
    
    # Attempt 4: Strip markdown fences
    text = _strip_markdown_fences(text)
    if data := _try_parse(text, allow_arrays):
        logger.debug("Recovered JSON after markdown cleanup")
        return data
    
    # Attempt 5: Extract JSON block again after cleanup
    if data := _try_extract_and_parse(text, allow_arrays):
        return data
    
        # Attempt 6: Handle Python-style dictionaries (single quotes)
    try:
        py_text = text.replace('false', 'False').replace('true', 'True').replace('null', 'None')
        data = ast.literal_eval(py_text)
        if isinstance(data, dict):
            logger.debug("Recovered JSON via ast.literal_eval")
            return data
    except (ValueError, SyntaxError):
        pass
    
    # All attempts failed
    logger.warning(
        "Failed parsing JSON (chars=%d arrays=%s): %.500r",
        len(raw_output),
        allow_arrays,
        raw_output,
    )
    return _failure(default, raw_output)