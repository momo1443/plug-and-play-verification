"""Reward function for DeepScaleR with A9-style uniform process reward schedule.

Schedule (controlled by DEEPSCALER_EM_WARMUP_STEPS, default 50):
  - Steps 1-WARMUP:   outcome-only (EM). Same as baseline DeepMath reward.
  - Steps WARMUP+1+:  w ~ U(0,1), reward = w * EM + (1-w) * process_reward

Process reward:
  Splits the solution into reasoning steps and checks whether each step
  contains at least one numerically verifiable equation. A step is
  "verified" if we can extract an equation of the form <expr> = <expr>
  where both sides are purely numeric expressions that evaluate to the
  same number.

    process_reward = (# verified steps) / (# steps with numeric equations)

  Steps without any numeric equation (e.g. prose explanations, variable
  manipulations) are simply not process-graded — they neither help nor
  hurt the process score.

  This is a deterministic, execution-based process signal — no PRM needed.
"""

from __future__ import annotations

import math
import os
import random
import re
from typing import Any

from verl.utils.reward_score import math_reward


_DEEPMATH_DATA_SOURCES = {"deepmath"}

EM_WARMUP_STEPS = int(os.environ.get("DEEPSCALER_EM_WARMUP_STEPS", "50"))
FORMAT_REWARD_WARMUP = 0.1
MIN_STEPS_FOR_PROCESS = 1
_ENV_GLOBAL_STEP = "DEEPSCALER_CURRENT_GLOBAL_STEP"


# ── Step splitting ────────────────────────────────────────────────

def _split_into_steps(text: str) -> list[str]:
    """Split solution text into reasoning steps."""
    # Try explicit step markers first
    step_marker_pattern = re.compile(
        r'(?:^|\n)\s*(?:Step\s+\d+[\.:]|\d+[\.\)]\s)',
        re.IGNORECASE,
    )
    markers = list(step_marker_pattern.finditer(text))
    if len(markers) >= 2:
        steps = []
        for i, m in enumerate(markers):
            start = m.start()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            step_text = text[start:end].strip()
            if step_text:
                steps.append(step_text)
        if steps:
            return steps

    # Fall back to blank-line paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    steps = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]
    if len(steps) >= 2:
        return steps

    # Last resort: split on lines
    lines = text.split('\n')
    steps = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
    if len(steps) >= 2:
        return steps

    return [text] if text.strip() else []


# ── Equation extraction ───────────────────────────────────────────

def _clean_lhs(lhs: str) -> str:
    """Remove leading non-math text from LHS of equation."""
    # Remove common prefixes like "First,", "Then,", "So,", "Therefore,"
    lhs = re.sub(r'^(?:First|Then|Next|Finally|So|Therefore|Thus|Hence)[,\s]+', '', lhs, flags=re.IGNORECASE)
    # Remove trailing dots from RHS
    return lhs.strip()


def _clean_rhs(rhs: str) -> str:
    """Remove trailing noise from RHS."""
    # Remove trailing period/comma
    rhs = rhs.rstrip('.,;')
    return rhs.strip()


def _extract_numeric_equations(step_text: str) -> list[tuple[str, str]]:
    """Extract (lhs, rhs) pairs that look like purely numeric equations.

    Only extracts equations where both sides appear to be numeric
    expressions (no unresolved variables). This ensures we can verify
    them by evaluation.
    """
    equations: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _try_add(lhs: str, rhs: str) -> None:
        lhs = _clean_lhs(lhs)
        rhs = _clean_rhs(rhs)
        if not lhs or not rhs:
            return
        # Quick check: both sides should have at least one digit
        if not re.search(r'\d', lhs) and not re.search(r'\d', rhs):
            return
        # Both sides should be purely numeric expressions
        # (no single-letter variables that would remain unresolved)
        if _is_numeric_expr(lhs) and _is_numeric_expr(rhs):
            key = f"{lhs}={rhs}"
            if key not in seen:
                seen.add(key)
                equations.append((lhs, rhs))

    # 1. Extract from $$...$$ blocks
    for m in re.finditer(r'\$\$(.+?)\$\$', step_text, re.DOTALL):
        _extract_eqs_from_text(m.group(1), _try_add)

    # 2. Extract from $...$ blocks
    for m in re.finditer(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', step_text):
        _extract_eqs_from_text(m.group(1), _try_add)

    # 3. Plain-text equations (skip lines already covered by LaTeX blocks)
    for line in step_text.split('\n'):
        line = line.strip()
        if not line or '$$' in line or line.startswith('$'):
            continue
        _extract_eqs_from_text(line, _try_add)

    return equations


def _extract_eqs_from_text(text: str, add_fn) -> None:
    """Extract 'LHS = RHS' from a piece of text, calling add_fn for each."""
    # Split on lines first
    for segment in re.split(r'\\\\|\n', text):
        segment = segment.strip()
        if not segment:
            continue
        # Find single '=' (not ==, !=, <=, >=)
        # Use lookahead/lookbehind to avoid matching comparison operators
        matches = re.finditer(r'([^=<>!]+?)\s*(?<!<)(?<!>)(?<!!)=\s*([^=]+)', segment)
        for m in matches:
            lhs = m.group(1).strip()
            rhs = m.group(2).strip()
            add_fn(lhs, rhs)


def _is_numeric_expr(expr: str) -> bool:
    """Check if an expression appears to be purely numeric (no variables).

    After LaTeX cleanup, a numeric expression should contain only:
    digits, operators (+-*/), parentheses, dots, spaces, and known functions.
    """
    # Remove LaTeX math commands
    cleaned = expr
    for func in ['\\sqrt', '\\frac', '\\sin', '\\cos', '\\tan', '\\log',
                 '\\ln', '\\exp', '\\abs', '\\pi', '\\cdot', '\\times',
                 '\\div', '\\left', '\\right', '\\lfloor', '\\rfloor',
                 '\\lceil', '\\rceil', '\\infty']:
        cleaned = cleaned.replace(func, '')
    # Remove remaining LaTeX commands
    cleaned = re.sub(r'\\[a-zA-Z]+', '', cleaned)
    # Remove braces/brackets
    cleaned = cleaned.replace('{', '').replace('}', '')
    cleaned = cleaned.replace('[', '').replace(']', '')
    cleaned = cleaned.replace('(', '').replace(')', '')
    # Remove digits, operators, spaces, dots, commas
    cleaned = re.sub(r'[\d+\-*/^.,\s]', '', cleaned)
    # If anything remains, it's likely a variable name
    # Allow common math symbols
    cleaned = cleaned.replace('=', '').replace('!', '').replace('>', '').replace('<', '')
    # After all removals, if there are alphabetic characters left,
    # they're likely variable names -> not purely numeric
    return not re.search(r'[a-zA-Z]', cleaned)


# ── Safe expression evaluation ────────────────────────────────────

def _latex_to_python(expr: str) -> str | None:
    """Convert a LaTeX math expression to a Python-evaluable string."""
    if not expr or not expr.strip():
        return None

    s = expr.strip()

    # LaTeX -> Python replacements (order matters)
    s = s.replace('\\left', '').replace('\\right', '')
    s = s.replace('\\{', '(').replace('\\}', ')')
    s = s.replace('\\(', '(').replace('\\)', ')')
    s = s.replace('\\times', '*')
    s = s.replace('\\div', '/')
    s = s.replace('\\cdot', '*')
    s = s.replace('\\pi', 'pi')
    s = s.replace('\\infty', 'inf')
    s = s.replace('\\sqrt', 'sqrt')
    s = s.replace('\\ln', 'log')
    s = s.replace('\\log', 'log')
    s = s.replace('\\exp', 'exp')
    s = s.replace('\\sin', 'sin')
    s = s.replace('\\cos', 'cos')
    s = s.replace('\\tan', 'tan')
    s = s.replace('\\abs', 'abs')
    # \frac{a}{b} -> (a)/(b)
    s = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', s)
    # Remove remaining LaTeX commands
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    # Braces -> parens
    s = s.replace('{', '(').replace('}', ')')
    s = s.replace('[', '(').replace(']', ')')
    # ^ -> **
    s = s.replace('^', '**')
    # Remove formatting spaces
    s = s.replace('\\,', '').replace('\\;', '').replace('\\!', '')
    s = s.replace('\\ ', ' ').replace('~', ' ')
    # Remove \text{...} etc
    s = re.sub(r'\\text\{[^}]*\}', '', s)
    s = re.sub(r'\\mathrm\{[^}]*\}', '', s)
    # Implicit multiplication: 2pi -> 2*pi, 3( -> 3*(, )2 -> )*2, )( -> )*(
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)
    s = re.sub(r'(\d)\(', r'\1*(', s)
    s = re.sub(r'\)(\d)', r')*\1', s)
    s = re.sub(r'\)\(', r')*(', s)
    s = re.sub(r'\)([a-zA-Z])', r')*\1', s)

    return s.strip() or None


def _safe_eval(expr_str: str) -> float | None:
    """Safely evaluate a Python math expression to a float."""
    if not expr_str:
        return None

    safe_ns = {
        '__builtins__': {},
        'sqrt': math.sqrt,
        'pi': math.pi,
        'e': math.e,
        'abs': abs,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'ln': math.log,
        'exp': math.exp,
        'inf': float('inf'),
        'floor': math.floor,
        'ceil': math.ceil,
        'factorial': math.factorial,
    }

    try:
        result = eval(expr_str, safe_ns, {})  # noqa: S307
        if isinstance(result, (int, float)) and math.isfinite(result):
            return float(result)
        return None
    except Exception:
        return None


def _verify_equation_pair(lhs_str: str, rhs_str: str) -> bool:
    """Check if LHS = RHS is numerically true after LaTeX conversion."""
    lhs_py = _latex_to_python(lhs_str)
    rhs_py = _latex_to_python(rhs_str)

    if lhs_py is None or rhs_py is None:
        return False

    lhs_val = _safe_eval(lhs_py)
    rhs_val = _safe_eval(rhs_py)

    if lhs_val is None or rhs_val is None:
        return False

    if lhs_val == rhs_val:
        return True

    try:
        denom = max(abs(lhs_val), abs(rhs_val), 1e-10)
        return abs(lhs_val - rhs_val) / denom < 1e-6
    except (ZeroDivisionError, OverflowError):
        return False


# ── Process reward computation ────────────────────────────────────

def compute_process_reward(solution_str: str) -> tuple[float, int, int]:
    """Compute process reward by verifying intermediate equations.

    Returns:
        (process_reward, num_verified_steps, num_graded_steps)
    """
    steps = _split_into_steps(solution_str)
    num_graded_steps = 0
    num_verified_steps = 0

    for step in steps:
        eqs = _extract_numeric_equations(step)
        if not eqs:
            continue  # no numeric equations in this step — not graded
        num_graded_steps += 1
        # Step is verified if at least one equation numerically checks out
        if any(_verify_equation_pair(lhs, rhs) for lhs, rhs in eqs):
            num_verified_steps += 1

    if num_graded_steps < MIN_STEPS_FOR_PROCESS:
        return 0.0, 0, num_graded_steps

    return num_verified_steps / num_graded_steps, num_verified_steps, num_graded_steps


# ── Terminal EM ───────────────────────────────────────────────────

def _has_boxed_answer(text: str) -> bool:
    return bool(re.search(r'\\boxed\s*\{', text))


def _fallback_extract_last_answer(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r'(?:[Tt]he )?[Aa]nswer\s*(?:is|:)\s*(.+?)(?:\.|$)', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'[Ss]o\s+(?:the )?[Aa]nswer\s*(?:is|:)\s*(.+?)(?:\.|$)', text)
    if m:
        return m.group(1).strip()
    return None


def compute_terminal_em(solution_str: str, ground_truth: str) -> float:
    score = math_reward.compute_score(solution_str, ground_truth)
    if score > 0:
        return 1.0
    fallback = _fallback_extract_last_answer(solution_str)
    if fallback is not None:
        try:
            if math_reward.is_equiv(fallback, ground_truth):
                return 1.0
        except Exception:
            pass
        if fallback.strip() == ground_truth.strip():
            return 1.0
    return 0.0


# ── Main reward function ─────────────────────────────────────────

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float | dict:
    """
    DeepScaleR A9 uniform reward schedule.

    Schedule:
      Steps 1-WARMUP:  pure EM (+ small format bonus like baseline)
      Steps WARMUP+1+: w ~ U(0,1), reward = w * EM + (1-w) * process_reward
    """
    if data_source not in _DEEPMATH_DATA_SOURCES:
        from verl.utils.reward_score import default_compute_score
        return default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)

    if not ground_truth or not solution_str:
        return 0.0

    gt_str = str(ground_truth)
    has_boxed = _has_boxed_answer(solution_str)
    terminal_em = compute_terminal_em(solution_str, gt_str)

    # ── Determine reward phase ────────────────────────────────────
    extra_info = extra_info or {}
    is_validation = bool(extra_info.get("_agent_r1_is_validation", False))
    global_step = int(extra_info.get("_agent_r1_global_step", -1))
    if global_step < 0:
        try:
            global_step = int(os.environ.get(_ENV_GLOBAL_STEP, "0"))
        except (ValueError, TypeError):
            global_step = 0

    if is_validation:
        reward_phase = "validation_terminal_em"
        terminal_weight = 1.0
        process_weight = 0.0
    elif global_step <= EM_WARMUP_STEPS:
        reward_phase = "em_warmup"
        terminal_weight = 1.0
        process_weight = 0.0
    else:
        reward_phase = "uniform"
        w = random.random()
        terminal_weight = w
        process_weight = 1.0 - w

    # ── Compute final reward ──────────────────────────────────────
    if reward_phase == "em_warmup":
        if terminal_em > 0:
            reward = 1.0
        elif has_boxed:
            reward = FORMAT_REWARD_WARMUP
        else:
            reward = 0.0
        process_reward_val = 0.0
        num_verified = 0
        num_graded = 0
        process_component = 0.0
        process_component_valid = False
    elif reward_phase == "validation_terminal_em":
        reward = float(terminal_em)
        process_reward_val = 0.0
        num_verified = 0
        num_graded = 0
        process_component = 0.0
        process_component_valid = False
    else:
        process_reward_val, num_verified, num_graded = compute_process_reward(solution_str)
        process_component = process_weight * process_reward_val
        reward = terminal_weight * terminal_em + process_component
        process_component_valid = num_graded >= MIN_STEPS_FOR_PROCESS

    return {
        "score": float(reward),
        "terminal_em": float(terminal_em),
        "process_reward": float(process_reward_val),
        "num_verified_steps": num_verified,
        "num_graded_steps": num_graded,
        "optimizer_reward_phase": reward_phase,
        "optimizer_terminal_weight": float(terminal_weight),
        "optimizer_process_weight": float(process_weight),
        "a9_process_component_valid": process_component_valid,
        "optimizer_process_component": float(process_component),
        "a9_em_warmup_steps": EM_WARMUP_STEPS,
        "training_global_step": global_step,
    }
