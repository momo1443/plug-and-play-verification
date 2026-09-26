"""Reward function for DeepScaleR with A9-style uniform process reward schedule.

Schedule (controlled by DEEPSCALER_EM_WARMUP_STEPS, default 50):
  - Steps 1-WARMUP:   strict outcome-only (EM).
  - Steps WARMUP+1+:  one w ~ U(0,1) per prompt group,
                       reward = w * EM + (1-w) * process_reward

Process reward:
  Splits the solution into reasoning steps and verifies every extracted
  numeric equation in every step. Each step receives the fraction of its
  equations that verify; a step without a verifiable equation receives zero.

    step_reward = (# verified equations) / (# extracted equations)
    process_reward = mean(step_reward over all reasoning steps)

  Trivial literal identities such as ``1 = 1`` do not receive credit. This
  prevents one unrelated tautology from making an otherwise invalid step pass.

  This is a deterministic, execution-based process signal — no PRM needed.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any

from recipes.reward_mixing import prompt_group_key_from_extra_info
from agent_r1.evaluation.answers import final_answer_record

_DEEPMATH_DATA_SOURCES = {"deepmath"}

EM_WARMUP_STEPS = int(os.environ.get("DEEPSCALER_EM_WARMUP_STEPS", "50"))
_ENV_GLOBAL_STEP = "DEEPSCALER_CURRENT_GLOBAL_STEP"
_GROUP_WEIGHT_NAMESPACE = b"deepscaler-a9-group-weight-v1"


# ── Step splitting ────────────────────────────────────────────────


def _split_into_steps(text: str) -> list[str]:
    """Split solution text into reasoning steps."""
    # Try explicit step markers first
    step_marker_pattern = re.compile(
        r"(?:^|\n)\s*(?:Step\s+\d+[\.:]|\d+[\.\)]\s)",
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
    paragraphs = re.split(r"\n\s*\n", text)
    steps = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]
    if len(steps) >= 2:
        return steps

    # Last resort: split on lines
    lines = text.split("\n")
    steps = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
    if len(steps) >= 2:
        return steps

    return [text] if text.strip() else []


# ── Equation extraction ───────────────────────────────────────────


def _clean_lhs(lhs: str) -> str:
    """Remove leading non-math text from LHS of equation."""
    # Keep the expression after prose/list separators when several equations
    # share one line (for example, "... 2+3=5 and 4+4=8").
    lhs = re.split(r"\b(?:and|then|next)\b|[,;:]", lhs, flags=re.IGNORECASE)[-1]
    # Plain-text model outputs often prefix an equation with a step marker.
    lhs = re.sub(r"^\s*(?:Step\s+\d+\s*[:.]|\d+[.)])\s*", "", lhs, flags=re.IGNORECASE)
    # Remove common prefixes like "First,", "Then,", "So,", "Therefore,"
    lhs = re.sub(r"^(?:First|Then|Next|Finally|So|Therefore|Thus|Hence)[,\s]+", "", lhs, flags=re.IGNORECASE)
    # Drop a remaining prose prefix before the first numeric/LaTeX token.
    token = re.search(r"(?:\\[A-Za-z]+|[0-9])", lhs)
    if token is not None:
        lhs = lhs[token.start() :]
    # Remove trailing dots from RHS
    return lhs.strip()


def _clean_rhs(rhs: str) -> str:
    """Remove trailing noise from RHS."""
    rhs = re.split(r"\b(?:and|then|next)\b|[,;]", rhs, maxsplit=1, flags=re.IGNORECASE)[0]
    # Remove trailing period/comma
    rhs = rhs.rstrip(".,;")
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
        if not re.search(r"\d", lhs) and not re.search(r"\d", rhs):
            return
        # Both sides should be purely numeric expressions
        # (no single-letter variables that would remain unresolved)
        if _is_numeric_expr(lhs) and _is_numeric_expr(rhs):
            key = f"{lhs}={rhs}"
            if key not in seen:
                seen.add(key)
                equations.append((lhs, rhs))

    # 1. Extract from $$...$$ blocks
    for m in re.finditer(r"\$\$(.+?)\$\$", step_text, re.DOTALL):
        _extract_eqs_from_text(m.group(1), _try_add)

    # 2. Extract from $...$ blocks
    for m in re.finditer(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", step_text):
        _extract_eqs_from_text(m.group(1), _try_add)

    # 3. Plain-text equations after removing LaTeX blocks already handled above.
    plain_text = re.sub(r"\$\$.*?\$\$|(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", " ", step_text, flags=re.DOTALL)
    for line in plain_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        _extract_eqs_from_text(line, _try_add)

    return equations


def _extract_eqs_from_text(text: str, add_fn) -> None:
    """Extract 'LHS = RHS' from a piece of text, calling add_fn for each."""
    # Split on lines first
    for segment in re.split(r"\\\\|\n|\b(?:and|then|next)\b", text, flags=re.IGNORECASE):
        segment = segment.strip()
        if not segment:
            continue
        # Find single '=' (not ==, !=, <=, >=)
        # Use lookahead/lookbehind to avoid matching comparison operators
        matches = re.finditer(r"([^=<>!]+?)\s*(?<!<)(?<!>)(?<!!)=\s*([^=]+)", segment)
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
    for func in [
        "\\sqrt",
        "\\frac",
        "\\sin",
        "\\cos",
        "\\tan",
        "\\log",
        "\\ln",
        "\\exp",
        "\\abs",
        "\\pi",
        "\\cdot",
        "\\times",
        "\\div",
        "\\left",
        "\\right",
        "\\lfloor",
        "\\rfloor",
        "\\lceil",
        "\\rceil",
        "\\infty",
    ]:
        cleaned = cleaned.replace(func, "")
    # Remove remaining LaTeX commands
    cleaned = re.sub(r"\\[a-zA-Z]+", "", cleaned)
    # Remove braces/brackets
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = cleaned.replace("[", "").replace("]", "")
    cleaned = cleaned.replace("(", "").replace(")", "")
    # Remove digits, operators, spaces, dots, commas
    cleaned = re.sub(r"[\d+\-*/^.,\s]", "", cleaned)
    # If anything remains, it's likely a variable name
    # Allow common math symbols
    cleaned = cleaned.replace("=", "").replace("!", "").replace(">", "").replace("<", "")
    # After all removals, if there are alphabetic characters left,
    # they're likely variable names -> not purely numeric
    return not re.search(r"[a-zA-Z]", cleaned)


# ── Safe expression evaluation ────────────────────────────────────


def _latex_to_python(expr: str) -> str | None:
    """Convert a LaTeX math expression to a Python-evaluable string."""
    if not expr or not expr.strip():
        return None

    s = expr.strip()

    # Non-ASCII multiplication sign (U+00D7) -> *
    s = s.replace("×", "*")
    # Repeating decimals require exact semantics. Fail closed instead of
    # silently changing 0.\overline{6} into 0.6.
    if r"\overline" in s:
        return None

    # Support literal integer binomial coefficients exactly. Symbolic or
    # malformed binomials remain unsupported and fail closed below.
    def _replace_binomial(match: re.Match[str]) -> str:
        n = int(match.group(1))
        k = int(match.group(2))
        if n < 0 or k < 0 or k > n:
            raise ValueError("invalid binomial coefficient")
        return str(math.comb(n, k))

    try:
        s = re.sub(
            r"\\binom\{\s*([+-]?\d+)\s*\}\{\s*([+-]?\d+)\s*\}",
            _replace_binomial,
            s,
        )
    except ValueError:
        return None
    if r"\binom" in s:
        return None
    # LaTeX -> Python replacements (order matters)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\{", "(").replace("\\}", ")")
    s = s.replace("\\(", "(").replace("\\)", ")")
    s = s.replace("\\times", "*")
    s = s.replace("\\div", "/")
    s = s.replace("\\cdot", "*")
    s = s.replace("\\pi", "pi")
    s = s.replace("\\infty", "inf")
    s = s.replace("\\sqrt", "sqrt")
    s = s.replace("\\ln", "log")
    s = s.replace("\\log", "log")
    s = s.replace("\\exp", "exp")
    s = s.replace("\\sin", "sin")
    s = s.replace("\\cos", "cos")
    s = s.replace("\\tan", "tan")
    s = s.replace("\\abs", "abs")
    # \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    # Remove remaining LaTeX commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # Braces -> parens
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("[", "(").replace("]", ")")
    # ^ -> **
    s = s.replace("^", "**")
    # Remove formatting spaces
    s = s.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    s = s.replace("\\ ", " ").replace("~", " ")
    # Remove \text{...} etc
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\mathrm\{[^}]*\}", "", s)
    # Implicit multiplication (order matters):
    # 1) digit directly before letter: 2pi -> 2*pi
    s = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", s)
    # 2) digit followed by optional whitespace then '(': 6 ( -> 6*(, 3( -> 3*(
    s = re.sub(r"(\d)\s*\(", r"\1*(", s)
    # 3) ')' followed by optional whitespace then digit: )2 -> )*2
    s = re.sub(r"\)\s*(\d)", r")*\1", s)
    # 4) ')' followed by optional whitespace then '(': ) ( -> )*(
    s = re.sub(r"\)\s*\(", r")*(", s)
    # 5) ')' followed by optional whitespace then letter: )pi -> )*pi
    s = re.sub(r"\)\s*([a-zA-Z])", r")*\1", s)

    return s.strip() or None


def _safe_eval(expr_str: str) -> float | None:
    """Safely evaluate a Python math expression to a float."""
    if not expr_str:
        return None

    safe_ns = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "ln": math.log,
        "exp": math.exp,
        "inf": float("inf"),
        "floor": math.floor,
        "ceil": math.ceil,
        "factorial": math.factorial,
    }

    try:
        result = eval(expr_str, safe_ns, {})  # noqa: S307
        if isinstance(result, (int, float)) and math.isfinite(result):
            return float(result)
        return None
    except Exception:
        return None


_NUMERIC_LITERAL_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def _is_trivial_literal_identity(lhs_str: str, rhs_str: str) -> bool:
    """Reject bare literal equalities such as 1=1 as process evidence."""
    lhs = lhs_str.strip().replace(",", "")
    rhs = rhs_str.strip().replace(",", "")
    return bool(_NUMERIC_LITERAL_RE.fullmatch(lhs) and _NUMERIC_LITERAL_RE.fullmatch(rhs))


def _verify_equation_pair(lhs_str: str, rhs_str: str) -> bool:
    """Check if LHS = RHS is numerically true after LaTeX conversion."""
    if _is_trivial_literal_identity(lhs_str, rhs_str):
        return False

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


def _process_reward_audit(solution_str: str) -> dict[str, Any]:
    """Verify each reasoning step and retain equation-level audit details.

    Only steps that contain at least one numeric equation participate in
    the process reward average.  Steps without equations (prose, variable
    manipulation, etc.) are excluded from the denominator so they neither
    help nor hurt the process score.
    """
    steps = _split_into_steps(solution_str)
    step_audits: list[dict[str, Any]] = []
    total_equations = 0
    verified_equations = 0

    for index, step in enumerate(steps, start=1):
        equations = _extract_numeric_equations(step)
        equation_results = [_verify_equation_pair(lhs, rhs) for lhs, rhs in equations]
        step_verified = sum(equation_results)
        step_total = len(equations)
        step_reward = step_verified / step_total if step_total else 0.0
        total_equations += step_total
        verified_equations += step_verified
        step_audits.append(
            {
                "step_index": index,
                "equation_count": step_total,
                "verified_equation_count": step_verified,
                "step_reward": step_reward,
                "equation_checks": [{"lhs": lhs, "rhs": rhs, "valid": bool(valid)}
                                    for (lhs, rhs), valid in zip(equations, equation_results)],
            }
        )

    # Only average over steps that have at least one extractable equation.
    graded_audits = [a for a in step_audits if a["equation_count"] > 0]
    process_reward = sum(audit["step_reward"] for audit in graded_audits) / len(graded_audits) if graded_audits else 0.0
    return {
        "process_reward": process_reward,
        "step_count": len(step_audits),
        "graded_step_count": len(graded_audits),
        "fully_verified_step_count": sum(
            audit["equation_count"] > 0 and audit["step_reward"] == 1.0 for audit in step_audits
        ),
        "equation_count": total_equations,
        "verified_equation_count": verified_equations,
        "steps": step_audits,
    }


def compute_process_reward(solution_str: str) -> tuple[float, int, int]:
    """Compute process reward by verifying all extracted equations per step.

    Only steps with at least one numeric equation are graded.

    Returns:
        (process_reward, num_fully_verified_steps, num_graded_steps)
    """
    audit = _process_reward_audit(solution_str)
    return (
        float(audit["process_reward"]),
        int(audit["fully_verified_step_count"]),
        int(audit["graded_step_count"]),
    )


# ── Terminal EM ───────────────────────────────────────────────────


def compute_terminal_em(solution_str: str, ground_truth: str) -> float:
    final = final_answer_record(solution_str)
    if final is None or ground_truth is None:
        return 0.0
    answer = str(final["answer"]).strip()
    if answer == str(ground_truth).strip():
        return 1.0
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", answer) and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", str(ground_truth).strip()):
        from decimal import Decimal
        return float(Decimal(answer) == Decimal(str(ground_truth).strip()))
    from verl.utils.reward_score import math_reward
    try:
        return float(math_reward.is_equiv(answer, str(ground_truth)))
    except (ValueError, TypeError):
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
      Steps 1-WARMUP:  strict terminal EM
      Steps WARMUP+1+: one w ~ U(0,1) per prompt group,
                       reward = w * EM + (1-w) * process_reward
    """
    if data_source not in _DEEPMATH_DATA_SOURCES:
        from verl.utils.reward_score import default_compute_score

        return default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)

    if ground_truth is None or not solution_str:
        return 0.0

    from agent_r1.evaluation.consistency import consistency_record
    from recipes.deepscaler.trajectory_reward import compute_tool_trajectory_reward, verify_process

    extra_info = extra_info or {}
    final = final_answer_record(solution_str)
    reasoning = str(final["reasoning"]) if final else solution_str
    verification = verify_process([(1, reasoning)])
    global_step = int(extra_info.get("_agent_r1_global_step", os.environ.get(_ENV_GLOBAL_STEP, "0")))
    result = compute_tool_trajectory_reward(
        reward_mode="uniform_equation_process",
        final_response=solution_str if final else None, ground_truth=ground_truth,
        reasoning_segments=[(1, reasoning)], global_step=global_step,
        is_validation=bool(extra_info.get("_agent_r1_is_validation", False)),
        em_warmup_steps=EM_WARMUP_STEPS,
        prompt_group_key=prompt_group_key_from_extra_info(extra_info, data_source=data_source),
        process_verification=verification,
    )
    return {
        **result.record(),
        **consistency_record(verification, result.terminal_em, eligible=final is not None),
        "acc": result.terminal_em,
        "deterministic_verification": verification.audit,
        "num_extracted_equations": sum(a["equation_count"] for a in verification.audit["process_segment_audits"]),
        "training_global_step": global_step,
    }
