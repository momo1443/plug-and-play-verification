"""Reward function for DeepScaleR with A9-style uniform process reward schedule.

Schedule (controlled by DEEPSCALER_EM_WARMUP_STEPS, default 50):
  - Steps 1-WARMUP:   strict outcome-only (EM).
  - Steps WARMUP+1+:  one w ~ U(0,1) per prompt group,
                       reward = w * EM + (1-w) * process_reward

Process reward:
  Checks supported, asserted numeric equalities in every reasoning turn.
  Unsupported or ambiguous contexts are excluded; real arithmetic failures
  remain in the denominator. A turn without evidence receives zero credit.

    step_reward = (# verified equations) / (# extracted equations)
    process_reward = mean(step_reward over equation-bearing reasoning segments)

  Trivial literal identities such as ``1 = 1`` do not receive credit. This
  prevents one unrelated tautology from making an otherwise invalid step pass.

  This is a deterministic, execution-based process signal — no PRM needed.
"""

from __future__ import annotations

import ast
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
    """Remove presentation prefixes, never search prose for a numeric suffix.

    Searching for the first digit changed ``Day 2`` into ``2`` and discarded
    operators in expressions such as ``20% of 100``. Unknown labels stay
    unsupported; the arithmetic later in an equality chain is still checked.
    """
    from recipes.deepscaler.math_expressions import normalize_thousands
    lhs = re.sub(r"\\?\$", "", normalize_thousands(lhs)).strip()
    lhs = re.sub(
        r"^\s*(?:Step\s+\d+\s*[:.]|\d+(?:\.(?!\d)|\)))\s*",
        "", lhs, flags=re.IGNORECASE,
    )
    lhs = re.sub(
        r"^(?:Check(?: the arithmetic)?|Rechecked|Continue with|Calculation(?: check)?|Compute|Calculate|Verify|"
        r"We (?:compute|calculate|verify)|First|Then|Next|Finally|So|Therefore|Thus|Hence)[,\s]+", "", lhs, flags=re.I,
    )
    return lhs.strip()


def _clean_rhs(rhs: str) -> str:
    """Remove trailing noise from RHS."""
    from recipes.deepscaler.math_expressions import normalize_thousands, split_top_level
    rhs = normalize_thousands(rhs).strip()
    rhs = re.sub(r"^(?:\\?\$)+", "", rhs)
    # A closing inline delimiter terminates this expression. A leading dollar
    # sign above is a presentation/currency prefix, not an empty expression.
    rhs = re.split(r"(?<!\\)\$", rhs, maxsplit=1)[0]
    rhs = re.split(r"(?<!\.)\.(?=\s+[A-Za-z])", rhs, maxsplit=1)[0]
    rhs = split_top_level(rhs, ",;")[0]
    rhs = re.split(r"\b(?:and|then|next)\b", rhs, maxsplit=1, flags=re.IGNORECASE)[0]
    # Remove trailing period/comma
    if not rhs.rstrip().endswith(("...", "…")):
        rhs = rhs.rstrip(".,;")
    return rhs.strip()


def _extract_numeric_equations(step_text: str, *, exclude_star: bool = False,
                               truncated: bool = False, exclude_pi: bool = False,
                               excluded: list[dict] | None = None,
                               prior_context: str = "") -> list[tuple[str, str]]:
    """Extract (lhs, rhs) pairs that look like purely numeric equations.

    Only extracts equations where both sides appear to be numeric
    expressions (no unresolved variables). This ensures we can verify
    them by evaluation.
    """
    equations: list[tuple[str, str]] = []
    seen: set[str] = set()
    digit_display_context = False

    def _exclude(lhs, rhs, reason):
        if excluded is not None:
            excluded.append({"lhs": lhs, "rhs": rhs, "reason": reason})

    def _try_add(lhs: str, rhs: str) -> None:
        if exclude_star and re.search(r"\*|×", lhs + rhs):
            return
        if exclude_pi and re.search(r"\\pi\b|\bpi\b", lhs + rhs):
            _exclude(lhs, rhs, "pi_used_as_a_variable")
            return
        lhs = _clean_lhs(lhs)
        rhs = _clean_rhs(rhs)
        if not lhs or not rhs:
            return
        if lhs.startswith("+") or (re.search(r"(?:\*|\\times)\s*100\s*$", lhs) and "%" in rhs):
            # Running-total fragments (+7=42) and percentage display shortcuts
            # are not complete mathematical equalities.
            return
        # Quick check: both sides should have at least one digit
        if not re.search(r"\d", lhs) and not re.search(r"\d", rhs):
            return
        # Both sides should be purely numeric expressions
        # (no single-letter variables that would remain unresolved)
        if _is_numeric_expr(lhs) and _is_numeric_expr(rhs):
            from recipes.deepscaler.math_expressions import evaluate, numerically_equal
            a, b = evaluate(_latex_to_python(lhs)), evaluate(_latex_to_python(rhs))
            if (digit_display_context and a is not None and b is not None
                    and a == int(a) and b == int(b) and 0 <= b <= 9
                    and a != b and int(a) % 10 == int(b)):
                _exclude(lhs, rhs, "units_digit_display")
                return
            if ("%" in rhs and "%" not in lhs and a is not None and b is not None
                    and numerically_equal(a, b * 100)):
                _exclude(lhs, rhs, "percentage_display_without_lhs_units")
                return
            if _is_rounded_display(lhs, rhs):
                _exclude(lhs, rhs, "rounded_decimal_display")
                return
            # A repeated literal is neither new process evidence nor a failed
            # arithmetic check. Incorrect literal equalities still fail.
            if _is_trivial_literal_identity(lhs, rhs) and _safe_eval(_latex_to_python(lhs)) == _safe_eval(_latex_to_python(rhs)):
                return
            key = f"{lhs}={rhs}"
            if key not in seen:
                seen.add(key)
                equations.append((lhs, rhs))

    # Keep expressions spanning several inline math/currency blocks intact:
    # ``20% of $100 = 20$`` and ``$135 - $105 = $30`` must not lose their LHS.
    lines = step_text.splitlines()
    context_lines = prior_context.splitlines()[-3:] + lines
    context_offset = len(context_lines) - len(lines)
    last_nonempty = max((i for i, line in enumerate(lines) if line.strip()), default=-1)
    running_total = False
    for index, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^(?:[-*•]\s+)", "", line)
        digit_display_context = bool(re.search(
            r"(?:units|last|ones)[ -]+digits?|possible ends|ends? (?:in|for|with)",
            "\n".join(context_lines[max(0, context_offset + index - 3):context_offset + index + 1]), re.I,
        ))
        bare_line = re.sub(r"\\?\$", "", line).strip()
        continuation = re.fullmatch(r"\s*[+-]\s*\d+(?:\.\d+)?\s*=\s*[+-]?\d+(?:\.\d+)?\s*[.]?", bare_line)
        if continuation and running_total:
            _exclude(bare_line.split("=", 1)[0], bare_line.split("=", 1)[1],
                     "running_total_requires_previous_value")
            continue
        parts = bare_line.split("=")
        running_total = (len(parts) == 2 and _is_numeric_expr(_clean_lhs(parts[0]))
                         and _is_numeric_expr(_clean_rhs(parts[1]))
                         and bool(re.search(r"\d\s*[+*/-]|\\(?:times|cdot)", parts[0])))
        unfinished = (truncated and index == last_nonempty
                      and not re.search(r"(?:\$|</think>|(?<!\d)[.!?])\s*$", line))
        _extract_eqs_from_text(line, _try_add, skip_last=unfinished,
                              exclude_fn=_exclude)

    return equations


def _extract_eqs_from_text(text: str, add_fn, *, skip_last: bool = False,
                            exclude_fn=None) -> None:
    """Extract 'LHS = RHS' from a piece of text, calling add_fn for each."""
    # Split on lines first
    from recipes.deepscaler.math_expressions import normalize_thousands, split_top_level
    segments = re.split(r"\\\\|\n|\b(?:and|then|next)\b", text, flags=re.IGNORECASE)
    candidates = []
    for item in segments:
        # A label before the first equality may end in a colon. A colon on
        # the RHS (``1=2:(1``) is unsupported notation, not a new equation.
        equality = item.find("=")
        if equality >= 0:
            prefix, suffix = item[:equality], item[equality:]
            item = split_top_level(prefix, ":")[-1] + suffix
        pieces = split_top_level(normalize_thousands(item), ",;")
        for index, piece in enumerate(pieces):
            # Comma-separated digits are a numeral/list, not several equations.
            # Signed comma fragments are running totals, not unary assertions.
            if index and re.search(r"\d", pieces[index - 1]):
                if "=" not in pieces[index - 1] or re.match(r"\s*[+-]", piece):
                    if exclude_fn and "=" in piece:
                        exclude_fn(piece, "", "numeric_list_or_running_total")
                    continue
            candidates.append(piece)
    for segment_index, segment in enumerate(candidates):
        segment = segment.strip()
        if not segment:
            continue
        # Check adjacent expressions in a chain without dropping a later
        # failure (2+3=5=6). Comparisons are never equality boundaries.
        # An approximate/congruent expression cannot be the exact RHS. Retain
        # only exact equalities before the non-equality boundary.
        segment = re.split(r"\\(?:approx|equiv|sim)\b|≈|≡", segment, maxsplit=1)[0]
        parts = re.split(r"(?<![=<>!])=(?!=)", segment)
        for pair_index, (lhs, rhs) in enumerate(zip(parts, parts[1:])):
            if (pair_index == 1 and re.search(
                    r"\b(?:for|when|if|case)\s+\$?\s*[A-Za-z]\s*$", parts[0], re.I)
                    and re.fullmatch(r"\s*\$?\s*[+-]?\d+(?:\.\d+)?\s*\$?\s*", lhs)):
                # "Total ways for D=5 = 12+24 = 36" labels a case, not an
                # assertion that the case number equals the computed total.
                if exclude_fn:
                    exclude_fn(lhs, rhs, "conditional_label_not_equation")
                continue
            if skip_last and segment_index == len(candidates) - 1 and pair_index == len(parts) - 2:
                if exclude_fn:
                    exclude_fn(lhs, rhs, "generation_truncated_before_equation_end")
                continue
            add_fn(lhs.strip(), rhs.strip())


def _is_rounded_display(lhs: str, rhs: str) -> bool:
    """Exclude plausible rounding, without awarding it exact-equality credit."""
    from recipes.deepscaler.math_expressions import evaluate, numerically_equal
    match = re.fullmatch(r"[+-]?\d*\.(\d{2,12})\s*(?:\\?%)?", rhs.strip())
    if match is None:
        return False
    a, b = evaluate(_latex_to_python(lhs)), evaluate(_latex_to_python(rhs))
    if a is None or b is None or numerically_equal(a, b):
        return False
    unit = 10.0 ** -len(match[1]) / (100 if "%" in rhs else 1)
    try:
        # Both rounding and truncation are common display conventions. Neither
        # earns exact credit when its literal value differs from the LHS.
        return abs(float(a) - float(b)) < unit + 1e-12
    except (ValueError, OverflowError):
        return False


def _is_numeric_expr(expr: str) -> bool:
    from recipes.deepscaler.math_expressions import numeric_tree, evaluation
    converted = _latex_to_python(expr)
    return converted is not None and numeric_tree(converted) is not None and evaluation(converted)[1] != 'limit'


# ── Safe expression evaluation ────────────────────────────────────


def _latex_to_python(expr: str) -> str | None:
    from recipes.deepscaler.math_expressions import latex_to_python
    return latex_to_python(expr)


def _safe_eval(expr_str: str):
    from recipes.deepscaler.math_expressions import evaluate
    return evaluate(expr_str) if expr_str else None


_NUMERIC_LITERAL_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def _is_trivial_literal_identity(lhs_str: str, rhs_str: str) -> bool:
    """Reject bare literal equalities such as 1=1 as process evidence."""
    lhs = lhs_str.strip().replace(",", "")
    rhs = rhs_str.strip().replace(",", "")
    return bool(_NUMERIC_LITERAL_RE.fullmatch(lhs) and _NUMERIC_LITERAL_RE.fullmatch(rhs))


def _verify_equation_pair(lhs_str: str, rhs_str: str) -> bool:
    from recipes.deepscaler.math_expressions import numerically_equal
    if _is_trivial_literal_identity(lhs_str, rhs_str):
        return False
    lhs, rhs = _latex_to_python(lhs_str), _latex_to_python(rhs_str)
    return numerically_equal(_safe_eval(lhs), _safe_eval(rhs))


# ── Process reward computation ────────────────────────────────────


def _process_reward_audit(solution_str: str, *, question: str = "", truncated: bool = False) -> dict[str, Any]:
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
    modular_context = bool(re.search(r"\\(?:pmod|bmod)\b|\bmod(?:ulo)?\s+\d+|\bmodular\b", solution_str, re.I))
    custom_star = bool(re.search(r"(?:\*|×).{0,80}\b(?:means|defined|denotes)\b", question, re.I | re.S))
    pi_variable = bool(re.search(r"\b(?:probability|let|denote)\b[^\n.]{0,60}(?:\\pi\b|\bpi\b)\s*=", solution_str, re.I))

    for index, step in enumerate(steps, start=1):
        exclusion = ("modular_arithmetic_context" if modular_context else
                     "quotient_and_remainder_context" if re.search(r"\b(?:remainder|rem)\b", step, re.I) else
                     "refuted_equality_context" if re.search(r"\bis false\b|\bcontradiction\b", step, re.I) else
                     "unit_conversion_context" if re.search(r"\bcents\b", step, re.I) else
                     "rounded_or_approximate_context" if re.search(r"\b(?:rounded|approximately|approximation)\b", step, re.I) else None)
        excluded_equations: list[dict] = []
        equations = [] if exclusion else _extract_numeric_equations(
            step, exclude_star=custom_star, exclude_pi=pi_variable,
            truncated=(truncated and index == len(steps)
                       and solution_str.rstrip().endswith(step.rstrip())),
            excluded=excluded_equations,
            prior_context="\n".join(steps[max(0, index - 3):index - 1]),
        )
        equation_results = [_verify_equation_pair(lhs, rhs) for lhs, rhs in equations]
        step_verified = sum(equation_results)
        step_total = len(equations)
        step_reward = step_verified / step_total if step_total else 0.0
        total_equations += step_total
        verified_equations += step_verified
        step_audits.append(
            {
                "step_index": index,
                "context_exclusion": exclusion,
                "custom_star_excluded": custom_star,
                "excluded_equations": excluded_equations,
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
    from recipes.deepscaler.math_expressions import answers_equal
    final = final_answer_record(solution_str)
    return float(final is not None and ground_truth is not None and
                 answers_equal(str(final["answer"]), str(ground_truth)))



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
    is_validation = bool(extra_info.get("_agent_r1_is_validation", False))
    result = compute_tool_trajectory_reward(
        reward_mode="uniform_equation_process",
        final_response=solution_str if final else None, ground_truth=ground_truth,
        reasoning_segments=[(1, reasoning)], global_step=global_step,
        is_validation=is_validation,
        em_warmup_steps=EM_WARMUP_STEPS,
        prompt_group_key=(prompt_group_key_from_extra_info(extra_info, data_source=data_source)
                          if global_step > EM_WARMUP_STEPS and not is_validation else None),
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
