"""Bounded arithmetic with exact rationals and explicit LaTeX semantics.

Unknown notation is excluded from process evidence, never silently deleted.
This parser checks arithmetic; it does not prove a problem's reasoning.
"""
from __future__ import annotations

import ast
import math
import re
import threading
from fractions import Fraction
from functools import lru_cache


def normalize_thousands(text: str) -> str:
    # Parenthesized comma lists and function arguments are not integer groups.
    protected = [(m.start(), m.end()) for m in re.finditer(r"\([^()]*,[^()]*\)|\[[^\[\]]*,[^\[\]]*\]", text)]
    def replace(m):
        return m.group() if any(a <= m.start() < b for a, b in protected) else m.group().replace(",", "")
    return re.sub(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\w|\.\d)", replace, text)


def split_top_level(text: str, separators: str) -> list[str]:
    parts, start, depth = [], 0, 0
    for i, char in enumerate(text):
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth -= 1
        elif depth == 0 and char in separators:
            parts.append(text[start:i]); start = i + 1
    parts.append(text[start:])
    return parts


def _group(text: str, start: int) -> tuple[str, int]:
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text):
        raise ValueError("missing argument")
    if text[start] != "{":
        # An unbraced LaTeX argument is exactly one token, not a whole number.
        if text[start].isdigit():
            return text[start], start + 1
        raise ValueError("unsupported unbraced argument")
    depth, end = 1, start + 1
    while end < len(text) and depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    if depth:
        raise ValueError("unclosed argument")
    return text[start + 1:end - 1], end


@lru_cache(maxsize=32768)
def latex_to_python(expression: str) -> str | None:
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 4096:
        return None
    s = normalize_thousands(expression.strip())
    if "..." in s or "…" in s or re.search(r"\\(?:ldots|dots|cdots|approx|equiv|pmod|bmod)\b", s):
        return None
    s = s.replace("−", "-").replace("×", "*").replace("÷", "/")
    s = re.sub(r"\\text\{\s*of\s*\}", " of ", s, flags=re.I)
    s = re.sub(r"\bof\b", "*", s, flags=re.I)
    s = re.sub(r"\bhalf\s*\*", "(1/2)*", s, flags=re.I)
    s = re.sub(r"\bsquare\s*\*\s*(\d+(?:\.\d+)?)", r"(\1)^2", s, flags=re.I)
    if re.search(r"/\s*\d+(?:\.\d+)?\s*(?:\\(?:pi|sqrt)|pi\b|sqrt\b)|\\pi\s*\(", s):
        return None  # Ambiguous implicit denominator or a named pi function.
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\(?:,|;|!|quad\b|qquad\b| )", " ", s)
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    if any(int(m[2]) >= int(m[3]) for m in re.finditer(
            r"(?<![\w.])([+-]?\d+)\s+\\frac\s*\{(\d+)\}\s*\{([1-9]\d*)\}", s)):
        return None  # Improper mixed-number/multiplication notation is ambiguous.
    # The conventional mixed number 7\frac12 means 7 + 1/2, not 7/2.
    s = re.sub(r"(?<![\w.])([+-]?\d+)\s*\\frac\s*\{(\d+)\}\s*\{(\d+)\}",
               lambda m: ("-(" + m[1][1:] + "+\\frac{" + m[2] + "}{" + m[3] + "})"
                          if m[1].startswith("-") else
                          "(" + m[1] + "+\\frac{" + m[2] + "}{" + m[3] + "})"), s)
    # Explicit repeating decimals have exact rational semantics.
    def repeating(m):
        whole, prefix, repeat = m[1], m[2], m[3]
        value = Fraction(int(whole + prefix), 10 ** len(prefix)) + Fraction(
            int(repeat), (10 ** len(repeat) - 1) * 10 ** len(prefix))
        return f"({value.numerator}/{value.denominator})"
    s = re.sub(r"(\d+)\.(\d*)\\overline\{(\d+)\}", repeating, s)
    # A numeric subscript is a base, never a Python digit separator.
    def base_number(m):
        digits, base = m[1] or m[2], int(m[3] or m[4])
        if not 2 <= base <= 36:
            raise ValueError("invalid base")
        return str(int(digits, base))
    try:
        s = re.sub(r"(?<![\w.)])(?:\(([0-9A-Fa-f]+)\)|([0-9]+))_\s*(?:\{(\d+)\}|(\d+))",
                   base_number, s)
        # Consume nested arguments recursively; simple regexes lose semantics.
        pattern = re.compile(r"\\(frac|sqrt|binom)\b")
        while (m := pattern.search(s)) is not None:
            kind, end = m[1], m.end()
            root = "2"
            if kind == "sqrt" and end < len(s) and s[end] == "[":
                close = s.index("]", end)
                root, end = s[end + 1:close], close + 1
            a, end = _group(s, end)
            a_py = latex_to_python(a)
            if a_py is None:
                return None
            if kind in {"frac", "binom"}:
                b, end = _group(s, end)
                arguments = split_top_level(b, ",") if kind == "binom" else [b]
                args_py = [latex_to_python(argument) for argument in arguments]
                if any(argument is None for argument in args_py):
                    return None
                replacement = (f"(({a_py})/({args_py[0]}))" if kind == "frac" else
                               f"{'multinomial' if len(args_py) > 1 else 'comb'}({a_py},{','.join(args_py)})")
            else:
                root_py = latex_to_python(root)
                if root_py is None:
                    return None
                replacement = f"root({a_py},{root_py})"
            s = s[:m.start()] + replacement + s[end:]
    except (ValueError, OverflowError):
        return None
    # Degree-marked trig arguments are explicit; bare sin(30) is ambiguous.
    s = re.sub(r"\\?(sin|cos|tan)\s*\(([^()]*)\^\s*(?:\{\\circ\}|\\circ)\)",
               lambda m: f"{m[1]}d({m[2]})", s)
    s = s.replace(r"\times", "*").replace(r"\cdot", "*").replace(r"\div", "/")
    s = s.replace(r"\pi", "pi")
    s = s.replace(r"\lfloor", "floor(").replace(r"\rfloor", ")")
    s = s.replace(r"\lceil", "ceil(").replace(r"\rceil", ")")
    for name in ("sin", "cos", "tan", "ln", "log", "exp", "abs"):
        s = re.sub(r"\\" + name + r"\b", name, s)
    s = re.sub(r"\\mathrm\{e\}", "exp(1)", s)
    s = s.replace(r"\%", "%")
    s = re.sub(r"(\d+(?:\.\d+)?|\([^()]+\))\s*%", r"(\1/100)", s)
    if "\\" in s or "_" in s or "%" in s or "|" in s:
        return None
    s = s.replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")")
    s = s.replace("^", "**")
    # Postfix factorial of a literal or group. Other forms stay unsupported.
    for _ in range(8):
        new = re.sub(r"(\d+|\([^()]+\))!", r"factorial(\1)", s)
        if new == s: break
        s = new
    s = re.sub(r"(\d)(pi\b|(?:sin|cos|tan|ln|log|exp|sqrt|root|comb|factorial|multinomial)\b)", r"\1*\2", s)
    s = re.sub(r"(\d)\s*\(", r"\1*(", s)
    s = re.sub(r"\)\s*(?=[\d(]|pi\b|(?:sin|cos|tan|ln|log|exp|sqrt|root|comb|factorial|multinomial)\b)", ")*", s)
    s = re.sub(r"\bpi\s*\(", "pi*(", s)
    return s.strip() or None


_FUNCTIONS = {"sqrt", "root", "abs", "sin", "cos", "tan", "sind", "cosd", "tand",
              "log", "ln", "exp", "floor", "ceil", "factorial", "comb", "multinomial"}


class ArithmeticLimit(ValueError):
    pass


def numeric_tree(expression: str) -> ast.Expression | None:
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return None
    nodes = list(ast.walk(tree))
    if len(nodes) > 256:
        return None
    callees = {id(node.func) for node in nodes if isinstance(node, ast.Call)}
    for node in nodes:
        if isinstance(node, ast.Name):
            if node.id != "pi" and (node.id not in _FUNCTIONS or id(node) not in callees):
                return None
        if isinstance(node, ast.Constant) and type(node.value) not in (int, float):
            return None
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS or node.keywords:
                return None
            if node.func.id in {"sin", "cos", "tan"} and not any(
                    isinstance(n, ast.Name) and n.id == "pi" for n in ast.walk(node)):
                return None
        if not isinstance(node, (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Call,
                                 ast.UnaryOp, ast.UAdd, ast.USub, ast.BinOp,
                                 ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            return None
    return tree


@lru_cache(maxsize=32768)
def evaluation(expression: str) -> tuple[Fraction | float | None, str]:
    tree = numeric_tree(expression)
    if tree is None:
        return None, "unsupported"
    def bounded(value):
        if isinstance(value, Fraction):
            if max(value.numerator.bit_length(), value.denominator.bit_length()) > 65536:
                raise ArithmeticLimit("arithmetic size limit")
        elif not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("nonfinite arithmetic")
        return value
    def visit(node):
        if isinstance(node, ast.Constant):
            return bounded(Fraction(ast.get_source_segment(expression, node)))
        if isinstance(node, ast.Name):
            if node.id == "pi": return math.pi
            raise ValueError("unresolved name")
        if isinstance(node, ast.UnaryOp):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): value = a + b
            elif isinstance(node.op, ast.Sub): value = a - b
            elif isinstance(node.op, ast.Mult): value = a * b
            elif isinstance(node.op, ast.Div): value = a / b
            else:
                if abs(b) > 10000: raise ArithmeticLimit("exponent limit")
                if isinstance(b, Fraction) and b.denominator == 1:
                    if isinstance(a, Fraction) and abs(b) * max(a.numerator.bit_length(), a.denominator.bit_length()) > 65536:
                        raise ArithmeticLimit("power size limit")
                    value = a ** int(b)
                else: value = float(a) ** float(b)
            return bounded(value)
        if isinstance(node, ast.Call):
            name, args = node.func.id, [visit(a) for a in node.args]
            if name == "root":
                a, n = args
                if n != int(n) or n < 1: raise ValueError("root degree")
                if n > 10000: raise ArithmeticLimit("root degree limit")
                if a < 0 and int(n) % 2 == 0: raise ValueError("nonreal root")
                value = (-1 if a < 0 else 1) * abs(float(a)) ** (1 / int(n))
            elif name == "factorial":
                if len(args) != 1 or args[0] != int(args[0]) or args[0] < 0:
                    raise ValueError("factorial domain")
                if args[0] > 5000: raise ArithmeticLimit("factorial limit")
                value = Fraction(math.factorial(int(args[0])))
            elif name == "comb":
                if len(args) != 2 or args[1] != int(args[1]) or args[1] < 0:
                    raise ValueError("binomial domain")
                n, k = args[0], int(args[1])
                if not isinstance(n, Fraction):
                    raise ArithmeticLimit("non-rational binomial upper argument")
                if abs(n) > 10000000 or k > 5000:
                    raise ArithmeticLimit("binomial limit")
                if n.denominator == 1 and n >= 0:
                    value = Fraction(math.comb(int(n), k))
                else:
                    # Generalized binomial coefficients include fractional or
                    # negative upper arguments; they are valid math evidence.
                    value = Fraction(1)
                    for i in range(k):
                        value = bounded(value * (n - i) / (i + 1))
            elif name == "multinomial":
                if len(args) < 3 or any(a != int(a) or a < 0 for a in args) or sum(args[1:]) != args[0]:
                    raise ValueError("multinomial domain")
                if args[0] > 5000: raise ArithmeticLimit("multinomial limit")
                remaining, value = int(args[0]), Fraction(1)
                for count in args[1:]:
                    value = bounded(value * math.comb(remaining, int(count)))
                    remaining -= int(count)
            elif name == "log":
                value = math.log(float(args[0]), float(args[1]) if len(args) == 2 else 10)
            elif name == "ln": value = math.log(float(args[0]))
            elif name.endswith("d") and name[:-1] in {"sin", "cos", "tan"}:
                value = getattr(math, name[:-1])(math.radians(float(args[0])))
            else: value = (abs(args[0]) if name == "abs" else getattr(math, name)(*args))
            return bounded(value)
        raise ValueError("unsupported arithmetic")
    try:
        return visit(tree.body), "ok"
    except ArithmeticLimit:
        return None, "limit"
    except (ValueError, TypeError, OverflowError, ZeroDivisionError, RecursionError):
        return None, "invalid_domain"


def evaluate(expression: str) -> Fraction | float | None:
    return evaluation(expression)[0]


def numerically_equal(a, b) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        return False  # Literal decimals and rationals have exact semantics.
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-10)
    except (OverflowError, ValueError):
        return False


def canonical_answer(answer: str) -> str:
    s = str(answer).strip()
    # Strip paired presentation delimiters only, never an internal operator.
    for left, right in (("$$", "$$"), ("$", "$"), (r"\[", r"\]"), (r"\(", r"\)")):
        if s.startswith(left) and s.endswith(right):
            s = s[len(left):-len(right)].strip()
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = re.sub(r"\\[,;!]", "", s)
    if not s.endswith(("...", "…")): s = s.rstrip(".").strip()
    s = re.sub(r"\^\s*(?:\{\\circ\}|\\circ)|°", "", s)
    s = re.sub(r"\\(?:text|mathrm)\{\s*(?:degrees?|dollars?|cents?|units?|feet|inches?|cm|m|hours?|minutes?|weeks?)\s*\}\s*$", "", s, flags=re.I)
    s = re.sub(r"^([+-]?)\\?\$", r"\1", s).strip()
    s = normalize_thousands(s).strip()
    # Plain units follow the same numeric-only normalization as LaTeX units.
    # Requiring whitespace preserves symbolic products such as 50m.
    unit = re.search(r"\s+(?:degrees?|dollars?|cents?|units?|feet|inches?|cm|m|hours?|minutes?|weeks?)\s*$", s, re.I)
    if unit:
        value = s[:unit.start()].strip()
        expression = latex_to_python(value)
        if expression and evaluate(expression) is not None:
            s = value
    return s


@lru_cache(maxsize=8192)
def _monomial_answer(answer: str):
    """Bounded exact comparison for products of single-letter variables.

    This avoids interpreter startup/parsing deadlines for answers such as
    ``x^{-1}y^{-1}``, ``(xy)^{-1}`` and ``1/(xy)``. No eval or symbolic parser
    is involved. Sums, functions, equations and unknown notation fall back.
    """
    if len(answer) > 256:
        return None
    if re.search(r"[A-Za-z]\d", answer):
        return None
    expression = latex_to_python(answer)
    if expression is None or re.search(r"\b(?:pi|[a-z]{3,})\b", expression, re.I):
        return None
    tokens = re.findall(r"\*\*|\d+(?:\.\d+)?|[A-Za-z]|[()+*/-]", expression)
    if "".join(tokens) != re.sub(r"\s+", "", expression):
        return None
    expanded = []
    for token in tokens:
        if expanded:
            previous = expanded[-1]
            ends_atom = previous == ")" or previous[0].isalnum()
            starts_atom = token == "(" or token[0].isalnum()
            if ends_atom and starts_atom:
                expanded.append("*")
        expanded.append(token)
    try:
        tree = ast.parse("".join(expanded), mode="eval")
        if len(list(ast.walk(tree))) > 128:
            return None
        def visit(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return Fraction(str(node.value)), {}
            if isinstance(node, ast.Name) and re.fullmatch(r"[A-Za-z]", node.id):
                return Fraction(1), {node.id: 1}
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                coefficient, powers = visit(node.operand)
                return (-coefficient if isinstance(node.op, ast.USub) else coefficient), powers
            if not isinstance(node, ast.BinOp):
                raise ValueError("unsupported monomial")
            a, pa = visit(node.left)
            b, pb = visit(node.right)
            if isinstance(node.op, ast.Pow):
                if pb or b.denominator != 1 or abs(b) > 64:
                    raise ValueError("unsupported monomial exponent")
                coefficient = a ** int(b)
                powers = {name: exponent * int(b) for name, exponent in pa.items()}
            elif isinstance(node.op, (ast.Mult, ast.Div)):
                sign = -1 if isinstance(node.op, ast.Div) else 1
                coefficient = a / b if sign == -1 else a * b
                powers = dict(pa)
                for name, exponent in pb.items():
                    powers[name] = powers.get(name, 0) + sign * exponent
            else:
                raise ValueError("unsupported monomial operator")
            if max(coefficient.numerator.bit_length(), coefficient.denominator.bit_length()) > 4096 or len(powers) > 8:
                raise ValueError("monomial size limit")
            return coefficient, powers
        coefficient, powers = visit(tree.body)
        return coefficient, tuple(sorted((name, exponent) for name, exponent in powers.items() if exponent))
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
        return None


@lru_cache(maxsize=16384)
def _parsed_answer(answer: str):
    from math_verify import LatexExtractionConfig, parse
    return parse("$" + answer + "$", extraction_config=[LatexExtractionConfig()],
                 fallback_mode="no_fallback", extraction_mode="first_match", parsing_timeout=2)


def _symbolic_equal(a: str, b: str) -> bool:
    from math_verify import verify
    try:
        ga, gb = _parsed_answer(a), _parsed_answer(b)
        return bool(ga and gb and verify(gb, ga, strict=True, float_rounding=12, timeout_seconds=2))
    except (ValueError, TypeError, TimeoutError, RecursionError):
        return False


@lru_cache(maxsize=16384)
def _isolated_symbolic_equal(a: str, b: str) -> bool:
    """Keep signal-based parser deadlines in an isolated process main thread.

    Ray async actors run on a worker thread. Disabling the library's deadline
    there would allow a symbolic comparison to block the entire rollout actor.
    The child retains two-second library deadlines and an eight-second overall
    deadline covering interpreter startup, both parses, and comparison.
    """
    import json
    import subprocess
    import sys
    try:
        result = subprocess.run(
            [sys.executable, __file__], input=json.dumps([a, b]),
            capture_output=True, text=True, timeout=8,
        )
        if result.returncode != 0 or result.stdout.strip() not in {"true", "false"}:
            raise RuntimeError("symbolic worker failed")
        return result.stdout.strip() == "true"
    except (OSError, subprocess.TimeoutExpired) as exc:
        # Transient execution failures must not be cached forever as mathematical
        # inequality. lru_cache stores only successful calls, not exceptions.
        raise RuntimeError("symbolic worker unavailable or timed out") from exc


def answers_equal(prediction: str, target: str) -> bool:
    a, b = canonical_answer(prediction), canonical_answer(target)
    if not a or not b or any(x in a or x in b for x in ("...", "…")):
        return False
    def constraint_form(text):
        parts = re.split(r"(?<![<>!=])=(?!=)", text)
        # A value assignment (x=5) may label a scalar answer. A polynomial
        # equation is a different requested artifact from its set of roots.
        return len(parts) > 1 and not (len(parts) == 2 and re.fullmatch(r"\s*[A-Za-z]\s*", parts[0]))
    has_equation = lambda text: bool(re.search(r"(?<![<>!=])=(?!=)", text))
    if ((constraint_form(a) and not has_equation(b)) or
            (constraint_form(b) and not has_equation(a))):
        return False
    pa, pb = latex_to_python(a), latex_to_python(b)
    va, vb = evaluate(pa) if pa else None, evaluate(pb) if pb else None
    if a == b:
        # An explicit numeric expression with an invalid domain is not a
        # correct answer merely because both strings are identical.
        return not (pa and evaluation(pa)[1] == 'invalid_domain')
    if va is not None and vb is not None:
        return numerically_equal(va, vb)
    ma, mb = _monomial_answer(a), _monomial_answer(b)
    if ma is not None and mb is not None:
        return ma == mb
    # Compare only the already-selected answer, never search the derivation.
    if max(len(a), len(b)) > 4096:
        return False
    if threading.current_thread() is not threading.main_thread():
        try:
            return _isolated_symbolic_equal(a, b)
        except RuntimeError as exc:
            import logging
            logging.getLogger(__name__).warning("Terminal symbolic comparison failed: %s", exc)
            return False
    return _symbolic_equal(a, b)


if __name__ == "__main__":
    import json
    import sys
    pair = json.loads(sys.stdin.read(65537))
    if (not isinstance(pair, list) or len(pair) != 2 or
            any(not isinstance(s, str) or len(s) > 4096 for s in pair)):
        raise ValueError("Expected two bounded answer strings")
    print(json.dumps(_symbolic_equal(*pair)))
