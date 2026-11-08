"""Invoke a call-based solution inside the code-execution sandbox.

Only arguments and the function name enter this process. Expected outputs are
compared by the parent executor and are never mounted in the sandbox.
"""

from __future__ import annotations

import contextlib
import io
import json
import runpy
import sys
import typing


def main() -> None:
    payload = json.loads(sys.stdin.read())
    arguments = payload["arguments"]
    # Follow TACO's integer-key convention for a dictionary argument.
    if arguments and isinstance(arguments[0], dict):
        try:
            arguments[0] = {int(key): value for key, value in arguments[0].items()}
        except ValueError:
            pass
    result_stream = sys.stdout
    # Candidate diagnostics must not become the serialized function result.
    with contextlib.redirect_stdout(io.StringIO()):
        scope = runpy.run_path(sys.argv[1], init_globals={
            name: getattr(typing, name) for name in ("List", "Tuple", "Dict", "Set", "Optional", "Any")
        })
        target = scope["Solution"]() if "Solution" in scope else None
        function = getattr(target, payload["function_name"]) if target is not None else scope[payload["function_name"]]
        result = function(*arguments)
    result_stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
