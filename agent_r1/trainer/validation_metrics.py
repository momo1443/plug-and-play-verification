"""Select scalar metrics while preserving rich reward audits for logging."""

from numbers import Real


def scalar_validation_metrics(infos_dict):
    """Keep numeric columns and string predictions supported by veRL.

    Equation audits contain dictionaries and lists. They remain in the original
    mapping for JSONL output but cannot be averaged by process_validation_metrics.
    """
    return {
        key: values
        for key, values in infos_dict.items()
        if values and (
            all(isinstance(value, Real) for value in values)
            or (key == "pred" and all(isinstance(value, str) for value in values))
        )
    }
