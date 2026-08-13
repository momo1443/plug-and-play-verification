"""Gold-free behavioral probes for the standalone A9 reward arm."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

A9_VERIFIER_VERSION = "hotpotqa-a9-behavioral-v1"
A9_PROBE_GENERATOR_VERSION = "hotpotqa-a9-probe-generator-v1"
A9_ENTITY_LIBRARY_PATH = Path(__file__).with_name("a9_entity_library.json")

_CAPITALIZED_SPAN = re.compile(
    r"(?<![\w])[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5}(?![\w])",
)
_WORD_CHAR = re.compile(r"[A-Za-z0-9_]")
_TYPE_CUES = {
    "person": (
        "actor", "actress", "author", "born", "composer", "director", "he ",
        " her ", " his ", "married", "nationality", "novelist", "poet", "she ",
        "spouse", "writer",
    ),
    "place": (
        "capital", "city", "country", "county", "district", "island", "located",
        "mountain", "municipality", "province", "region", "river", "territory", "where",
    ),
    "organization": (
        "association", "club", "company", "corporation", "council", "founded", "foundation",
        "institute", "organization", "school", "society", "team", "university",
    ),
    "work": (
        "album", "book", "film", "novel", "published", "recorded", "released", "series",
        "song", "television", "written by",
    ),
}
_DISTRACTOR_TEMPLATES = {
    "person": "{main} was discussed alongside {distractor} in a biographical index published in {year}.",
    "place": "{main} was discussed alongside {distractor} in a geographic index published in {year}.",
    "organization": "{main} was discussed alongside {distractor} in an institutional directory from {year}.",
    "work": "{main} was discussed alongside {distractor} in a cultural review published in {year}.",
}
_DISTRACTOR_RELATION_TERMS = frozenset(
    {"alongside", "biographical", "cultural", "directory", "discussed", "geographic", "index", "review"}
)


@dataclass(frozen=True)
class TargetSlot:
    surface: str
    query_start: int
    query_end: int
    entity_type: str
    source_passage_index: int
    suffix: str


@dataclass(frozen=True)
class ProbePlan:
    sample_key: str
    turn_index: int
    probe_seed: int
    target: TargetSlot
    sensitivity_candidate: str
    distractor_candidate: str
    distractor_sentence: str
    sensitivity_passages: tuple[dict[str, Any], ...]
    invariance_passages: tuple[dict[str, Any], ...]
    entity_library_version: str
    generator_version: str = A9_PROBE_GENERATOR_VERSION


@dataclass(frozen=True)
class CandidateEncoding:
    token_ids: tuple[int, ...]
    target_token_start: int
    target_token_end: int


@dataclass(frozen=True)
class CandidateScore:
    candidate: str
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    token_average_logprob: float
    policy_snapshot_id: str


@dataclass(frozen=True)
class BehavioralVerification:
    eligible: bool
    no_valid_probe: bool
    sensitivity_pass: bool
    invariance_pass: bool
    joint_pass: bool
    raw_process_reward: float | None
    artifact: dict[str, Any]

    @property
    def process_component_valid(self) -> bool:
        """Whether the binary A9 process verdict is known for optimization."""

        return self.eligible and self.raw_process_reward in (0.0, 1.0)

    @property
    def optimizer_process_value(self) -> float:
        return float(self.raw_process_reward) if self.process_component_valid else 0.0

    @property
    def verdict_class(self) -> str:
        return _verdict_class(self.eligible, self.raw_process_reward)


def _verdict_class(eligible: bool, raw_process_reward: float | None) -> str:
    if not eligible:
        return "C"
    return {1.0: "A", 0.0: "B", 0.5: "D"}[raw_process_reward]


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_entity_library(path: Path = A9_ENTITY_LIBRARY_PATH) -> tuple[str, dict[str, tuple[str, ...]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = str(payload.pop("schema_version"))
    library = {
        str(entity_type): tuple(str(item).strip() for item in values if str(item).strip())
        for entity_type, values in payload.items()
    }
    if set(library) != set(_TYPE_CUES):
        raise ValueError(f"A9 entity library types must be {sorted(_TYPE_CUES)}, got {sorted(library)}")
    if any(len(values) < 4 for values in library.values()):
        raise ValueError("A9 entity library requires at least four candidates per type")
    return version, library


def _contains_exact(text: str, surface: str) -> bool:
    start = 0
    while True:
        index = text.find(surface, start)
        if index < 0:
            return False
        left_ok = index == 0 or not _WORD_CHAR.match(text[index - 1])
        end = index + len(surface)
        right_ok = end == len(text) or not _WORD_CHAR.match(text[end])
        if left_ok and right_ok:
            return True
        start = index + 1


def _replace_exact(text: str, old: str, new: str) -> tuple[str, int]:
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])")
    return pattern.subn(new, text)


def _passage_text(record: Mapping[str, Any]) -> str:
    # The retained raw HotpotQA prompt renders passage text, not title metadata.
    return str(record.get("text", "")).strip()


def _classify_entity(target: str, query: str, source_text: str) -> str | None:
    query_start = query.find(target)
    suffix = query[query_start + len(target) :] if query_start >= 0 else query
    source_index = source_text.find(target)
    if source_index >= 0:
        source_window = source_text[max(0, source_index - 180) : source_index + len(target) + 240]
    else:
        source_window = source_text[:420]
    evidence = f" {suffix} {source_window} ".lower()
    scores = {
        entity_type: sum(1 for cue in cues if cue in evidence)
        for entity_type, cues in _TYPE_CUES.items()
    }
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if not ordered or ordered[0][1] == 0:
        return None
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None
    return ordered[0][0]


def parse_target_slot(query: str, passages: Sequence[Mapping[str, Any]]) -> TargetSlot | None:
    """Find one observation-grounded, typed proper-name target in a search query."""

    query = str(query)
    candidates: dict[str, int] = {}
    for passage_index, record in enumerate(passages):
        text = _passage_text(record)
        surfaces = {match.group(0).strip() for match in _CAPITALIZED_SPAN.finditer(text)}
        for surface in surfaces:
            if len(surface) < 3 or query.count(surface) != 1 or not _contains_exact(query, surface):
                continue
            previous = candidates.get(surface)
            if previous is None or passage_index < previous:
                candidates[surface] = passage_index

    for surface, passage_index in sorted(
        candidates.items(), key=lambda item: (-len(item[0]), item[1], item[0])
    ):
        start = query.find(surface)
        source_text = _passage_text(passages[passage_index])
        entity_type = _classify_entity(surface, query, source_text)
        if entity_type is None:
            continue
        return TargetSlot(
            surface=surface,
            query_start=start,
            query_end=start + len(surface),
            entity_type=entity_type,
            source_passage_index=passage_index,
            suffix=query[start + len(surface) :].strip(),
        )
    return None


def _select_candidate(
    values: Sequence[str],
    *,
    seed_material: str,
    forbidden_text: str,
    target: str,
    token_length: Callable[[str], int],
    excluded: frozenset[str] = frozenset(),
) -> str | None:
    target_length = token_length(target)
    eligible = [
        value
        for value in values
        if value not in excluded
        and value != target
        and not _contains_exact(forbidden_text, value)
        and 0 < token_length(value)
    ]
    if not eligible:
        return None
    ranked = sorted(
        eligible,
        key=lambda value: (
            abs(token_length(value) - target_length),
            _canonical_hash({"seed": seed_material, "candidate": value}),
        ),
    )
    return ranked[0]


def build_probe_plan(
    *,
    query: str,
    question: str,
    history_actions: Sequence[str],
    passages: Sequence[Mapping[str, Any]],
    sample_key: str,
    turn_index: int,
    token_length: Callable[[str], int],
    probe_seed: int = 42,
    entity_library_path: Path = A9_ENTITY_LIBRARY_PATH,
) -> ProbePlan | None:
    """Build one sensitivity and one D-on invariance probe without gold data."""

    if not passages:
        return None
    target = parse_target_slot(query, passages)
    if target is None:
        return None
    if set(re.findall(r"[a-z]+", target.suffix.lower())) & _DISTRACTOR_RELATION_TERMS:
        return None

    library_version, library = load_entity_library(entity_library_path)
    observation = "\n".join(_passage_text(record) for record in passages)
    forbidden = "\n".join([question, *history_actions, observation])
    base_seed = f"{probe_seed}:{sample_key}:{turn_index}:{target.surface}:{target.entity_type}"
    sensitivity_candidate = _select_candidate(
        library[target.entity_type],
        seed_material=f"{base_seed}:sensitivity",
        forbidden_text=forbidden,
        target=target.surface,
        token_length=token_length,
    )
    if sensitivity_candidate is None:
        return None
    distractor_candidate = _select_candidate(
        library[target.entity_type],
        seed_material=f"{base_seed}:invariance",
        forbidden_text=forbidden,
        target=target.surface,
        token_length=token_length,
        excluded=frozenset({sensitivity_candidate}),
    )
    if distractor_candidate is None:
        return None

    sensitivity_passages: list[dict[str, Any]] = []
    replacement_count = 0
    for record in passages:
        copied = dict(record)
        copied["text"], text_count = _replace_exact(
            str(copied.get("text", "")), target.surface, sensitivity_candidate
        )
        replacement_count += text_count
        sensitivity_passages.append(copied)
    if replacement_count == 0:
        return None
    sensitivity_text = "\n".join(_passage_text(record) for record in sensitivity_passages)
    if _contains_exact(sensitivity_text, target.surface):
        return None

    years_in_observation = {int(value) for value in re.findall(r"\b(?:18|19|20)\d{2}\b", observation)}
    year = next((candidate for candidate in range(1931, 2000, 7) if candidate not in years_in_observation), None)
    if year is None:
        return None
    visible_entities = [
        match.group(0).strip()
        for record in passages
        for match in _CAPITALIZED_SPAN.finditer(str(record.get("text", "")))
    ]
    main_entity = next(
        (surface for surface in visible_entities if surface != target.surface and " " in surface),
        target.surface,
    )
    distractor_sentence = _DISTRACTOR_TEMPLATES[target.entity_type].format(
        main=main_entity,
        distractor=distractor_candidate,
        year=year,
    )
    invariance_passages = [dict(record) for record in passages]
    invariance_passages[0]["text"] = (
        f"{distractor_sentence} {str(invariance_passages[0].get('text', '')).strip()}".strip()
    )
    invariance_text = "\n".join(_passage_text(record) for record in invariance_passages)
    if not _contains_exact(invariance_text, target.surface) or not _contains_exact(
        invariance_text, distractor_candidate
    ):
        return None

    return ProbePlan(
        sample_key=sample_key,
        turn_index=int(turn_index),
        probe_seed=int(probe_seed),
        target=target,
        sensitivity_candidate=sensitivity_candidate,
        distractor_candidate=distractor_candidate,
        distractor_sentence=distractor_sentence,
        sensitivity_passages=tuple(sensitivity_passages),
        invariance_passages=tuple(invariance_passages),
        entity_library_version=library_version,
    )


def visible_probe_inputs_valid(
    plan: ProbePlan,
    *,
    sensitivity_passages: Sequence[Mapping[str, Any]],
    invariance_passages: Sequence[Mapping[str, Any]],
) -> bool:
    """Check that prompt truncation preserved both intended interventions."""

    if len(sensitivity_passages) != len(plan.sensitivity_passages):
        return False
    if len(invariance_passages) != len(plan.invariance_passages):
        return False
    sensitivity_text = "\n".join(_passage_text(record) for record in sensitivity_passages)
    invariance_text = "\n".join(_passage_text(record) for record in invariance_passages)
    return (
        _contains_exact(sensitivity_text, plan.sensitivity_candidate)
        and not _contains_exact(sensitivity_text, plan.target.surface)
        and _contains_exact(invariance_text, plan.target.surface)
        and _contains_exact(invariance_text, plan.distractor_candidate)
    )


def encode_action_candidate(
    tokenizer: Any,
    *,
    response_text: str,
    query: str,
    target: TargetSlot,
    candidate: str,
) -> CandidateEncoding | None:
    """Tokenize the sampled action prefix through a replacement target slot."""

    if response_text.count(query) != 1:
        return None
    query_start = response_text.find(query)
    target_start = query_start + target.query_start
    target_end = query_start + target.query_end
    if response_text[target_start:target_end] != target.surface:
        return None
    candidate_text = response_text[:target_start] + candidate
    encoded = tokenizer(
        candidate_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = tuple(int(value) for value in encoded["input_ids"])
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    candidate_start = target_start
    candidate_end = target_start + len(candidate)
    positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > candidate_start and start < candidate_end
    ]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        return None
    if offsets[positions[0]][0] < candidate_start or offsets[positions[-1]][1] > candidate_end:
        return None
    return CandidateEncoding(
        token_ids=token_ids,
        target_token_start=positions[0],
        target_token_end=positions[-1] + 1,
    )


def extract_candidate_score(
    *,
    candidate: str,
    full_prompt_token_ids: Sequence[int],
    absolute_target_start: int,
    absolute_target_end: int,
    prompt_logprob_ids: Sequence[Sequence[int]],
    prompt_logprobs: Sequence[Sequence[float]],
    policy_snapshot_id: str,
) -> CandidateScore | None:
    """Extract chosen-token prompt logprobs returned by verl/vLLM."""

    if absolute_target_start <= 0 or absolute_target_end <= absolute_target_start:
        return None
    if len(prompt_logprob_ids) != len(full_prompt_token_ids) or len(prompt_logprobs) != len(
        full_prompt_token_ids
    ):
        return None
    expected_ids = tuple(int(value) for value in full_prompt_token_ids[absolute_target_start:absolute_target_end])
    values: list[float] = []
    for token_position, expected_id in zip(
        range(absolute_target_start, absolute_target_end), expected_ids
    ):
        extraction_index = token_position - 1
        returned_ids = prompt_logprob_ids[extraction_index]
        returned_values = prompt_logprobs[extraction_index]
        if len(returned_ids) != 1 or len(returned_values) != 1:
            return None
        if int(returned_ids[0]) != expected_id:
            return None
        value = float(returned_values[0])
        if not math.isfinite(value):
            return None
        values.append(value)
    if not values:
        return None
    return CandidateScore(
        candidate=candidate,
        token_ids=expected_ids,
        token_logprobs=tuple(values),
        token_average_logprob=sum(values) / len(values),
        policy_snapshot_id=str(policy_snapshot_id),
    )


def verify_behavioral_scores(
    *,
    plan: ProbePlan,
    sensitivity_new: CandidateScore,
    sensitivity_old: CandidateScore,
    invariance_target: CandidateScore,
    invariance_distractor: CandidateScore,
    source_state_hash: str,
    expected_policy_snapshot_id: str,
) -> BehavioralVerification:
    scores = (sensitivity_new, sensitivity_old, invariance_target, invariance_distractor)
    snapshots = {score.policy_snapshot_id for score in scores}
    if len(snapshots) != 1:
        return invalid_verification("policy_snapshot_mismatch")
    if snapshots != {str(expected_policy_snapshot_id)}:
        return invalid_verification("sampled_action_snapshot_mismatch")
    sensitivity_pass = sensitivity_new.token_average_logprob > sensitivity_old.token_average_logprob
    invariance_pass = invariance_target.token_average_logprob > invariance_distractor.token_average_logprob
    raw_reward = (float(sensitivity_pass) + float(invariance_pass)) / 2.0
    artifact: dict[str, Any] = {
        "schema_version": A9_VERIFIER_VERSION,
        "eligible": True,
        "no_valid_probe": False,
        "generator_version": plan.generator_version,
        "entity_library_version": plan.entity_library_version,
        "probe_seed": plan.probe_seed,
        "sample_key": plan.sample_key,
        "turn_index": plan.turn_index,
        "policy_snapshot_id": next(iter(snapshots)),
        "sampled_action_policy_snapshot_id": str(expected_policy_snapshot_id),
        "source_state_hash": source_state_hash,
        "target": asdict(plan.target),
        "sensitivity_candidate": plan.sensitivity_candidate,
        "distractor_candidate": plan.distractor_candidate,
        "distractor_sentence": plan.distractor_sentence,
        "transformed_state_hashes": {
            "sensitivity": _canonical_hash(plan.sensitivity_passages),
            "invariance": _canonical_hash(plan.invariance_passages),
        },
        "scores": {
            "sensitivity_new": asdict(sensitivity_new),
            "sensitivity_old": asdict(sensitivity_old),
            "invariance_target": asdict(invariance_target),
            "invariance_distractor": asdict(invariance_distractor),
        },
        "sensitivity_pass": sensitivity_pass,
        "invariance_pass": invariance_pass,
        "joint_pass": sensitivity_pass and invariance_pass,
        "raw_process_reward": raw_reward,
        "process_component_valid": raw_reward in (0.0, 1.0),
        "verdict_class": _verdict_class(True, raw_reward),
        "validity_checks": {
            "candidate_scores_finite": True,
            "policy_snapshot_consistent": True,
            "gold_inputs_used": False,
        },
    }
    artifact["replay_hash"] = _canonical_hash(artifact)
    return BehavioralVerification(
        eligible=True,
        no_valid_probe=False,
        sensitivity_pass=sensitivity_pass,
        invariance_pass=invariance_pass,
        joint_pass=sensitivity_pass and invariance_pass,
        raw_process_reward=raw_reward,
        artifact=artifact,
    )


def replay_behavioral_artifact(artifact: Mapping[str, Any]) -> bool:
    """Recompute a persisted A9 result without model or gold inputs."""

    payload = dict(artifact)
    replay_hash = payload.pop("replay_hash", None)
    if not isinstance(replay_hash, str) or _canonical_hash(payload) != replay_hash:
        return False
    if payload.get("schema_version") != A9_VERIFIER_VERSION:
        return False
    if payload.get("eligible") is False:
        return (
            payload.get("raw_process_reward") is None
            and payload.get("process_component_valid") is False
            and payload.get("verdict_class") == "C"
            and isinstance(payload.get("reason"), str)
        )
    if payload.get("eligible") is not True or payload.get("no_valid_probe") is not False:
        return False
    validity = payload.get("validity_checks")
    if not isinstance(validity, Mapping) or validity.get("gold_inputs_used") is not False:
        return False
    if validity.get("candidate_scores_finite") is not True:
        return False
    if validity.get("policy_snapshot_consistent") is not True:
        return False
    scores = payload.get("scores")
    if not isinstance(scores, Mapping):
        return False

    parsed: dict[str, tuple[float, str]] = {}
    for name in (
        "sensitivity_new",
        "sensitivity_old",
        "invariance_target",
        "invariance_distractor",
    ):
        score = scores.get(name)
        if not isinstance(score, Mapping):
            return False
        token_ids = score.get("token_ids")
        token_logprobs = score.get("token_logprobs")
        average = score.get("token_average_logprob")
        snapshot = score.get("policy_snapshot_id")
        if (
            not isinstance(token_ids, (list, tuple))
            or not isinstance(token_logprobs, (list, tuple))
            or not token_ids
            or len(token_ids) != len(token_logprobs)
            or not isinstance(snapshot, str)
        ):
            return False
        try:
            values = [float(value) for value in token_logprobs]
            stored_average = float(average)
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in values) or not math.isfinite(stored_average):
            return False
        computed_average = sum(values) / len(values)
        if not math.isclose(computed_average, stored_average, rel_tol=0.0, abs_tol=1e-12):
            return False
        parsed[name] = (stored_average, snapshot)

    snapshots = {snapshot for _, snapshot in parsed.values()}
    expected_snapshot = str(payload.get("policy_snapshot_id"))
    sampled_snapshot = str(payload.get("sampled_action_policy_snapshot_id"))
    if snapshots != {expected_snapshot} or sampled_snapshot != expected_snapshot:
        return False
    sensitivity_pass = parsed["sensitivity_new"][0] > parsed["sensitivity_old"][0]
    invariance_pass = parsed["invariance_target"][0] > parsed["invariance_distractor"][0]
    raw_reward = (float(sensitivity_pass) + float(invariance_pass)) / 2.0
    return (
        payload.get("sensitivity_pass") is sensitivity_pass
        and payload.get("invariance_pass") is invariance_pass
        and payload.get("joint_pass") is (sensitivity_pass and invariance_pass)
        and payload.get("raw_process_reward") == raw_reward
        and payload.get("process_component_valid") is (raw_reward in (0.0, 1.0))
        and payload.get("verdict_class") == _verdict_class(True, raw_reward)
    )


def invalid_verification(reason: str) -> BehavioralVerification:
    artifact = {
        "schema_version": A9_VERIFIER_VERSION,
        "eligible": False,
        "reason": str(reason),
        "raw_process_reward": None,
        "process_component_valid": False,
        "verdict_class": "C",
    }
    artifact["replay_hash"] = _canonical_hash(artifact)
    return BehavioralVerification(
        eligible=False,
        no_valid_probe=True,
        sensitivity_pass=False,
        invariance_pass=False,
        joint_pass=False,
        raw_process_reward=None,
        artifact=artifact,
    )
