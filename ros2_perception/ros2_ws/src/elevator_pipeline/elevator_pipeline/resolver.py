"""Deterministic target selection from task context and observations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    active: bool
    request_id: str
    building_id: str
    requested_floor: str


@dataclass(frozen=True)
class Candidate:
    found: bool
    request_id: str
    building_id: str
    requested_floor: str
    effective_label: str
    button: object
    candidate_confidence: float
    reason: str


def resolve_target(task, observations, floor_map, minimum_confidence=0.0):
    if not task.active:
        return Candidate(
            found=False,
            request_id=task.request_id,
            building_id=task.building_id,
            requested_floor=task.requested_floor,
            effective_label='',
            button=None,
            candidate_confidence=0.0,
            reason='task_inactive',
        )

    floor_entry = floor_map.get_floor(task.building_id, task.requested_floor)
    effective_label = floor_entry.displayed_label
    eligible = []
    for observation in observations:
        if not observation.valid:
            continue
        if not observation.pressable:
            continue
        if observation.button_type != 'floor_button':
            continue
        if observation.overall_confidence < minimum_confidence:
            continue
        if not floor_map.matches(observation.label, floor_entry):
            continue
        eligible.append(observation)

    if not eligible:
        return Candidate(
            found=False,
            request_id=task.request_id,
            building_id=task.building_id,
            requested_floor=task.requested_floor,
            effective_label=effective_label,
            button=None,
            candidate_confidence=0.0,
            reason='no_matching_observation',
        )
    selected = sorted(
        eligible,
        key=lambda item: (-item.overall_confidence, item.detection_id),
    )[0]
    return Candidate(
        found=True,
        request_id=task.request_id,
        building_id=task.building_id,
        requested_floor=task.requested_floor,
        effective_label=effective_label,
        button=selected,
        candidate_confidence=selected.overall_confidence,
        reason='matched_effective_label_or_alias',
    )
