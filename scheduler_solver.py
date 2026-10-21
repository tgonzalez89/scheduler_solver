import math
from collections.abc import Callable
from typing import Any, Literal, get_args

from ortools.sat.python import cp_model

# ==========================================================
# Constants
# ==========================================================


_MAX_AVAILABILITY_SCORE = 4
ObjectiveName = Literal[
    "coverage",
    "best_slot",
    "minimum_fulfillment",
    "activities",
    "total_attendance",
    "total_preference",
]
_VALID_OBJECTIVES = get_args(ObjectiveName)
DEFAULT_OBJECTIVE_ORDER = _VALID_OBJECTIVES
# Per slot, either {person: score} or a frozenset of people who all get the top
# score.
AvailabilityInput = dict[str, dict[str, int] | frozenset[str]]


# ==========================================================
# Input Normalization
# ==========================================================


def _normalize_availability(
    availability: AvailabilityInput,
) -> dict[str, dict[str, int]]:
    """Convert frozenset-valued slots to {person: max score} dicts."""
    return {
        slot: (
            people
            if isinstance(people, dict)
            else dict.fromkeys(people, _MAX_AVAILABILITY_SCORE)
        )
        for slot, people in availability.items()
    }


# ==========================================================
# Validation Functions
# ==========================================================


def _validate_capacity(capacity: dict[str, int]) -> None:
    if any(not person.strip() for person in capacity):
        raise ValueError("capacity keys must be non-empty person names")
    for person, limit in capacity.items():
        if limit < 0:
            raise ValueError(f"capacity for {person!r} must be non-negative")


def _validate_availability(
    availability: dict[str, dict[str, int]], capacity: dict[str, int]
) -> None:
    for slot, person_scores in availability.items():
        if not slot.strip():
            raise ValueError("availability keys must not be empty")
        if any(not person.strip() for person in person_scores):
            raise ValueError(
                f"availability for slot {slot!r} contains an empty person name"
            )
        for person, score in person_scores.items():
            if not 0 <= score <= _MAX_AVAILABILITY_SCORE:
                raise ValueError(
                    f"availability score for {person!r} at {slot!r} must be "
                    f"between 0 and {_MAX_AVAILABILITY_SCORE}"
                )

    unknown_people = {
        person
        for person_scores in availability.values()
        for person in person_scores
        if person not in capacity
    }
    if unknown_people:
        raise ValueError(
            "availability contains people missing from capacity: "
            f"{sorted(unknown_people)}"
        )


def _validate_min_people(min_people: int, capacity: dict[str, int]) -> None:
    if min_people < 1:
        raise ValueError("min_people must be a positive integer")
    people_with_capacity = sum(limit > 0 for limit in capacity.values())
    if min_people > people_with_capacity:
        raise ValueError(
            "min_people cannot exceed the number of people with positive capacity"
        )


def _validate_groups(
    availability: dict[str, dict[str, int]], groups: frozenset[frozenset[str]] | None
) -> None:
    if groups is None:
        return

    grouped_slots = set()
    availability_slots = set(availability)
    for group_index, group in enumerate(groups):
        if len(group) < 2:
            raise ValueError(f"group {group_index} must contain at least two slots")
        if any(not slot.strip() for slot in group):
            raise ValueError(f"group {group_index} contains an empty slot name")
        unknown_slots = set(group) - availability_slots
        if unknown_slots:
            raise ValueError(
                f"group {group_index} contains slots missing from availability: "
                f"{sorted(unknown_slots)}"
            )

        overlapping_slots = grouped_slots.intersection(group)
        if overlapping_slots:
            raise ValueError(
                f"slots may appear in only one group: {sorted(overlapping_slots)}"
            )
        grouped_slots.update(group)


def _validate_objective_order(
    objective_order: tuple[ObjectiveName, ...],
) -> None:
    if (
        not objective_order
        or len(objective_order) != len(set(objective_order))
        or not set(objective_order).issubset(_VALID_OBJECTIVES)
    ):
        raise ValueError(
            "objective_order must contain one or more unique objectives from "
            f"{', '.join(map(repr, _VALID_OBJECTIVES))}"
        )


def _validate_inputs(
    capacity: dict[str, int],
    availability: dict[str, dict[str, int]],
    min_people: int,
    groups: frozenset[frozenset[str]] | None,
    objective_order: tuple[ObjectiveName, ...],
) -> None:
    _validate_capacity(capacity)
    _validate_availability(availability, capacity)
    _validate_min_people(min_people, capacity)
    _validate_groups(availability, groups)
    _validate_objective_order(objective_order)


# ==========================================================
# Result Functions
# ==========================================================


def _ideal_metrics(
    capacity: dict[str, int], availability: dict[str, dict[str, int]]
) -> dict[str, int]:
    """Best values the input allows if every person got their favorite slots.

    Ignores slot conflicts, groups, and min_people, so these are upper bounds
    that a real schedule may not reach.
    """
    ideal = {
        "coverage_ideal": 0,
        "best_slot_ideal": 0,
        "total_attendance_ideal": 0,
        "total_preference_ideal": 0,
    }
    for person, limit in capacity.items():
        scores = sorted(
            (
                person_scores[person]
                for person_scores in availability.values()
                if person_scores.get(person, 0) > 0
            ),
            reverse=True,
        )[:limit]
        if scores:
            ideal["coverage_ideal"] += 1
            ideal["best_slot_ideal"] += scores[0]
            ideal["total_attendance_ideal"] += len(scores)
            ideal["total_preference_ideal"] += sum(scores)
    return ideal


def _summarize(
    participants: dict[str, list[str]],
    capacity: dict[str, int],
    availability: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Build a schedule result, computing every metric from the participants."""
    attendance = {person: 0 for person in sorted(capacity)}
    best_score = {person: 0 for person in sorted(capacity)}
    for slot, attendees in participants.items():
        for person in attendees:
            attendance[person] += 1
            best_score[person] = max(best_score[person], availability[slot][person])

    covered = [person for person, count in attendance.items() if count > 0]
    return {
        "coverage_count": len(covered),
        "best_slot_score": sum(best_score.values()),
        "total_attendance": sum(attendance.values()),
        "total_preference": sum(
            availability[slot][person]
            for slot, attendees in participants.items()
            for person in attendees
        ),
        **_ideal_metrics(capacity, availability),
        "minimum_fulfillment": min(
            (100 * attendance[person] // capacity[person] for person in covered),
            default=0.0,
        ),
        "activity_count": len(participants),
        "activities": list(participants),
        "participation_count": attendance,
        "participants": participants,
    }


def _filter_prefer_spread(schedules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not schedules:
        return schedules

    def consecutive_repeats(schedule: dict[str, Any]) -> int:
        return sum(
            len(
                set(schedule["participants"][first_slot])
                & set(schedule["participants"][second_slot])
            )
            for first_slot, second_slot in zip(
                schedule["activities"], schedule["activities"][1:]
            )
        )

    best_repeat_count = min(consecutive_repeats(schedule) for schedule in schedules)
    return [
        schedule
        for schedule in schedules
        if consecutive_repeats(schedule) == best_repeat_count
    ]


def _enumerate_solutions(
    model: cp_model.CpModel,
    extract_participants: Callable[[Callable[..., int]], dict[str, list[str]]],
) -> list[dict[str, list[str]]]:
    """Return the participants of every feasible solution of `model`."""
    model.clear_objective()

    class SolutionCollector(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.solutions = []

        def OnSolutionCallback(self):
            self.solutions.append(extract_participants(self.value))

    collector = SolutionCollector()
    enumerator = cp_model.CpSolver()
    enumerator.parameters.enumerate_all_solutions = True
    enumerator.parameters.num_workers = 1
    enumerator.solve(model, collector)
    return collector.solutions


# ==========================================================
# Main Solver Function
# ==========================================================


def solve_schedule(
    capacity: dict[str, int],
    availability: AvailabilityInput,
    min_people: int,
    groups: frozenset[frozenset[str]] | None = None,
    objective_order: tuple[ObjectiveName, ...] = DEFAULT_OBJECTIVE_ORDER,
    return_all_solutions: bool = True,
    prefer_spread: bool = True,
) -> list[dict[str, Any]]:
    """Optimize activity schedules subject to availability and capacity.

    Parameters
    ----------
    capacity : dict[str, int]
        Maps each person to their maximum number of activities. This is the
        authoritative list of valid people; values must be non-negative.

        Example:
            {"Mario": 2, "Carlos": 3, "Ana": 1}

    availability : dict[str, dict[str, int] | frozenset[str]]
        Maps each slot to per-person availability scores from 0 to 4. Zero means
        unavailable; positive scores mean eligible, with 4 as the strongest
        preference. People not mentioned for a slot count as 0. A slot may
        instead map to a frozenset of names, meaning those people score 4 and
        everyone else 0. Every person named here must also appear in
        `capacity`.

        Example:
            {
                "09-06 Fri": {"Mario": 4, "Ana": 2},
                "09-07 Sat": {"Mario": 3, "Carlos": 4},
                "09-08 Sun": frozenset({"Carlos", "Ana"}),
            }

    min_people : int
        Minimum number of attendees required for every selected activity. Must
        be positive and no larger than the number of people in `capacity`.

    groups : frozenset[frozenset[str]] | None
        Optional, disjoint groups of known slots. At most one activity may be
        selected from each group. A slot may occur in only one group, and each
        group must contain at least two distinct slots.

        Example:
            {
                {"09-06 Fri", "09-07 Sat", "09-08 Sun"},
                {"09-13 Fri", "09-14 Sat", "09-15 Sun"},
            }

    objective_order : tuple[str, ...]
        Non-empty ordered subset of the supported objectives. Each objective
        may appear once; objectives not listed are not optimized.

        * coverage: Include as many different people as possible, even if
          some only get a slot they rated low. Nobody should be left out when
          there is a way to fit them in.
          Definition: maximize the number of people with at least one
          assignment. Any positive score counts.
        * best_slot: Give each person at least one slot they really like.
          This raises the best slot each person gets, rather than giving extra
          dates to people who already have a good one.
          Definition: maximize the sum, over people, of the best score among
          their assignments (0 if unassigned).
        * minimum_fulfillment: Share activities fairly, so no participant
          gets far fewer activities than they said they could attend.
          Definition: maximize the lowest assigned/capacity ratio among
          covered participants.
        * activities: Hold as many activities as possible.
          Definition: maximize the number of selected activities.
        * total_attendance: Fill as many seats as possible across all
          activities. Unlike coverage, repeat attendances by the same person
          also count.
          Definition: maximize the number of person/slot assignments.
        * total_preference: Fill seats with people who rated those dates highly.
          Like total_attendance, every attendance counts, but each one is
          weighted by the person's score for that slot.
          Definition: maximize the sum of scores over all assignments.

        coverage and best_slot count each person once; total_attendance and
        total_preference count every attendance. In each pair, the first counts
        people and the second weights them by score. With binary 0/4 scores,
        best_slot is 4x coverage and total_preference is 4x total_attendance.

        The default order is coverage, best slot, minimum fulfillment,
        activities, total attendance, then total preference.

    return_all_solutions : bool
        If true, return every schedule tied at the best objective values. If
        false, return a list with one optimal schedule.

    prefer_spread : bool
        When returning all optimal schedules, keep only those with the fewest
        attendees repeated across consecutive selected activities. Slots are
        ordered by name, so names should sort chronologically.

    Returns
    -------
    list[dict[str, Any]]
        List of schedules containing participants, participation counts, and
        objective metrics. Keys ending in `_ideal` give, for comparison, the
        best value of a metric if every person got their favorite slots up to
        their capacity, ignoring all other constraints.

    Raises
    ------
    ValueError
        If the inputs are invalid.
    RuntimeError
        If the solver fails to prove an objective optimal. Valid inputs always
        have a schedule (possibly with no activities), so this indicates a bug
        or solver failure, not an impossible schedule.
    """
    slot_scores = _normalize_availability(availability)
    _validate_inputs(capacity, slot_scores, min_people, groups, objective_order)

    people = sorted(capacity)
    slots = sorted(slot_scores)

    # Eligible (person, slot) pairs and their scores. A score of zero, an
    # omitted person, or zero capacity means the person can never attend.
    score = {
        (p, s): slot_scores[s].get(p, 0)
        for p in people
        for s in slots
        if slot_scores[s].get(p, 0) > 0 and capacity[p] > 0
    }
    slots_of = {p: [s for s in slots if (p, s) in score] for p in people}
    people_of = {s: [p for p in people if (p, s) in score] for s in slots}

    model = cp_model.CpModel()

    # ==========================================================
    # Decision variables
    # ==========================================================

    selected = {s: model.new_bool_var(f"selected_{s}") for s in slots}
    assigned = {(p, s): model.new_bool_var(f"assigned_{p}_{s}") for p, s in score}

    # ==========================================================
    # Constraints
    # ==========================================================

    # A person can only attend a selected activity.
    for (_, s), is_assigned in assigned.items():
        model.add(is_assigned <= selected[s])

    # Every selected activity needs at least min_people attendees.
    for s in slots:
        attendees = sum(assigned[(p, s)] for p in people_of[s])
        model.add(attendees >= min_people * selected[s])

    # Nobody attends more activities than their capacity.
    attendance = {p: sum(assigned[(p, s)] for s in slots_of[p]) for p in people}
    for p in people:
        model.add(attendance[p] <= capacity[p])

    # At most one activity per group.
    for group in groups or ():
        model.add(sum(selected[s] for s in group) <= 1)

    # ==========================================================
    # Per-person quantities used by the objectives
    # ==========================================================

    # Attends at least once.
    covered = {p: model.new_bool_var(f"covered_{p}") for p in people}
    for p in people:
        model.add_max_equality(
            covered[p], [0] + [assigned[(p, s)] for s in slots_of[p]]
        )

    # Best score among the person's assignments (0 if unassigned).
    best_score = {
        p: model.new_int_var(0, _MAX_AVAILABILITY_SCORE, f"best_score_{p}")
        for p in people
    }
    for p in people:
        model.add_max_equality(
            best_score[p],
            [0] + [score[(p, s)] * assigned[(p, s)] for s in slots_of[p]],
        )

    # Lowest attendance/capacity ratio among covered people. CP-SAT only
    # supports integers, so the ratio is stored multiplied by `scale`. Using
    # the LCM of all capacities makes every possible ratio attendance/capacity
    # a whole number of 1/scale units, so no ratio is rounded and two different
    # ratios never look equal. A fixed scale like 1000 would round ratios down
    # and could tie distinct ones once capacities get large.
    # Only bounded from above, so maximizing makes it exact.
    scale = math.lcm(*(c for c in capacity.values() if c > 0))
    min_fulfillment = model.new_int_var(0, scale, "min_fulfillment")
    for p in people:
        model.add(
            capacity[p] * min_fulfillment <= scale * attendance[p]
        ).only_enforce_if(covered[p])
    model.add(min_fulfillment <= scale * sum(covered.values()))
    if "minimum_fulfillment" not in objective_order:
        # Otherwise it stays free, and enumeration would repeat each schedule
        # once per possible value.
        model.add(min_fulfillment == 0)

    # ==========================================================
    # Objectives, optimized lexicographically in objective_order
    # ==========================================================

    objectives = {
        "coverage": sum(covered.values()),
        "best_slot": sum(best_score.values()),
        "minimum_fulfillment": min_fulfillment,
        "activities": sum(selected.values()),
        "total_attendance": sum(assigned.values()),
        "total_preference": sum(score[key] * assigned[key] for key in assigned),
    }

    solver = cp_model.CpSolver()
    for objective_name in objective_order:
        objective = objectives[objective_name]
        model.maximize(objective)
        status = solver.solve(model)
        if status != cp_model.OPTIMAL:
            # Selecting nothing is always feasible, so this signals a bug or
            # solver problem rather than an impossible schedule.
            raise RuntimeError(
                f"solver returned {solver.status_name(status)} while optimizing "
                f"{objective_name!r}"
            )
        # Keep this optimum fixed while optimizing the next objectives.
        model.add(objective == solver.value(objective))

    # ==========================================================
    # Results
    # ==========================================================

    def extract_participants(value: Callable[..., int]) -> dict[str, list[str]]:
        return {
            s: [p for p in people_of[s] if value(assigned[(p, s)])]
            for s in slots
            if value(selected[s])
        }

    if not return_all_solutions:
        participants = extract_participants(solver.value)
        return [_summarize(participants, capacity, slot_scores)]

    schedules = [
        _summarize(participants, capacity, slot_scores)
        for participants in _enumerate_solutions(model, extract_participants)
    ]
    if prefer_spread:
        schedules = _filter_prefer_spread(schedules)
    return schedules
