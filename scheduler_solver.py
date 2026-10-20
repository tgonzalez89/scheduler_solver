from collections.abc import Callable
from functools import partial
from pprint import pprint
from typing import Any, Literal

from ortools.sat.python import cp_model

# ==========================================================
# Constants
# ==========================================================


_VERBOSE = False
_FULLFILLMENT_SCALE = 1000


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
    availability: dict[str, frozenset[str]], capacity: dict[str, int]
) -> None:
    for slot, available_people in availability.items():
        if not slot.strip():
            raise ValueError("availability keys must not be empty")
        if any(not person.strip() for person in available_people):
            raise ValueError(
                f"availability for slot {slot!r} contains an empty person name"
            )

    unknown_people = {
        person
        for available_people in availability.values()
        for person in available_people
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
    availability: dict[str, frozenset[str]], groups: frozenset[frozenset[str]] | None
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
    objective_order: tuple[
        Literal["coverage", "minimum_fulfillment", "activities", "total_assignments"],
        ...,
    ],
) -> None:
    valid_objectives = {
        "coverage",
        "minimum_fulfillment",
        "activities",
        "total_assignments",
    }
    if (
        not objective_order
        or len(objective_order) != len(set(objective_order))
        or not set(objective_order).issubset(valid_objectives)
    ):
        raise ValueError(
            "objective_order must contain one or more unique objectives from "
            "'coverage', 'minimum_fulfillment', 'activities', and "
            "'total_assignments'"
        )


def _validate_inputs(
    availability: dict[str, frozenset[str]],
    capacity: dict[str, int],
    min_people: int,
    groups: frozenset[frozenset[str]] | None,
    objective_order: tuple[
        Literal["coverage", "minimum_fulfillment", "activities", "total_assignments"],
        ...,
    ],
) -> None:
    _validate_capacity(capacity)
    _validate_availability(availability, capacity)
    _validate_min_people(min_people, capacity)
    _validate_groups(availability, groups)
    _validate_objective_order(objective_order)


# ==========================================================
# Result Collecting Functions
# ==========================================================


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


def _get_all_solutions(
    model: cp_model.CpModel,
    make_result: Callable[..., dict[str, Any]],
    prefer_spread: bool,
) -> list[dict[str, Any]]:
    model.clear_objective()

    class OptimalScheduleCollector(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.schedules = []
            self.seen = set()

        def OnSolutionCallback(self):
            result = make_result(self.value)
            key = tuple(
                (slot, tuple(result["participants"][slot]))
                for slot in result["activities"]
            )
            if key not in self.seen:
                self.seen.add(key)
                self.schedules.append(result)

    collector = OptimalScheduleCollector()
    enumerator = cp_model.CpSolver()
    enumerator.parameters.enumerate_all_solutions = True
    enumerator.parameters.num_search_workers = 1
    enumerator.solve(model, collector)

    if prefer_spread:
        return _filter_prefer_spread(collector.schedules)

    return collector.schedules


def _make_result(
    make_result: Callable[..., int],
    *,
    slots: list[str],
    people: list[str],
    activity_selected: dict[str, Any],
    person_assigned: dict[tuple[str, str], Any],
    participant_count: dict[str, Any],
    person_covered: dict[str, Any],
    min_fulfillment: Any,
    total_assignments: Any,
) -> dict[str, Any]:
    selected_slots = [slot for slot in slots if make_result(activity_selected[slot])]
    participants = {
        slot: [
            person for person in people if make_result(person_assigned[(person, slot)])
        ]
        for slot in selected_slots
    }
    return {
        "activities": selected_slots,
        "participants": participants,
        "participation_count": {
            person: make_result(participant_count[person]) for person in people
        },
        "activity_count": len(selected_slots),
        "coverage_count": sum(make_result(person_covered[person]) for person in people),
        "minimum_fulfillment": make_result(min_fulfillment) / _FULLFILLMENT_SCALE,
        "total_assignments": make_result(total_assignments),
    }


# ==========================================================
# Main Solver Function
# ==========================================================


def solve_schedule(
    availability: dict[str, frozenset[str]],
    capacity: dict[str, int],
    min_people: int,
    groups: frozenset[frozenset[str]] | None = None,
    objective_order: tuple[
        Literal["coverage", "minimum_fulfillment", "activities", "total_assignments"],
        ...,
    ] = ("coverage", "minimum_fulfillment", "activities", "total_assignments"),
    return_all_solutions: bool = True,
    prefer_spread: bool = True,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Optimize activity schedules subject to availability and capacity.

    Parameters
    ----------
    availability : dict[str, frozenset[str]]
        Maps each slot name to an immutable set of people available for it.
        Every person named here must also appear in `capacity`.

        Example:
            {
                "09-06 Fri": {"Mario", "Ana"},
                "09-07 Sat": {"Mario", "Carlos"},
                "09-08 Sun": {"Carlos", "Ana"},
            }

    capacity : dict[str, int]
        Maps each person to their maximum number of activities. This is the
        authoritative list of valid people; values must be non-negative.

        Example:
            {"Mario": 2, "Carlos": 3, "Ana": 1}

    min_people : int
        Minimum number of attendees required for every selected activity. Must
        be positive and no larger than the number of people with positive
        capacity.

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

        * "coverage": maximize the number of unique participants.
        * "minimum_fulfillment": maximize the lowest assigned/capacity ratio
          among covered participants.
        * "activities": maximize the number of selected activities.
        * "total_assignments": maximize the number of participant/activity
          assignments.

        The default order is coverage, minimum fulfillment, activities, then
        total assignments.

    return_all_solutions : bool
        If true, return every schedule tied at the best objective values. If
        false, return one schedule.

    prefer_spread : bool
        When returning all optimal schedules, keep only those with the fewest
        attendees repeated across consecutive selected activities.

    Returns
    -------
    list[dict[str, Any]]
        List of schedules containing participants, participation counts, and
        objective metrics.
    """
    _validate_inputs(availability, capacity, min_people, groups, objective_order)
    groups = groups or frozenset()

    model = cp_model.CpModel()

    people = sorted(capacity.keys())
    slots = sorted(availability.keys())

    # ==========================================================
    # Variables
    # ==========================================================

    # Activity selected on slot
    activity_selected = {s: model.new_bool_var(f"activity_{s}") for s in slots}

    # Participation assignment
    person_assigned = {}

    for p in people:
        for s in slots:
            person_assigned[(p, s)] = model.new_bool_var(f"assign_{p}_{s}")

    # Participates at least once
    person_covered = {p: model.new_bool_var(f"covered_{p}") for p in people}

    # Participation count per person
    participation_count = {}

    for p in people:
        participation_count[p] = model.new_int_var(0, capacity[p], f"count_{p}")

    # Minimum fulfillment
    min_fulfillment = model.new_int_var(0, _FULLFILLMENT_SCALE, "min_fulfillment")

    # ==========================================================
    # Availability constraints
    # ==========================================================

    for p in people:
        for s in slots:
            # unavailable => cannot be assigned
            if p not in availability[s]:
                model.add(person_assigned[(p, s)] == 0)

            # cannot participate if activity doesn't exist
            model.add(person_assigned[(p, s)] <= activity_selected[s])

    # ==========================================================
    # Participation count and capacity
    # ==========================================================

    for p in people:
        participation_sum = sum(person_assigned[(p, s)] for s in slots)

        model.add(participation_count[p] == participation_sum)

        model.add(participation_sum <= capacity[p])

    # ==========================================================
    # Coverage definition
    # ==========================================================

    M = len(slots)

    for p in people:
        model.add(participation_count[p] >= person_covered[p])
        model.add(participation_count[p] <= M * person_covered[p])

    # ==========================================================
    # Minimum attendance
    # ==========================================================

    for s in slots:
        attendees = sum(person_assigned[(p, s)] for p in people)

        model.add(attendees >= min_people * activity_selected[s])

    # ==========================================================
    # One activity per group
    # ==========================================================

    if groups:
        for group in groups:
            valid_slots = [s for s in group if s in activity_selected]

            if valid_slots:
                model.add(sum(activity_selected[s] for s in valid_slots) <= 1)

    # ==========================================================
    # Fulfillment constraints: maximize the lowest assigned/capacity ratio.
    # ==========================================================

    for p in people:
        if capacity[p] > 0:
            model.add(
                capacity[p] * min_fulfillment
                <= _FULLFILLMENT_SCALE * participation_count[p]
            ).only_enforce_if(person_covered[p])

    model.add(
        min_fulfillment <= _FULLFILLMENT_SCALE * sum(person_covered[p] for p in people)
    )

    # ==========================================================
    # Objective helpers
    # ==========================================================

    activity_count = sum(activity_selected[s] for s in slots)

    coverage_count = sum(person_covered[p] for p in people)

    total_assignments = sum(person_assigned[(p, s)] for p in people for s in slots)

    # ==========================================================
    # Objective order
    # ==========================================================

    objective_terms = {
        "coverage": coverage_count,
        "minimum_fulfillment": min_fulfillment,
        "activities": activity_count,
        "total_assignments": total_assignments,
    }

    # ==========================================================
    # Solve
    # ==========================================================

    solver = cp_model.CpSolver()
    for objective_name in objective_order:
        objective_term = objective_terms[objective_name]
        model.maximize(objective_term)
        status = solver.solve(model)
        if status != cp_model.OPTIMAL:
            return []
        model.add(objective_term == solver.value(objective_term))

    # ==========================================================
    # Return the final result
    # ==========================================================

    make_result = partial(
        _make_result,
        slots=slots,
        people=people,
        activity_selected=activity_selected,
        person_assigned=person_assigned,
        participant_count=participation_count,
        person_covered=person_covered,
        min_fulfillment=min_fulfillment,
        total_assignments=total_assignments,
    )

    if return_all_solutions:
        return _get_all_solutions(model, make_result, prefer_spread)

    return [make_result(solver.value)]


# ==========================================================
# Main
# ==========================================================


if __name__ == "__main__":
    capacity = {
        "Tomas": 3,
        "Victor": 3,
        "Daniel": 2,
        "Hernan": 2,
        "Jose": 2,
        "Nacho": 2,
        "Matteo": 1,
        "Daniel2": 1,
    }

    availability = {
        "09-06 Fri": frozenset({"Tomas", "Hernan", "Nacho", "Victor"}),
        "09-07 Sat": frozenset({"Tomas", "Hernan", "Nacho", "Daniel"}),
        "09-08 Sun": frozenset({"Tomas", "Hernan"}),
        "09-13 Fri": frozenset({"Tomas", "Hernan", "Nacho", "Victor", "Matteo"}),
        "09-14 Sat": frozenset({"Tomas", "Nacho"}),
        "09-15 Sun": frozenset({"Tomas", "Hernan"}),
        "09-20 Fri": frozenset({"Tomas", "Hernan", "Nacho", "Victor"}),
        "09-21 Sat": frozenset({"Tomas", "Hernan", "Nacho", "Daniel"}),
        "09-22 Sun": frozenset({"Tomas", "Hernan"}),
        "09-27 Fri": frozenset({"Tomas", "Hernan", "Victor"}),
        "09-28 Sat": frozenset({"Tomas", "Hernan"}),
        "09-29 Sun": frozenset({"Tomas", "Hernan"}),
    }

    groups = frozenset(
        frozenset(group) for group in zip(*[iter(availability.keys())] * 3)
    )

    result = solve_schedule(
        availability=availability,
        capacity=capacity,
        min_people=4,
        groups=groups,
        objective_order=(
            "coverage",
            "minimum_fulfillment",
            "activities",
            "total_assignments",
        ),
        return_all_solutions=True,
        prefer_spread=True,
    )

    if _VERBOSE:
        pprint(result)
    else:
        schedules = result if isinstance(result, list) else [result]
        if schedules:
            print()
        for schedule in schedules:
            print(
                "\n".join(
                    f"{slot}: {', '.join(p for p in sorted(participants))}"
                    for slot, participants in schedule["participants"].items()
                )
            )
            print()
