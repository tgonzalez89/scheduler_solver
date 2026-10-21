"""Command-line interface: read availability from a CSV and print schedules.

Example:
    python cli.py example.csv --min-people 4
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scheduler_solver import DEFAULT_OBJECTIVE_ORDER, AvailabilityInput, solve_schedule

# ==========================================================
# CSV Input
# ==========================================================


@dataclass(frozen=True)
class ScheduleData:
    capacity: dict[str, int]
    availability: AvailabilityInput
    groups: frozenset[frozenset[str]]


def _parse_int(cell: str, line: int, column: str, blank: int | None) -> int:
    """Parse a cell as a whole number; `blank` is the value for an empty cell."""
    if not cell and blank is not None:
        return blank
    try:
        return int(cell)
    except ValueError:
        raise ValueError(
            f"line {line}, column {column!r}: expected a whole number, got {cell!r}"
        ) from None


def parse_csv(text: str) -> ScheduleData:
    """Parse availability CSV text.

    The first row is a header: `slot`, an optional `group` column, then one
    column per person. A row named `capacity` gives each person's capacity;
    every other row is a slot with one score per person. Blank scores mean 0.
    Slots sharing a group name form a group; a blank group means no group.
    Cells may be separated by commas, semicolons, or tabs.

    Only problems that would be lost when converting to dicts are checked
    here; value ranges and the like are left to `solve_schedule`.
    """
    # The header has one delimiter per column, so the most frequent candidate
    # there is the delimiter. Spreadsheets in some locales save with ';'.
    header_line = next((line for line in text.splitlines() if line.strip()), "")
    delimiter = max(",;\t", key=header_line.count)

    # Strip cells and drop trailing empty ones, which spreadsheets often add.
    rows: list[tuple[int, list[str]]] = []
    reader = csv.reader(text.splitlines(), delimiter=delimiter)
    for row in reader:
        cells = [cell.strip() for cell in row]
        while cells and not cells[-1]:
            cells.pop()
        if cells:
            rows.append((reader.line_num, cells))

    if not rows or rows[0][1][0].lower() != "slot":
        raise ValueError("the first row must be a header starting with 'slot'")
    header = rows[0][1]
    has_group = len(header) > 1 and header[1].lower() == "group"
    people = header[2:] if has_group else header[1:]
    first_person_column = len(header) - len(people)

    duplicate_people = sorted({p for p in people if people.count(p) > 1})
    if duplicate_people:
        raise ValueError(f"people appear more than once: {duplicate_people}")

    capacity: dict[str, int] | None = None
    availability: AvailabilityInput = {}
    group_slots: dict[str, list[str]] = {}
    for line, cells in rows[1:]:
        if len(cells) > len(header):
            raise ValueError(f"line {line}: more cells than header columns")
        cells += [""] * (len(header) - len(cells))
        name = cells[0]
        values = cells[first_person_column:]

        if name.lower() == "capacity":
            if capacity is not None:
                raise ValueError(f"line {line}: duplicate 'capacity' row")
            capacity = {
                person: _parse_int(cell, line, person, blank=None)
                for person, cell in zip(people, values)
            }
            continue

        if name in availability:
            raise ValueError(f"line {line}: duplicate slot {name!r}")
        availability[name] = {
            person: _parse_int(cell, line, person, blank=0)
            for person, cell in zip(people, values)
        }
        if has_group and cells[1]:
            group_slots.setdefault(cells[1], []).append(name)

    if capacity is None:
        raise ValueError("missing 'capacity' row")

    groups = frozenset(frozenset(slots) for slots in group_slots.values())
    return ScheduleData(capacity, availability, groups)


def load_csv(path: Path) -> ScheduleData:
    # utf-8-sig skips the byte order mark Excel adds when saving as UTF-8.
    return parse_csv(path.read_text(encoding="utf-8-sig"))


# ==========================================================
# Arguments
# ==========================================================


def _parse_objectives(text: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in text.split(","))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the best activity schedules for a group of people."
    )
    parser.add_argument(
        "csv_file", type=Path, help="CSV file with capacities and availability"
    )
    parser.add_argument(
        "--min-people",
        type=int,
        required=True,
        help="minimum number of attendees for every selected activity",
    )
    parser.add_argument(
        "--objectives",
        type=_parse_objectives,
        default=DEFAULT_OBJECTIVE_ORDER,
        help=(
            "comma-separated objectives in priority order "
            f"(default: {','.join(DEFAULT_OBJECTIVE_ORDER)})"
        ),
    )
    parser.add_argument(
        "--one",
        action="store_true",
        help="return a single optimal schedule instead of all tied ones",
    )
    parser.add_argument(
        "--no-spread",
        action="store_true",
        help="keep schedules that repeat attendees in consecutive activities",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the full results as JSON"
    )
    return parser.parse_args(argv)


# ==========================================================
# Output
# ==========================================================


def format_schedule(schedule: dict[str, Any]) -> str:
    lines = [
        f"Coverage: {schedule['coverage_count']}/{schedule['coverage_ideal']}",
        f"Best slot score: {schedule['best_slot_score']}/{schedule['best_slot_ideal']}",
        "",
    ]
    lines += [
        f"{slot}: {', '.join(sorted(participants))}"
        for slot, participants in schedule["participants"].items()
    ]
    return "\n".join(lines)


# ==========================================================
# Main
# ==========================================================


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        data = load_csv(args.csv_file)
        schedules = solve_schedule(
            data.capacity,
            data.availability,
            args.min_people,
            data.groups,
            args.objectives,
            return_all_solutions=not args.one,
            prefer_spread=not args.no_spread,
        )
    except (OSError, ValueError) as error:
        sys.exit(f"error: {error}")

    if args.json:
        print(json.dumps(schedules, indent=2, ensure_ascii=False))
        return
    for schedule in schedules:
        print()
        print(format_schedule(schedule))
    print()


if __name__ == "__main__":
    main()
