import unittest
from typing import Any, cast
from unittest.mock import patch

from ortools.sat.python import cp_model

from scheduler_solver import AvailabilityInput, ObjectiveName, solve_schedule


class TestObjectives(unittest.TestCase):
    def solve_for_objective(
        self,
        objective: ObjectiveName,
        capacity: dict[str, int],
        availability: AvailabilityInput,
        min_people: int,
    ):
        results = solve_schedule(
            capacity=capacity,
            availability=availability,
            min_people=min_people,
            objective_order=(objective,),
            return_all_solutions=True,
            prefer_spread=False,
        )
        if not isinstance(results, list):
            self.fail("expected all-optimal mode to return a list")
        self.assertEqual(len(results), 1, "expected exactly one optimal schedule")
        return results[0]

    def test_maximize_coverage(self):
        result = self.solve_for_objective(
            "coverage",
            capacity={"A": 3, "B": 3, "C": 2, "D": 1, "E": 3},
            availability={
                "slot 1": {"A": 4, "B": 4, "C": 4, "E": 4},
                "slot 2": {"D": 4},
            },
            min_people=1,
        )

        self.assertEqual(
            result["participants"],
            {"slot 1": ["A", "B", "C", "E"], "slot 2": ["D"]},
        )
        self.assertEqual(result["coverage_count"], 5)

    def test_maximize_minimum_fulfillment(self):
        result = self.solve_for_objective(
            "minimum_fulfillment",
            capacity={"A": 2, "B": 2, "C": 2},
            availability={
                "slot 1": {"A": 4, "B": 4, "C": 4},
                "slot 2": {"A": 4, "B": 4},
            },
            min_people=2,
        )

        self.assertEqual(
            result["participants"],
            {"slot 1": ["A", "B"], "slot 2": ["A", "B"]},
        )
        self.assertEqual(result["minimum_fulfillment"], 100)

    def test_metrics_are_reported_for_objectives_not_optimized(self):
        result = self.solve_for_objective(
            "total_attendance",
            capacity={"A": 2, "B": 2},
            availability={"slot 1": {"A": 4, "B": 3}, "slot 2": {"A": 4}},
            min_people=1,
        )

        self.assertEqual(
            result["participants"], {"slot 1": ["A", "B"], "slot 2": ["A"]}
        )
        self.assertEqual(result["minimum_fulfillment"], 50)
        self.assertEqual(result["coverage_count"], 2)
        self.assertEqual(result["best_slot_score"], 7)
        self.assertEqual(result["total_preference"], 11)

    def test_ideal_metrics_assume_everyone_gets_favorite_slots(self):
        [result] = solve_schedule(
            capacity={"A": 2, "B": 1, "C": 0, "D": 1},
            availability={
                "slot 1": {"A": 2, "B": 4, "C": 4},
                "slot 2": {"A": 3, "B": 1},
                "slot 3": {"A": 4},
            },
            min_people=2,
            return_all_solutions=False,
        )

        # A: best two of 4, 3, 2. B: best one of 4, 1. C has no capacity and D
        # no positive score, so neither counts.
        self.assertEqual(result["coverage_ideal"], 2)
        self.assertEqual(result["best_slot_ideal"], 8)
        self.assertEqual(result["total_attendance_ideal"], 3)
        self.assertEqual(result["total_preference_ideal"], 11)
        # min_people=2 forces A and B together, so the real schedule falls short.
        self.assertLess(result["total_preference"], 11)

    def test_minimum_fulfillment_distinguishes_close_ratios(self):
        # 1/39 and 1/40 differ by less than 1/1000, so a fixed scale of 1000
        # would round both down to 25 and report a tie.
        result = solve_schedule(
            capacity={"P": 39, "Q": 40},
            availability={"slot P": {"P": 4}, "slot Q": {"Q": 4}},
            min_people=1,
            groups=frozenset({frozenset({"slot P", "slot Q"})}),
            objective_order=("minimum_fulfillment",),
            prefer_spread=False,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["participants"], {"slot P": ["P"]})

    def test_maximize_activities(self):
        result = self.solve_for_objective(
            "activities",
            capacity={"A": 1, "B": 2, "C": 2, "D": 2, "E": 1},
            availability={
                "slot 1": {"A": 4, "C": 4, "D": 4, "E": 4},
                "slot 2": {"A": 4, "B": 4, "C": 4},
            },
            min_people=3,
        )

        self.assertEqual(
            result["participants"],
            {"slot 1": ["C", "D", "E"], "slot 2": ["A", "B", "C"]},
        )
        self.assertEqual(result["activity_count"], 2)

    def test_maximize_total_attendance(self):
        result = self.solve_for_objective(
            "total_attendance",
            capacity={"A": 2, "B": 1, "C": 2, "D": 1, "E": 3},
            availability={
                "slot 1": {"B": 4, "D": 4, "E": 4},
                "slot 2": {"A": 4, "D": 4},
            },
            min_people=2,
        )

        self.assertEqual(
            result["participants"],
            {"slot 1": ["B", "E"], "slot 2": ["A", "D"]},
        )
        self.assertEqual(result["total_attendance"], 4)

    def test_total_attendance_keeps_seats_total_preference_would_trade_away(self):
        availability: AvailabilityInput = {
            "base": {"A": 4, "B": 4, "C": 4},
            "X": {"A": 4, "B": 4},
            "Y": {"A": 1, "B": 1, "C": 1},
        }
        expected: dict[ObjectiveName, str] = {
            "total_attendance": "Y",
            "total_preference": "X",
        }
        for objective, extra_slot in expected.items():
            with self.subTest(objective=objective):
                result = solve_schedule(
                    capacity={"A": 2, "B": 2, "C": 2},
                    availability=availability,
                    min_people=1,
                    groups=frozenset({frozenset({"X", "Y"})}),
                    objective_order=("coverage", objective),
                    return_all_solutions=True,
                    prefer_spread=False,
                )

                if not isinstance(result, list):
                    self.fail("expected all-optimal mode to return a list")
                self.assertEqual(len(result), 1)
                self.assertEqual(set(result[0]["activities"]), {"base", extra_slot})

    def test_maximize_total_preference_generalizes_total_attendance(self):
        result = self.solve_for_objective(
            "total_preference",
            capacity={"A": 2, "B": 1, "C": 2, "D": 1, "E": 3},
            availability={
                "slot 1": {"B": 4, "D": 4, "E": 4},
                "slot 2": {"A": 4, "D": 4},
            },
            min_people=2,
        )

        self.assertEqual(
            result["participants"],
            {"slot 1": ["B", "E"], "slot 2": ["A", "D"]},
        )
        self.assertEqual(result["total_preference"], 16)

    def test_best_slot_puts_people_in_their_best_slot(self):
        result = self.solve_for_objective(
            "best_slot",
            capacity={"Alex": 1, "Blair": 1, "Casey": 1, "Drew": 1, "Daniel": 1},
            availability={
                "Friday": {"Alex": 4, "Blair": 4, "Casey": 4, "Drew": 4, "Daniel": 2},
                "Saturday": {
                    "Alex": 4,
                    "Blair": 4,
                    "Casey": 4,
                    "Drew": 4,
                    "Daniel": 4,
                },
            },
            min_people=4,
        )

        self.assertEqual(
            result["participants"],
            {"Saturday": ["Alex", "Blair", "Casey", "Daniel", "Drew"]},
        )
        self.assertEqual(result["best_slot_score"], 20)

    def test_coverage_never_trades_people_for_better_slots(self):
        result = solve_schedule(
            capacity={"A": 1, "B": 1, "C": 1, "D": 1},
            availability={
                "one loves it": {"A": 4},
                "three only if needed": {"B": 1, "C": 1, "D": 1},
            },
            min_people=1,
            groups=frozenset({frozenset({"one loves it", "three only if needed"})}),
            return_all_solutions=True,
            prefer_spread=False,
        )

        if not isinstance(result, list):
            self.fail("expected all-optimal mode to return a list")
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0]["participants"], {"three only if needed": ["B", "C", "D"]}
        )
        self.assertEqual(result[0]["coverage_count"], 3)

    def test_best_slot_upgrades_people_where_total_preference_adds_attendance(self):
        availability: AvailabilityInput = {
            "P": {"A": 4, "B": 1},
            "Q": {"A": 4},
            "R": {"B": 4},
        }
        expected: dict[ObjectiveName, dict[str, list[str]]] = {
            "best_slot": {"P": ["A"], "R": ["B"]},
            "total_preference": {"P": ["A", "B"], "Q": ["A"]},
        }
        for objective, participants in expected.items():
            with self.subTest(objective=objective):
                result = solve_schedule(
                    capacity={"A": 2, "B": 1},
                    availability=availability,
                    min_people=1,
                    groups=frozenset({frozenset({"Q", "R"})}),
                    objective_order=("coverage", objective),
                    return_all_solutions=True,
                    prefer_spread=False,
                )

                if not isinstance(result, list):
                    self.fail("expected all-optimal mode to return a list")
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]["participants"], participants)

    def test_total_preference_picks_better_scored_slot_in_group(self):
        result = solve_schedule(
            capacity={"A": 2, "B": 2, "C": 2},
            availability={
                "base": {"A": 4, "B": 4, "C": 4},
                "maybe 1": {"A": 4, "B": 4, "C": 1},
                "maybe 3": {"A": 4, "B": 4, "C": 3},
            },
            min_people=2,
            groups=frozenset({frozenset({"maybe 1", "maybe 3"})}),
            objective_order=("coverage", "total_preference"),
            return_all_solutions=True,
            prefer_spread=False,
        )

        if not isinstance(result, list):
            self.fail("expected all-optimal mode to return a list")
        self.assertEqual(len(result), 1)
        self.assertEqual(set(result[0]["activities"]), {"base", "maybe 3"})
        self.assertEqual(result[0]["total_preference"], 23)

    def test_total_preference_rewards_actual_attendance(self):
        result = self.solve_for_objective(
            "total_preference",
            capacity={"A": 1, "B": 1, "C": 1},
            availability={"slot": {"A": 4, "B": 3, "C": 1}},
            min_people=1,
        )

        self.assertEqual(result["participants"], {"slot": ["A", "B", "C"]})
        self.assertEqual(result["total_preference"], 8)

    def test_zero_score_and_omitted_people_are_unavailable(self):
        result = solve_schedule(
            capacity={"A": 1, "B": 1, "C": 1},
            availability={"slot": {"A": 0, "B": 4}},
            min_people=1,
            return_all_solutions=True,
            prefer_spread=False,
        )

        if not isinstance(result, list):
            self.fail("expected all-optimal mode to return a list")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["participants"], {"slot": ["B"]})
        self.assertEqual(result[0]["coverage_count"], 1)


class TestAvailabilityFormats(unittest.TestCase):
    def solve(self, availability: AvailabilityInput):
        return solve_schedule(
            capacity={"A": 1, "B": 1, "C": 1},
            availability=availability,
            min_people=1,
            prefer_spread=False,
        )

    def test_frozenset_means_listed_people_score_four_and_others_zero(self):
        self.assertEqual(
            self.solve(
                {"slot 1": frozenset({"A", "B"}), "slot 2": frozenset({"C"})}
            ),
            self.solve({"slot 1": {"A": 4, "B": 4}, "slot 2": {"C": 4, "A": 0}}),
        )

    def test_frozensets_and_score_dicts_can_be_mixed(self):
        result = self.solve(
            {"slot 1": frozenset({"A"}), "slot 2": {"B": 2, "C": 4}}
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0]["participants"], {"slot 1": ["A"], "slot 2": ["B", "C"]}
        )
        self.assertEqual(result[0]["total_preference"], 10)


class TestSolverFailure(unittest.TestCase):
    def test_non_optimal_status_raises_with_objective_name(self):
        with (
            patch.object(cp_model.CpSolver, "solve", return_value=cp_model.UNKNOWN),
            self.assertRaisesRegex(RuntimeError, "UNKNOWN.*'coverage'"),
        ):
            solve_schedule(
                capacity={"A": 1},
                availability={"slot": {"A": 4}},
                min_people=1,
                objective_order=("coverage",),
            )


class TestInputValidation(unittest.TestCase):
    def solve(
        self,
        capacity: dict[str, int] | None = None,
        availability: AvailabilityInput | None = None,
        min_people: int | None = None,
        groups: frozenset[frozenset[str]] | None = None,
        objective_order: tuple[
            ObjectiveName,
            ...,
        ] = ("activities",),
    ):
        return solve_schedule(
            capacity={"A": 2} if capacity is None else capacity,
            availability=(
                {
                    "slot 1": {"A": 4},
                    "slot 2": {"A": 4},
                }
                if availability is None
                else availability
            ),
            min_people=1 if min_people is None else min_people,
            groups=groups,
            objective_order=objective_order,
            return_all_solutions=False,
        )

    def test_availability_people_must_exist_in_capacity(self):
        with self.assertRaisesRegex(ValueError, "missing from capacity"):
            self.solve(availability={"slot 1": {"B": 4}})

    def test_availability_scores_must_be_between_zero_and_four(self):
        for score in (-1, 5):
            with (
                self.subTest(score=score),
                self.assertRaisesRegex(ValueError, "availability score"),
            ):
                self.solve(availability={"slot 1": {"A": score}})

    def test_group_slots_must_exist_in_availability(self):
        with self.assertRaisesRegex(ValueError, "missing from availability"):
            self.solve(groups=frozenset({frozenset({"slot 1", "missing slot"})}))

    def test_groups_must_be_valid_disjoint_sets(self):
        invalid_groups = (
            frozenset({frozenset({"slot 1"})}),
            frozenset({frozenset({"slot 1", ""})}),
            frozenset(
                {
                    frozenset({"slot 1", "slot 2"}),
                    frozenset({"slot 2", "slot 3"}),
                }
            ),
        )

        for groups in invalid_groups:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                self.solve(
                    availability={
                        "slot 1": {"A": 4},
                        "slot 2": {"A": 4},
                        "slot 3": {"A": 4},
                    },
                    groups=groups,
                )

    def test_capacity_values_must_be_nonnegative_integers(self):
        invalid_capacities = ({"A": -1}, {" ": 1})
        for capacity in invalid_capacities:
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                self.solve(capacity=capacity)

    def test_min_people_must_be_possible_and_positive(self):
        for min_people in (0, -1, 2):
            with self.subTest(min_people=min_people), self.assertRaises(ValueError):
                self.solve(min_people=min_people)

    def test_none_and_empty_groups_are_allowed(self):
        for groups in (
            None,
            frozenset(),
            frozenset({frozenset({"slot 1", "slot 2"})}),
        ):
            with self.subTest(groups=groups):
                result = self.solve(
                    availability={
                        "slot 1": {"A": 4},
                        "slot 2": {"A": 4},
                    },
                    groups=groups,
                )
                self.assertIsInstance(result, list)
                self.assertTrue(result)

    def test_objective_order_must_be_nonempty_unique_and_supported(self):
        invalid_orders = (
            (),
            ("coverage", "coverage"),
            cast(Any, ("unsupported_objective",)),
        )
        for objective_order in invalid_orders:
            with (
                self.subTest(objective_order=objective_order),
                self.assertRaisesRegex(ValueError, "objective_order"),
            ):
                self.solve(objective_order=objective_order)

    def test_blank_names_and_slots_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty person name"):
            self.solve(availability={"slot 1": {"": 4}})

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.solve(availability={" ": {"A": 4}})


if __name__ == "__main__":
    unittest.main()
