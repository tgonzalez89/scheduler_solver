import unittest
from typing import Any, Literal, cast

from scheduler_solver import solve_schedule


class TestObjectives(unittest.TestCase):
    def solve_for_objective(
        self,
        objective: Literal[
            "coverage",
            "minimum_fulfillment",
            "activities",
            "total_assignments",
        ],
        availability: dict[str, frozenset[str]],
        capacity: dict[str, int],
        min_people: int,
    ):
        results = solve_schedule(
            availability=availability,
            capacity=capacity,
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
            availability={
                "slot 1": frozenset({"A", "B", "C", "E"}),
                "slot 2": frozenset({"D"}),
            },
            capacity={"A": 3, "B": 3, "C": 2, "D": 1, "E": 3},
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
            availability={
                "slot 1": frozenset({"A", "B", "C"}),
                "slot 2": frozenset({"A", "B"}),
            },
            capacity={"A": 2, "B": 2, "C": 2},
            min_people=2,
        )

        self.assertEqual(
            result["participants"],
            {"slot 1": ["A", "B"], "slot 2": ["A", "B"]},
        )
        self.assertEqual(result["minimum_fulfillment"], 1.0)

    def test_maximize_activities(self):
        result = self.solve_for_objective(
            "activities",
            availability={
                "slot 1": frozenset({"A", "C", "D", "E"}),
                "slot 2": frozenset({"A", "B", "C"}),
            },
            capacity={"A": 1, "B": 2, "C": 2, "D": 2, "E": 1},
            min_people=3,
        )

        self.assertEqual(
            result["participants"],
            {"slot 1": ["C", "D", "E"], "slot 2": ["A", "B", "C"]},
        )
        self.assertEqual(result["activity_count"], 2)

    def test_maximize_total_assignments(self):
        result = self.solve_for_objective(
            "total_assignments",
            availability={
                "slot 1": frozenset({"B", "D", "E"}),
                "slot 2": frozenset({"A", "D"}),
            },
            capacity={"A": 2, "B": 1, "C": 2, "D": 1, "E": 3},
            min_people=2,
        )

        self.assertEqual(
            result["participants"],
            {"slot 1": ["B", "E"], "slot 2": ["A", "D"]},
        )
        self.assertEqual(result["total_assignments"], 4)


class TestInputValidation(unittest.TestCase):
    def solve(
        self,
        availability: dict[str, frozenset[str]] | None = None,
        capacity: dict[str, int] | None = None,
        min_people: int | None = None,
        groups: frozenset[frozenset[str]] | None = None,
        objective_order: tuple[
            Literal[
                "coverage",
                "minimum_fulfillment",
                "activities",
                "total_assignments",
            ],
            ...,
        ] = ("activities",),
    ):
        return solve_schedule(
            availability=(
                {
                    "slot 1": frozenset({"A"}),
                    "slot 2": frozenset({"A"}),
                }
                if availability is None
                else availability
            ),
            capacity={"A": 2} if capacity is None else capacity,
            min_people=1 if min_people is None else min_people,
            groups=groups,
            objective_order=objective_order,
            return_all_solutions=False,
        )

    def test_availability_people_must_exist_in_capacity(self):
        with self.assertRaisesRegex(ValueError, "missing from capacity"):
            self.solve(availability={"slot 1": frozenset({"B"})})

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
                        "slot 1": frozenset({"A"}),
                        "slot 2": frozenset({"A"}),
                        "slot 3": frozenset({"A"}),
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
                        "slot 1": frozenset({"A"}),
                        "slot 2": frozenset({"A"}),
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
            self.solve(availability={"slot 1": frozenset({""})})

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.solve(availability={" ": frozenset({"A"})})


if __name__ == "__main__":
    unittest.main()
