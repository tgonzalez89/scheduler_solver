import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from cli import ScheduleData, load_csv, main, parse_args, parse_csv
from scheduler_solver import DEFAULT_OBJECTIVE_ORDER

EXAMPLE_CSV = Path(__file__).with_name("example.csv")


class TestParseCsv(unittest.TestCase):
    def test_parses_capacity_scores_and_groups(self):
        data = parse_csv(
            "slot,group,A,B\n"
            "capacity,,2,1\n"
            "Fri,week 1,4,\n"
            "Sat,week 1,,3\n"
            "Sun,,1,0\n"
        )

        self.assertEqual(
            data,
            ScheduleData(
                capacity={"A": 2, "B": 1},
                availability={
                    "Fri": {"A": 4, "B": 0},
                    "Sat": {"A": 0, "B": 3},
                    "Sun": {"A": 1, "B": 0},
                },
                groups=frozenset({frozenset({"Fri", "Sat"})}),
            ),
        )

    def test_group_column_is_optional(self):
        data = parse_csv("slot,A\ncapacity,1\nFri,4\n")

        self.assertEqual(data.availability, {"Fri": {"A": 4}})
        self.assertEqual(data.groups, frozenset())

    def test_spaces_semicolons_empty_rows_and_trailing_cells_are_accepted(self):
        data = parse_csv("slot ; A ; B ;;\n\ncapacity ; 1 ; 2\nFri ; 4 ;  ;;\n")

        self.assertEqual(data.capacity, {"A": 1, "B": 2})
        self.assertEqual(data.availability, {"Fri": {"A": 4, "B": 0}})

    def test_out_of_range_values_are_left_to_the_solver(self):
        data = parse_csv("slot,A\ncapacity,-1\nFri,5\n")

        self.assertEqual(data.capacity, {"A": -1})
        self.assertEqual(data.availability, {"Fri": {"A": 5}})

    def test_structural_problems_are_rejected(self):
        cases = {
            "missing header": ("capacity,1\nFri,4\n", "header starting with 'slot'"),
            "missing capacity": ("slot,A\nFri,4\n", "missing 'capacity' row"),
            "duplicate capacity": (
                "slot,A\ncapacity,1\ncapacity,2\n",
                "line 3: duplicate 'capacity' row",
            ),
            "duplicate person": ("slot,A,A\ncapacity,1,1\n", "more than once"),
            "duplicate slot": (
                "slot,A\ncapacity,1\nFri,4\nFri,3\n",
                "line 4: duplicate slot 'Fri'",
            ),
            "extra cells": ("slot,A\ncapacity,1,2\n", "line 2: more cells"),
            "not a number": (
                "slot,A\ncapacity,1\nFri,yes\n",
                "line 3, column 'A': expected a whole number, got 'yes'",
            ),
            "blank capacity": (
                "slot,A,B\ncapacity,1,\n",
                "line 2, column 'B': expected a whole number, got ''",
            ),
        }
        for name, (text, message) in cases.items():
            with (
                self.subTest(name),
                self.assertRaisesRegex(ValueError, message),
            ):
                parse_csv(text)

    def test_tab_delimiter_is_detected(self):
        data = parse_csv("slot\tA\ncapacity\t1\nFri, 6 Sep\t4\n")

        self.assertEqual(data.availability, {"Fri, 6 Sep": {"A": 4}})

    def test_load_csv_skips_excel_byte_order_mark(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "data.csv")
            path.write_text("slot,A\ncapacity,1\nFri,4\n", encoding="utf-8-sig")

            self.assertEqual(load_csv(path).availability, {"Fri": {"A": 4}})


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        args = parse_args(["data.csv", "--min-people", "3"])

        self.assertEqual(args.csv_file, Path("data.csv"))
        self.assertEqual(args.min_people, 3)
        self.assertEqual(args.objectives, DEFAULT_OBJECTIVE_ORDER)
        self.assertFalse(args.one)
        self.assertFalse(args.no_spread)
        self.assertFalse(args.json)

    def test_objectives_are_split_on_commas(self):
        args = parse_args(
            ["data.csv", "--min-people", "3", "--objectives", "coverage, activities"]
        )

        self.assertEqual(args.objectives, ("coverage", "activities"))

    def test_min_people_is_required(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["data.csv"])


class TestMain(unittest.TestCase):
    def run_main(self, *argv: str) -> str:
        output = io.StringIO()
        with redirect_stdout(output):
            main([str(EXAMPLE_CSV), *argv])
        return output.getvalue()

    def test_prints_readable_schedules(self):
        output = self.run_main("--min-people", "4", "--one")

        self.assertIn("Coverage: 6/6", output)
        self.assertIn("Best slot score: 24/24", output)
        self.assertIn("09-13 Fri: Hernan, Matteo, Nacho, Tomas, Victor", output)

    def test_json_prints_full_results(self):
        schedules = json.loads(self.run_main("--min-people", "4", "--json"))

        self.assertTrue(schedules)
        self.assertEqual(schedules[0]["coverage_count"], 6)
        self.assertIn("participants", schedules[0])

    def test_input_errors_exit_with_message(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_main("--min-people", "99")

        self.assertIn("min_people", str(raised.exception.code))


if __name__ == "__main__":
    unittest.main()
