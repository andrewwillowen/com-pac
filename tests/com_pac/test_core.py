"""
Unit tests for functions in core.py
"""

import pytest

from com_pac.core import _non_negative_int, build_parser, read_args

import argparse


class Test_non_negative_int:
    """Test the _non_negative_int argparse type validator."""

    def test_zero_is_valid(self):
        assert _non_negative_int("0") == 0

    def test_positive_integer_is_valid(self):
        assert _non_negative_int("6") == 6

    def test_large_positive_integer_is_valid(self):
        assert _non_negative_int("100") == 100

    def test_negative_integer_raises(self):
        with pytest.raises(
            argparse.ArgumentTypeError, match="not a non-negative integer"
        ):
            _non_negative_int("-1")

    def test_float_string_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="not a valid integer"):
            _non_negative_int("3.5")

    def test_non_numeric_string_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="not a valid integer"):
            _non_negative_int("abc")


class Test_build_parser:
    """Test build_parser() produces a correctly configured ArgumentParser."""

    def test_returns_argument_parser(self):
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_input_file_required(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([])
        assert exc_info.value.code != 0

    def test_input_file_parsed_as_path(self):
        from pathlib import Path

        parser = build_parser()
        args = parser.parse_args(["some_file.txt"])
        assert args.input_file == Path("some_file.txt")

    def test_decimals_default_is_six(self):
        parser = build_parser()
        args = parser.parse_args(["some_file.txt"])
        assert args.num_of_decimals == 6

    def test_decimals_option_sets_value(self):
        parser = build_parser()
        args = parser.parse_args(["some_file.txt", "--decimals", "3"])
        assert args.num_of_decimals == 3

    def test_decimals_zero_is_valid(self):
        parser = build_parser()
        args = parser.parse_args(["some_file.txt", "--decimals", "0"])
        assert args.num_of_decimals == 0

    def test_decimals_negative_exits_with_error(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["some_file.txt", "--decimals", "-1"])
        assert exc_info.value.code != 0

    def test_decimals_non_integer_exits_with_error(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["some_file.txt", "--decimals", "abc"])
        assert exc_info.value.code != 0

    def test_output_dir_default_is_none(self):
        parser = build_parser()
        args = parser.parse_args(["some_file.txt"])
        assert args.output_dir is None

    def test_output_dir_option_sets_value(self):
        from pathlib import Path

        parser = build_parser()
        args = parser.parse_args(["some_file.txt", "--output-dir", "/tmp/out"])
        assert args.output_dir == Path("/tmp/out")

    def test_version_flag_exits(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_theta_flag_defaults_to_false(self):
        parser = build_parser()
        args = parser.parse_args(["some_file.txt"])
        assert args.theta is False

    def test_theta_flag_sets_true(self):
        parser = build_parser()
        args = parser.parse_args(["some_file.txt", "--theta"])
        assert args.theta is True


class Test_read_args:
    """Test read_args() using monkeypatched sys.argv."""

    def test_returns_path_and_decimals(self, monkeypatch, tmp_path):
        from pathlib import Path

        monkeypatch.setattr("sys.argv", ["com-pac", str(tmp_path / "input.txt")])
        path, decimals, theta = read_args()
        assert path == Path(str(tmp_path / "input.txt"))
        assert decimals == 6
        assert theta is False

    def test_decimals_argument_is_respected(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "sys.argv", ["com-pac", str(tmp_path / "input.txt"), "--decimals", "3"]
        )
        _, decimals, _ = read_args()
        assert decimals == 3

    def test_theta_flag_is_returned(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "sys.argv", ["com-pac", str(tmp_path / "input.txt"), "--theta"]
        )
        _, _, theta = read_args()
        assert theta is True

    def test_output_dir_raises_not_implemented(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "sys.argv",
            ["com-pac", str(tmp_path / "input.txt"), "--output-dir", str(tmp_path)],
        )
        with pytest.raises(NotImplementedError):
            read_args()

    def test_no_args_exits(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["com-pac"])
        with pytest.raises(SystemExit) as exc_info:
            read_args()
        assert exc_info.value.code != 0
