import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from parse_ae import parse_wait, over_2h


def test_new_format_strings():
    assert parse_wait("0 minute") == (0.0, "exact")
    assert parse_wait("less than 15 minutes") == (7.5, "lt")
    assert parse_wait("23 minutes") == (23.0, "exact")
    assert parse_wait("1 hour") == (60.0, "exact")
    assert parse_wait("3 hour") == (180.0, "exact")
    assert parse_wait("2.5 hours") == (150.0, "exact")
    assert parse_wait("6.5 hours") == (390.0, "exact")


def test_old_format_categories():
    assert parse_wait("Around 1 hour") == (60.0, "around")
    assert parse_wait("Over 1 hour") == (60.0, "over")
    assert parse_wait("Over 8 hours") == (480.0, "over")


def test_garbage():
    assert parse_wait(None) == (None, None)
    assert parse_wait("") == (None, None)
    assert parse_wait("N/A") == (None, None)


def test_over_2h_label():
    assert over_2h(*parse_wait("Over 2 hours")) == 1
    assert over_2h(*parse_wait("Over 1 hour")) == 0
    assert over_2h(*parse_wait("Around 1 hour")) == 0
    assert over_2h(*parse_wait("2 hours")) == 0      # exactly 120 is not > 2h
    assert over_2h(*parse_wait("2.5 hours")) == 1
    assert over_2h(*parse_wait("less than 15 minutes")) == 0
    assert over_2h(None, None) is None
