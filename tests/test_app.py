# Tests for the input handling and output of the web app (needs gradio installed)
import pytest

pytest.importorskip("gradio")

from app import parse_numbers, parse_irrep, parse_factors, compute, young_svg, DYNKIN, PARTITION, EXAMPLES
from sun_irrep_CG import IrrepSUN


@pytest.mark.parametrize("text, numbers", [
    ("1, 1", [1, 1]), ("(1 1)", [1, 1]), ("[2,1]", [2, 1]), ("  3 ", [3]), ("", []), ("0", []), ("()", []),
])
def test_parse_numbers(text, numbers):
    assert parse_numbers(text) == numbers


@pytest.mark.parametrize("text", ["a, 1", "1.5", "1, -2"])
def test_parse_numbers_rejects(text):
    with pytest.raises(ValueError):
        parse_numbers(text)


def test_parse_irrep():
    assert parse_irrep(3, "1, 1", DYNKIN) == IrrepSUN(3, (1, 1))
    assert parse_irrep(3, "0", DYNKIN) == IrrepSUN(3, (0, 0))
    assert parse_irrep(3, "2, 1", PARTITION) == IrrepSUN(3, (1, 1))
    assert parse_irrep(4, "1, 1, 1, 1", PARTITION) == IrrepSUN(4, (0, 0, 0))
    with pytest.raises(ValueError):
        parse_irrep(3, "1, 1, 1", DYNKIN)


@pytest.mark.parametrize("n, texts", [
    (1, ["1", "1"]), (13, ["1"] * 2), (3, ["1, 1", ""]), (3, ["1, 0"] * 5), (3, ["40, 1", "1, 0"]),
    (3, ["10, 10", "20, 5"]),
])
def test_parse_factors_rejects(n, texts):
    with pytest.raises(ValueError):
        parse_factors(n, DYNKIN, texts)


def test_compute_octets():
    page, text = compute(3, DYNKIN, "1, 1", "1, 1", "", "")
    assert text == "(1, 1)[8] ⊗ (1, 1)[8] = (2, 2)[27] ⊕ (3, 0)[10] ⊕ (0, 3)[10 bar] ⊕ " \
                   "2 (1, 1)[8] ⊕ (0, 0)[1]"
    assert "64 = 64 ✔" in page
    assert "text-decoration: overline" in page   # the 10bar


def test_compute_error_is_shown():
    page, text = compute(3, DYNKIN, "1, x", "1, 1", "", "")
    assert text == ""
    assert "not a list of integers" in page


@pytest.mark.parametrize("example", EXAMPLES)
def test_examples_run(example):
    page, text = compute(*example, "")
    assert "✔" in page and text


def test_young_svg():
    assert young_svg(IrrepSUN(3, (1, 1))).count("<rect") == 3
    assert "<svg" not in young_svg(IrrepSUN(3, (0, 0)))
