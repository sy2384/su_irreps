# Tests for the input handling and output of the web app (needs gradio installed)
import pytest

pytest.importorskip("gradio")

from app import parse_numbers, parse_irrep, parse_factors, compute, young_svg, DYNKIN, PARTITION, EXAMPLES
from lie_algebra_CG import LieIrrep, lie_algebra
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
    su3, su4 = lie_algebra('SU(3)'), lie_algebra('SU(4)')
    assert parse_irrep(su3, "1, 1", DYNKIN) == IrrepSUN(3, (1, 1))
    assert parse_irrep(su3, "0", DYNKIN) == IrrepSUN(3, (0, 0))
    assert parse_irrep(su3, "2, 1", PARTITION) == IrrepSUN(3, (1, 1))
    assert parse_irrep(su4, "1, 1, 1, 1", PARTITION) == IrrepSUN(4, (0, 0, 0))
    assert parse_irrep(lie_algebra('G2'), "1, 0", DYNKIN) == LieIrrep('G2', (1, 0))
    assert parse_irrep(lie_algebra('E6'), "", DYNKIN) == LieIrrep('E6', (0,) * 6)
    with pytest.raises(ValueError):
        parse_irrep(su3, "1, 1, 1", DYNKIN)
    with pytest.raises(ValueError):
        parse_irrep(lie_algebra('G2'), "1", DYNKIN)
    with pytest.raises(ValueError):
        parse_irrep(lie_algebra('SO(10)'), "1, 1", PARTITION)   # Young diagrams only for SU(N)


@pytest.mark.parametrize("group, texts", [
    ("SU(1)", ["", ""]), ("SU(13)", ["1"] * 2), ("SU(3)", ["1, 1", ""]), ("SU(3)", ["1, 0"] * 5),
    ("SU(3)", ["40, 1", "1, 0"]), ("SU(3)", ["10, 10", "20, 5"]), ("X7", ["1", "1"]), ("SO(4)", ["1, 1", "1, 1"]),
    ("SO(40)", ["1"] * 2),
])
def test_parse_factors_rejects(group, texts):
    with pytest.raises(ValueError):
        parse_factors(group, DYNKIN, texts)


def test_parse_factors_picks_the_right_code():
    assert all(isinstance(f, IrrepSUN) for f in parse_factors("SU(3)", DYNKIN, ["1, 0", "0, 1"]))
    assert all(isinstance(f, IrrepSUN) for f in parse_factors(" a2 ", DYNKIN, ["1, 0", "0, 1"]))
    assert all(isinstance(f, LieIrrep) for f in parse_factors("SO(10)", DYNKIN, ["1, 0, 0, 0, 0", "0"]))


def test_compute_octets():
    page, text = compute("SU(3)", DYNKIN, "1, 1", "1, 1", "", "")
    assert text == "(1, 1)[8] ⊗ (1, 1)[8] = (2, 2)[27] ⊕ (3, 0)[10] ⊕ (0, 3)[10 bar] ⊕ " \
                   "2 (1, 1)[8] ⊕ (0, 0)[1]"
    assert "64 = 64 ✔" in page
    assert "text-decoration: overline" in page   # the 10bar


def test_compute_error_is_shown():
    page, text = compute("SU(3)", DYNKIN, "1, x", "1, 1", "", "")
    assert text == ""
    assert "not a list of integers" in page


@pytest.mark.parametrize("example", EXAMPLES)
def test_examples_run(example):
    page, text = compute(*example, "")
    assert "✔" in page and text


def test_compute_so10():
    page, text = compute("SO(10)", DYNKIN, "0, 0, 0, 1, 0", "0, 0, 0, 1, 0", "", "")
    assert text == "(0, 0, 0, 1, 0)[16] \u2297 (0, 0, 0, 1, 0)[16] = (0, 0, 0, 2, 0)[126] \u2295 (0, 0, 1, 0, 0)[120] " \
                   "\u2295 (1, 0, 0, 0, 0)[10]"
    assert "256 = 256 \u2714" in page
    assert "Young diagram" not in page


def test_compute_too_many_weights():
    page, text = compute("E8", DYNKIN, "0, 0, 0, 1, 0, 0, 0, 0", "0, 0, 0, 1, 0, 0, 0, 0", "", "")
    assert text == "" and "too large" in page


def test_young_svg():
    assert young_svg(IrrepSUN(3, (1, 1))).count("<rect") == 3
    assert "<svg" not in young_svg(IrrepSUN(3, (0, 0)))
    assert young_svg(LieIrrep('G2', (1, 0))) == ''

