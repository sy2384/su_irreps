# Tests for the original SU(3) code in su3_irrep_CG.py
from collections import Counter
import itertools

import pytest

from su3_irrep_CG import Irrep, multiply_irreps, to_irrep, big_check
from sun_irrep_CG import IrrepSUN, decompose
from characters import character, character_product, character_sum


def labels_of(irreps):
    return Counter((irrep.idx1, irrep.idx2) for irrep in irreps)


SMALL = [(n, m) for n in range(4) for m in range(4)]


@pytest.mark.parametrize("labels, dim, bar", [
    ((0, 0), 1, False), ((1, 0), 3, False), ((0, 1), 3, True), ((2, 0), 6, False), ((0, 2), 6, True),
    ((1, 1), 8, False), ((3, 0), 10, False), ((0, 3), 10, True), ((2, 1), 15, False), ((4, 0), 15, False),
    ((2, 2), 27, False), ((4, 1), 35, False), ((3, 3), 64, False),
])
def test_dimension_and_bar(labels, dim, bar):
    irrep = Irrep(*labels)
    assert irrep.dim == dim
    assert irrep.bar == bar


def test_repr():
    assert str(Irrep(1, 1)) == "(1, 1)[8]"
    assert str(Irrep(0, 3)) == "(0, 3)[10 bar]"


@pytest.mark.parametrize("a, b, expected", [
    ((1, 0), (1, 0), [(2, 0), (0, 1)]),                          # 3 x 3 = 6 + 3bar
    ((1, 0), (0, 1), [(1, 1), (0, 0)]),                          # 3 x 3bar = 8 + 1
    ((0, 1), (0, 1), [(0, 2), (1, 0)]),                          # 3bar x 3bar = 6bar + 3
    ((2, 0), (1, 0), [(3, 0), (1, 1)]),                          # 6 x 3 = 10 + 8
    ((2, 0), (0, 1), [(2, 1), (1, 0)]),                          # 6 x 3bar = 15 + 3
    ((2, 0), (2, 0), [(4, 0), (2, 1), (0, 2)]),                  # 6 x 6 = 15' + 15 + 6bar
    ((2, 0), (0, 2), [(2, 2), (1, 1), (0, 0)]),                  # 6 x 6bar = 27 + 8 + 1
    ((1, 1), (1, 1), [(2, 2), (3, 0), (0, 3), (1, 1), (1, 1), (0, 0)]),  # 8 x 8
    ((1, 1), (3, 0), [(4, 1), (2, 2), (3, 0), (1, 1)]),          # 8 x 10 = 35 + 27 + 10 + 8
    ((3, 0), (0, 3), [(3, 3), (2, 2), (1, 1), (0, 0)]),          # 10 x 10bar = 64 + 27 + 8 + 1
    ((0, 0), (2, 1), [(2, 1)]),                                  # 1 x 15 = 15
])
def test_known_products(a, b, expected):
    assert labels_of(multiply_irreps(Irrep(*a), Irrep(*b))) == Counter(expected)


@pytest.mark.parametrize("a, b", list(itertools.product(SMALL, repeat=2)))
def test_dimensions_add_up(a, b):
    terms = multiply_irreps(Irrep(*a), Irrep(*b))
    assert sum(irrep.dim for irrep in terms) == Irrep(*a).dim * Irrep(*b).dim


@pytest.mark.parametrize("a, b", list(itertools.product(SMALL, repeat=2)))
def test_commutative(a, b):
    assert labels_of(multiply_irreps(Irrep(*a), Irrep(*b))) == labels_of(multiply_irreps(Irrep(*b), Irrep(*a)))


@pytest.mark.parametrize("a, b", list(itertools.product(SMALL, repeat=2)))
def test_agrees_with_sun_code(a, b):
    general = Counter({irrep.labels: mult for irrep, mult in decompose(IrrepSUN(3, a), IrrepSUN(3, b))})
    assert labels_of(multiply_irreps(Irrep(*a), Irrep(*b))) == general


@pytest.mark.parametrize("a, b", list(itertools.product([(n, m) for n in range(3) for m in range(3)], repeat=2)))
def test_agrees_with_characters(a, b):
    terms = [(IrrepSUN(3, (irrep.idx1, irrep.idx2)), 1) for irrep in multiply_irreps(Irrep(*a), Irrep(*b))]
    assert character_sum(terms) == character_product(character(IrrepSUN(3, a)), character(IrrepSUN(3, b)))


def test_larger_product():
    # (2, 1) x (6, 7), the largest case in test_cases.csv
    terms = multiply_irreps(Irrep(2, 1), Irrep(6, 7))
    assert sum(irrep.dim for irrep in terms) == 15 * 420
    assert labels_of(terms)[(7, 7)] == 2


def test_to_irrep_drops_full_columns():
    assert (to_irrep([['1', '1'], ['2'], ['a']]).idx1, to_irrep([['1', '1'], ['2'], ['a']]).idx2) == (1, 0)
    assert (to_irrep([]).idx1, to_irrep([]).idx2) == (0, 0)
    assert (to_irrep([[]]).idx1, to_irrep([[]]).idx2) == (0, 0)


@pytest.mark.parametrize("matrix, ok", [
    ([['1', 'a'], ['a']], True),
    ([['1', 'a'], ['2', 'a']], False),         # two a's in one column
    ([['1', '1', 'a'], ['2', 'a', 'b']], True),
    ([['1', '1', 'b'], ['2', 'a']], False),    # b read before any a
    ([['1'], ['2'], ['a'], ['b']], False),     # 4 rows
    ([['1'], ['2', 'a']], False),              # rows must not get longer
])
def test_big_check(matrix, ok):
    assert big_check(matrix) == ok
