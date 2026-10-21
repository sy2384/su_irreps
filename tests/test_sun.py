# Tests for the SU(N) code in sun_irrep_CG.py
from collections import Counter
import itertools

import pytest

from sun_irrep_CG import IrrepSUN, decompose, decompose_many, multiply_irreps, format_product, horizontal_strips
from characters import character, character_product, character_sum


def irreps_up_to(n, max_label):
    return [IrrepSUN(n, labels) for labels in itertools.product(range(max_label + 1), repeat=n - 1)]


def as_counter(terms):
    return Counter({irrep.labels: mult for irrep, mult in terms})


def fundamental(n):
    return IrrepSUN(n, [1] + [0] * (n - 2))


def adjoint(n):
    return IrrepSUN(n, [1] + [0] * (n - 3) + [1]) if n > 2 else IrrepSUN(2, [2])


# Hook-content formula: an independent way to get the dimension of an SU(N) irrep
def hook_content_dim(irrep):
    rows = irrep.partition
    cols = [sum(1 for r in rows if r > c) for c in range(rows[0])] if rows else []
    numerator = 1
    denominator = 1
    for r, length in enumerate(rows):
        for c in range(length):
            numerator *= irrep.n + c - r
            denominator *= (length - c - 1) + (cols[c] - r - 1) + 1
    return numerator // denominator


# Irreps used for the exhaustive checks, kept small enough that the character check stays fast
GRIDS = {2: irreps_up_to(2, 6), 3: irreps_up_to(3, 2), 4: irreps_up_to(4, 1), 5: irreps_up_to(5, 1)}
PAIRS = [(a, b) for n, irreps in GRIDS.items() for a, b in itertools.product(irreps, repeat=2)]
PAIR_IDS = [str(a.labels) + 'x' + str(b.labels) for a, b in PAIRS]


# ---- the irrep class ----

@pytest.mark.parametrize("n, labels, dim", [
    (2, (0,), 1), (2, (1,), 2), (2, (4,), 5),
    (3, (1, 1), 8), (3, (2, 1), 15), (3, (4, 0), 15),
    (4, (1, 0, 0), 4), (4, (0, 1, 0), 6), (4, (1, 0, 1), 15), (4, (0, 2, 0), 20), (4, (2, 1, 0), 45),
    (4, (2, 0, 2), 84),
    (5, (1, 0, 0, 0), 5), (5, (0, 1, 0, 0), 10), (5, (1, 0, 0, 1), 24), (5, (1, 0, 1, 0), 45),
    (5, (0, 2, 0, 0), 50),
    (6, (0, 0, 1, 0, 0), 20), (10, (0,) * 9, 1),
])
def test_known_dimensions(n, labels, dim):
    assert IrrepSUN(n, labels).dim == dim


@pytest.mark.parametrize("irrep", [irrep for n in range(2, 7) for irrep in irreps_up_to(n, 2 if n < 5 else 1)],
                         ids=str)
def test_dimension_matches_hook_content(irrep):
    assert irrep.dim == hook_content_dim(irrep)


@pytest.mark.parametrize("irrep", [irrep for n in range(2, 6) for irrep in irreps_up_to(n, 2)], ids=str)
def test_partition_round_trip(irrep):
    assert IrrepSUN.from_partition(irrep.n, irrep.partition) == irrep


def test_from_partition_drops_full_columns():
    assert IrrepSUN.from_partition(3, (3, 2, 1)) == IrrepSUN(3, (1, 1))
    assert IrrepSUN.from_partition(4, (2, 2, 2, 2)) == IrrepSUN(4, (0, 0, 0))


@pytest.mark.parametrize("n, labels", [(1, ()), (3, (1,)), (3, (1, 2, 3)), (3, (-1, 0))])
def test_bad_labels(n, labels):
    with pytest.raises(ValueError):
        IrrepSUN(n, labels)


@pytest.mark.parametrize("partition", [(1, 1, 1, 1), (1, 2)])
def test_bad_partition(partition):
    with pytest.raises(ValueError):
        IrrepSUN.from_partition(3, partition)


def test_repr_and_bar():
    assert str(IrrepSUN(3, (0, 3))) == "(0, 3)[10 bar]"
    assert str(IrrepSUN(2, (1,))) == "(1)[2]"
    assert str(IrrepSUN(5, (0, 0, 0, 1))) == "(0, 0, 0, 1)[5 bar]"
    assert not IrrepSUN(4, (0, 1, 0)).bar   # self-conjugate
    assert IrrepSUN(4, (1, 2, 0)).conjugate() == IrrepSUN(4, (0, 2, 1))


def test_young_diagram():
    assert IrrepSUN(3, (2, 1)).young_diagram(box='#') == "###\n#"
    assert IrrepSUN(3, (0, 0)).young_diagram() == ""


def test_horizontal_strips():
    assert sorted(horizontal_strips((2, 1), 2, 3)) == [(0, 1, 1), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
    assert sorted(horizontal_strips((2, 1, 1), 1, 3)) == [(0, 1, 0), (1, 0, 0)]
    # With two a's in the first row, the b's must all go below them
    assert list(horizontal_strips((2,), 2, 3, previous=(2,))) == [(0, 2)]
    assert sorted(horizontal_strips((2, 1), 2, 3, previous=(1, 1))) == [(0, 1, 1)]


# ---- known decompositions ----

@pytest.mark.parametrize("n, a, b, expected", [
    (2, (1,), (1,), [(2,), (0,)]),                                         # spin 1/2 x 1/2 = 1 + 0
    (2, (2,), (3,), [(5,), (3,), (1,)]),                                   # spin 1 x 3/2 = 5/2 + 3/2 + 1/2
    (3, (1, 1), (1, 1), [(2, 2), (3, 0), (0, 3), (1, 1), (1, 1), (0, 0)]),  # 8 x 8
    (3, (2, 0), (2, 0), [(4, 0), (2, 1), (0, 2)]),                         # 6 x 6 = 15' + 15 + 6bar
    (4, (1, 0, 0), (1, 0, 0), [(2, 0, 0), (0, 1, 0)]),                     # 4 x 4 = 10 + 6
    (4, (1, 0, 0), (0, 0, 1), [(1, 0, 1), (0, 0, 0)]),                     # 4 x 4bar = 15 + 1
    (4, (0, 1, 0), (0, 1, 0), [(0, 2, 0), (1, 0, 1), (0, 0, 0)]),          # 6 x 6 = 20' + 15 + 1
    (4, (1, 0, 1), (1, 0, 1),                                              # 15 x 15
     [(2, 0, 2), (2, 1, 0), (0, 1, 2), (0, 2, 0), (1, 0, 1), (1, 0, 1), (0, 0, 0)]),
    (5, (1, 0, 0, 0), (0, 0, 0, 1), [(1, 0, 0, 1), (0, 0, 0, 0)]),         # 5 x 5bar = 24 + 1
    (5, (0, 0, 0, 1), (0, 1, 0, 0), [(0, 1, 0, 1), (1, 0, 0, 0)]),         # 5bar x 10 = 45 + 5
    (5, (0, 1, 0, 0), (0, 1, 0, 0), [(0, 2, 0, 0), (1, 0, 1, 0), (0, 0, 0, 1)]),  # 10 x 10 = 50 + 45 + 5bar
])
def test_known_products(n, a, b, expected):
    assert as_counter(decompose(IrrepSUN(n, a), IrrepSUN(n, b))) == Counter(expected)


@pytest.mark.parametrize("j1, j2", list(itertools.product(range(9), repeat=2)))
def test_su2_angular_momentum(j1, j2):
    # In SU(2), the label is 2j: spin j1 x spin j2 = |j1 - j2| + ... + (j1 + j2), each once
    expected = Counter({(j,): 1 for j in range(abs(j1 - j2), j1 + j2 + 1, 2)})
    assert as_counter(decompose(IrrepSUN(2, (j1,)), IrrepSUN(2, (j2,)))) == expected


@pytest.mark.parametrize("n", range(2, 11))
def test_fundamental_times_antifundamental(n):
    terms = decompose(fundamental(n), fundamental(n).conjugate())
    assert as_counter(terms) == Counter({adjoint(n).labels: 1, (0,) * (n - 1): 1})
    assert adjoint(n).dim == n * n - 1


@pytest.mark.parametrize("n", range(3, 11))
def test_fundamental_squared(n):
    # symmetric + antisymmetric
    terms = decompose(fundamental(n), fundamental(n))
    assert as_counter(terms) == Counter({(2,) + (0,) * (n - 2): 1, (0, 1) + (0,) * (n - 3): 1})
    assert [irrep.dim for irrep, _ in terms] == [n * (n + 1) // 2, n * (n - 1) // 2]


def test_three_quarks():
    # SU(3): 3 x 3 x 3 = 10 + 8 + 8 + 1
    terms = decompose_many([IrrepSUN(3, (1, 0))] * 3)
    assert as_counter(terms) == Counter({(3, 0): 1, (1, 1): 2, (0, 0): 1})


# ---- exhaustive checks on small irreps ----

@pytest.mark.parametrize("a, b", PAIRS, ids=PAIR_IDS)
def test_agrees_with_characters(a, b):
    assert character_sum(decompose(a, b)) == character_product(character(a), character(b))


@pytest.mark.parametrize("a, b", PAIRS, ids=PAIR_IDS)
def test_dimensions_add_up(a, b):
    assert sum(irrep.dim * mult for irrep, mult in decompose(a, b)) == a.dim * b.dim


@pytest.mark.parametrize("a, b", PAIRS, ids=PAIR_IDS)
def test_commutative(a, b):
    assert decompose(a, b) == decompose(b, a)


@pytest.mark.parametrize("a, b", PAIRS, ids=PAIR_IDS)
def test_conjugation(a, b):
    conjugated = Counter({irrep.conjugate(): mult for irrep, mult in decompose(a, b)})
    assert conjugated == Counter(dict(decompose(a.conjugate(), b.conjugate())))


@pytest.mark.parametrize("a, b", PAIRS, ids=PAIR_IDS)
def test_singlet_only_with_conjugate(a, b):
    singlet = IrrepSUN(a.n, (0,) * (a.n - 1))
    assert dict(decompose(a, b)).get(singlet, 0) == (1 if b == a.conjugate() else 0)


@pytest.mark.parametrize("a, b, c", [(a, b, c) for a, b, c in itertools.product(GRIDS[3], repeat=3)][::7])
def test_associative(a, b, c):
    left = decompose_many([a, b, c])
    right = Counter()
    for irrep, mult in decompose(b, c):
        for result, m in decompose(a, irrep):
            right[result] += mult * m
    assert Counter(dict(left)) == right


def test_large_product_dimension():
    a, b = IrrepSUN(6, (2, 0, 1, 0, 1)), IrrepSUN(6, (1, 1, 0, 1, 0))
    assert sum(irrep.dim * mult for irrep, mult in decompose(a, b)) == a.dim * b.dim


# ---- helpers ----

def test_multiply_irreps_repeats_multiplicities():
    terms = multiply_irreps(IrrepSUN(3, (1, 1)), IrrepSUN(3, (1, 1)))
    assert [irrep.labels for irrep in terms].count((1, 1)) == 2
    assert len(terms) == 6


def test_format_product():
    a = IrrepSUN(3, (1, 1))
    assert format_product([a, a], decompose(a, a)) == \
        "(1, 1)[8] ⊗ (1, 1)[8] = (2, 2)[27] ⊕ (3, 0)[10] ⊕ (0, 3)[10 bar] ⊕ 2 (1, 1)[8] ⊕ (0, 0)[1]"


def test_mismatched_groups():
    with pytest.raises(ValueError):
        decompose(IrrepSUN(3, (1, 0)), IrrepSUN(4, (1, 0, 0)))
