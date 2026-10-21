# Tests for the general simple Lie algebra code in lie_algebra_CG.py
from collections import Counter
import itertools
from math import comb

import pytest

from lie_algebra_CG import LieIrrep, SimpleLieAlgebra, lie_algebra, decompose, decompose_many, format_product
from sun_irrep_CG import IrrepSUN, decompose as decompose_sun

ALL_ALGEBRAS = ['A1', 'A2', 'A3', 'A4', 'B2', 'B3', 'B4', 'C2', 'C3', 'C4', 'D3', 'D4', 'D5', 'G2', 'F4', 'E6', 'E7',
                'E8']


def fundamentals(algebra):
    g = lie_algebra(algebra)
    return [LieIrrep(g, [int(i == j) for j in range(g.rank)]) for i in range(g.rank)]


def irreps_up_to(algebra, total):
    g = lie_algebra(algebra)
    return [LieIrrep(g, labels) for labels in itertools.product(range(total + 1), repeat=g.rank)
            if sum(labels) <= total]


def dims(terms):
    return sorted((irrep.dim for irrep, mult in terms for _ in range(mult)), reverse=True)


def as_counter(terms):
    return Counter({irrep.labels: mult for irrep, mult in terms})


def adjoint(algebra):
    g = lie_algebra(algebra)
    highest_root = max(g.positive_roots, key=lambda alpha: sum(g.weight_to_root(alpha)))
    return LieIrrep(g, highest_root)


# Small irreps of every algebra, for the exhaustive checks; the exceptional ones only get their smallest irreps
GRID = {name: irreps_up_to(name, 2 if lie_algebra(name).rank <= 2 else 1) for name in ALL_ALGEBRAS[:13]}
GRID['G2'] = irreps_up_to('G2', 2)
GRID['F4'] = [LieIrrep('F4', (0, 0, 0, 0)), LieIrrep('F4', (0, 0, 0, 1)), LieIrrep('F4', (1, 0, 0, 0))]
GRID['E6'] = [LieIrrep('E6', (0,) * 6), LieIrrep('E6', (1, 0, 0, 0, 0, 0)), LieIrrep('E6', (0, 0, 0, 0, 0, 1)),
              LieIrrep('E6', (0, 1, 0, 0, 0, 0))]
GRID['E7'] = [LieIrrep('E7', (0,) * 7), LieIrrep('E7', (0,) * 6 + (1,)), LieIrrep('E7', (1,) + (0,) * 6)]
GRID['E8'] = [LieIrrep('E8', (0,) * 8), LieIrrep('E8', (0,) * 7 + (1,))]
PAIRS = [(a, b) for name in ALL_ALGEBRAS for a, b in itertools.product(GRID[name], repeat=2)
         if a.dim * b.dim <= 70000]
PAIR_IDS = ['{}:{}x{}'.format(a.algebra.series + str(a.algebra.rank), a.labels, b.labels) for a, b in PAIRS]


# ---- root systems ----

@pytest.mark.parametrize("name, n_positive", [
    ('A1', 1), ('A4', 10), ('B3', 9), ('C4', 16), ('D4', 12), ('D5', 20), ('G2', 6), ('F4', 24), ('E6', 36),
    ('E7', 63), ('E8', 120),
])
def test_number_of_positive_roots(name, n_positive):
    assert len(lie_algebra(name).positive_roots) == n_positive


@pytest.mark.parametrize("name, cartan", [
    ('A3', ((2, -1, 0), (-1, 2, -1), (0, -1, 2))),
    ('B3', ((2, -1, 0), (-1, 2, -2), (0, -1, 2))),
    ('C3', ((2, -1, 0), (-1, 2, -1), (0, -2, 2))),
    ('D4', ((2, -1, 0, 0), (-1, 2, -1, -1), (0, -1, 2, 0), (0, -1, 0, 2))),
    ('G2', ((2, -1), (-3, 2))),
    ('F4', ((2, -1, 0, 0), (-1, 2, -2, 0), (0, -1, 2, -1), (0, 0, -1, 2))),
    ('E6', ((2, 0, -1, 0, 0, 0), (0, 2, 0, -1, 0, 0), (-1, 0, 2, -1, 0, 0), (0, -1, -1, 2, -1, 0),
            (0, 0, 0, -1, 2, -1), (0, 0, 0, 0, -1, 2))),
])
def test_cartan_matrix(name, cartan):
    # Row i = Dynkin labels of the simple root alpha_i (Bourbaki numbering)
    assert lie_algebra(name).cartan == cartan


@pytest.mark.parametrize("name", ALL_ALGEBRAS)
def test_adjoint_dimension(name):
    g = lie_algebra(name)
    assert adjoint(name).dim == g.rank + 2 * len(g.positive_roots)


@pytest.mark.parametrize("name, fundamental_dims", [
    ('SU(5)', [5, 10, 10, 5]),
    ('SO(7)', [7, 21, 8]),
    ('SO(9)', [9, 36, 84, 16]),
    ('Sp(6)', [6, 14, 14]),
    ('SO(8)', [8, 28, 8, 8]),
    ('SO(10)', [10, 45, 120, 16, 16]),
    ('G2', [7, 14]),
    ('F4', [52, 1274, 273, 26]),
    ('E6', [27, 78, 351, 2925, 351, 27]),
    ('E7', [133, 912, 8645, 365750, 27664, 1539, 56]),
    ('E8', [3875, 147250, 6696000, 6899079264, 146325270, 2450240, 30380, 248]),
])
def test_fundamental_dimensions(name, fundamental_dims):
    assert [irrep.dim for irrep in fundamentals(name)] == fundamental_dims


@pytest.mark.parametrize("name, expected", [
    ('SU(3)', ('A', 2)), ('su3', ('A', 2)), ('SO(10)', ('D', 5)), ('so(7)', ('B', 3)), ('Sp(4)', ('C', 2)),
    ('USp(6)', ('C', 3)), ('G2', ('G', 2)), ('g_2', ('G', 2)), ('E 8', ('E', 8)), ('F4', ('F', 4)),
])
def test_parse_names(name, expected):
    g = lie_algebra(name)
    assert (g.series, g.rank) == expected


@pytest.mark.parametrize("name", ['SU(1)', 'SO(4)', 'SO(3)', 'Sp(3)', 'E5', 'E9', 'G3', 'F2', 'B1', 'D2', 'H3', ''])
def test_bad_names(name):
    with pytest.raises(ValueError):
        lie_algebra(name)


def test_names():
    assert [lie_algebra(n).name for n in ['A2', 'B3', 'C2', 'D5', 'E6']] == ['SU(3)', 'SO(7)', 'Sp(4)', 'SO(10)', 'E6']


# ---- irreps ----

@pytest.mark.parametrize("irrep", [irrep for name in ALL_ALGEBRAS[:13] for irrep in irreps_up_to(name, 2)] +
                         [irrep for name in ['G2', 'F4', 'E6', 'E7'] for irrep in irreps_up_to(name, 1)], ids=str)
def test_weight_multiplicities_add_up_to_dimension(irrep):
    # Freudenthal's formula and the Weyl dimension formula are independent of each other
    assert sum(irrep.weights().values()) == irrep.dim


@pytest.mark.parametrize("name, labels, multiplicities", [
    ('SU(3)', (1, 1), {(1, 1): 1, (0, 0): 2}),                          # octet: zero weight twice
    ('SU(3)', (2, 2), {(2, 2): 1, (3, 0): 1, (0, 3): 1, (1, 1): 2, (0, 0): 3}),
    ('G2', (0, 1), {(0, 1): 1, (1, 0): 1, (0, 0): 2}),                  # 14 = 12 roots + rank 2
    ('E8', (0,) * 7 + (1,), {(0,) * 7 + (1,): 1, (0,) * 8: 8}),         # 248 = 240 roots + rank 8
    ('SO(10)', (0, 0, 0, 1, 0), {(0, 0, 0, 1, 0): 1}),                  # spinors are minuscule
])
def test_dominant_multiplicities(name, labels, multiplicities):
    g = lie_algebra(name)
    assert g.dominant_multiplicities(labels) == multiplicities


@pytest.mark.parametrize("name, labels, conjugate", [
    ('SU(4)', (1, 2, 0), (0, 2, 1)), ('SO(10)', (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)), ('SO(8)', (0, 0, 1, 0), (0, 0, 1, 0)),
    ('E6', (1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)), ('E6', (0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 1, 0)),
    ('E7', (0,) * 6 + (1,), (0,) * 6 + (1,)), ('G2', (1, 0), (1, 0)), ('Sp(6)', (1, 0, 1), (1, 0, 1)),
])
def test_conjugate(name, labels, conjugate):
    assert LieIrrep(name, labels).conjugate().labels == conjugate


def test_repr():
    assert str(LieIrrep('E6', (0, 0, 0, 0, 0, 1))) == "(0, 0, 0, 0, 0, 1)[27 bar]"
    assert str(LieIrrep('G2', (1, 0))) == "(1, 0)[7]"


@pytest.mark.parametrize("name, labels", [('SU(3)', (1,)), ('G2', (1, 0, 0)), ('E6', (-1, 0, 0, 0, 0, 0))])
def test_bad_labels(name, labels):
    with pytest.raises(ValueError):
        LieIrrep(name, labels)


# ---- known decompositions ----

@pytest.mark.parametrize("name, a, b, expected", [
    ('SO(10)', (0, 0, 0, 1, 0), (0, 0, 0, 1, 0), [126, 120, 10]),          # 16 x 16
    ('SO(10)', (0, 0, 0, 1, 0), (0, 0, 0, 0, 1), [210, 45, 1]),            # 16 x 16bar
    ('SO(10)', (1, 0, 0, 0, 0), (0, 0, 0, 1, 0), [144, 16]),               # 10 x 16
    ('SO(7)', (0, 0, 1), (0, 0, 1), [35, 21, 7, 1]),                       # spinor x spinor
    ('SO(9)', (0, 0, 0, 1), (0, 0, 0, 1), [126, 84, 36, 9, 1]),
    ('SO(8)', (1, 0, 0, 0), (1, 0, 0, 0), [35, 28, 1]),                    # 8v x 8v
    ('SO(8)', (1, 0, 0, 0), (0, 0, 1, 0), [56, 8]),                        # 8v x 8s = 56 + 8c
    ('Sp(4)', (1, 0), (1, 0), [10, 5, 1]),
    ('Sp(4)', (0, 1), (0, 1), [14, 10, 1]),
    ('Sp(6)', (1, 0, 0), (1, 0, 0), [21, 14, 1]),
    ('G2', (1, 0), (1, 0), [27, 14, 7, 1]),
    ('G2', (0, 1), (0, 1), [77, 77, 27, 14, 1]),
    ('G2', (1, 0), (0, 1), [64, 27, 7]),
    ('F4', (0, 0, 0, 1), (0, 0, 0, 1), [324, 273, 52, 26, 1]),
    ('F4', (1, 0, 0, 0), (1, 0, 0, 0), [1274, 1053, 324, 52, 1]),
    ('E6', (1, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), [351, 351, 27]),
    ('E6', (1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1), [650, 78, 1]),
    ('E6', (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), [1728, 351, 27]),
    ('E7', (0,) * 6 + (1,), (0,) * 6 + (1,), [1539, 1463, 133, 1]),
    ('E7', (0,) * 6 + (1,), (1,) + (0,) * 6, [6480, 912, 56]),
    ('E8', (0,) * 7 + (1,), (0,) * 7 + (1,), [30380, 27000, 3875, 248, 1]),
])
def test_known_products(name, a, b, expected):
    assert dims(decompose(LieIrrep(name, a), LieIrrep(name, b))) == expected


def test_known_labels():
    terms = decompose(LieIrrep('SO(10)', (0, 0, 0, 1, 0)), LieIrrep('SO(10)', (0, 0, 0, 1, 0)))
    assert as_counter(terms) == Counter({(0, 0, 0, 2, 0): 1, (0, 0, 1, 0, 0): 1, (1, 0, 0, 0, 0): 1})
    terms = decompose(LieIrrep('E6', (1, 0, 0, 0, 0, 0)), LieIrrep('E6', (1, 0, 0, 0, 0, 0)))
    assert as_counter(terms) == Counter({(2, 0, 0, 0, 0, 0): 1, (0, 0, 1, 0, 0, 0): 1, (0, 0, 0, 0, 0, 1): 1})


@pytest.mark.parametrize("name", ALL_ALGEBRAS)
def test_adjoint_squared(name):
    # adj x adj contains the singlet once, and the adjoint once (twice for SU(N), N >= 3, which includes SO(6))
    terms = dict(decompose(adjoint(name), adjoint(name)))
    g = lie_algebra(name)
    assert terms[LieIrrep(g, (0,) * g.rank)] == 1
    assert terms[adjoint(name)] == (2 if (g.series, g.rank) == ('D', 3) or (g.series == 'A' and g.rank >= 2) else 1)


@pytest.mark.parametrize("n", range(2, 6))
def test_odd_orthogonal_spinor_squared(n):
    # SO(2n+1): spinor x spinor = sum of the antisymmetric tensors with 0, 1, ..., n indices
    spinor = LieIrrep(SimpleLieAlgebra('B', n), (0,) * (n - 1) + (1,))
    expected = sorted((comb(2 * n + 1, k) for k in range(n + 1)), reverse=True)
    assert dims(decompose(spinor, spinor)) == expected


# ---- agreement with the Young tableau code ----

@pytest.mark.parametrize("n, a, b", [(n, a, b) for n in range(2, 6)
                                     for a, b in itertools.product(
                                         [l for l in itertools.product(range(3 if n < 4 else 2), repeat=n - 1)],
                                         repeat=2)])
def test_agrees_with_littlewood_richardson(n, a, b):
    weights = Counter({irrep.labels: mult for irrep, mult in decompose(LieIrrep('SU({})'.format(n), a),
                                                                         LieIrrep('SU({})'.format(n), b))})
    tableaux = Counter({irrep.labels: mult for irrep, mult in decompose_sun(IrrepSUN(n, a), IrrepSUN(n, b))})
    assert weights == tableaux


# ---- isomorphisms between small algebras ----

@pytest.mark.parametrize("a, b", list(itertools.product(itertools.product(range(3), repeat=2), repeat=2)))
def test_b2_is_c2(a, b):
    # SO(5) = Sp(4) with the two nodes of the Dynkin diagram swapped
    swap = lambda labels: labels[::-1]
    so5 = Counter({swap(irrep.labels): m for irrep, m in decompose(LieIrrep('B2', a), LieIrrep('B2', b))})
    sp4 = as_counter(decompose(LieIrrep('C2', swap(a)), LieIrrep('C2', swap(b))))
    assert so5 == sp4


@pytest.mark.parametrize("a, b", list(itertools.product(itertools.product(range(2), repeat=3), repeat=2)))
def test_d3_is_a3(a, b):
    # SO(6) = SU(4): the middle node of A3 is the first node of D3
    to_d3 = lambda labels: (labels[1], labels[0], labels[2])
    su4 = Counter({to_d3(irrep.labels): m for irrep, m in decompose(LieIrrep('A3', a), LieIrrep('A3', b))})
    so6 = as_counter(decompose(LieIrrep('D3', to_d3(a)), LieIrrep('D3', to_d3(b))))
    assert su4 == so6


# ---- exhaustive checks on small irreps ----

def character_product(a, b):
    result = Counter()
    for w1, m1 in a.weights().items():
        for w2, m2 in b.weights().items():
            result[tuple(x + y for x, y in zip(w1, w2))] += m1 * m2
    return result


def character_sum(terms):
    result = Counter()
    for irrep, mult in terms:
        for w, m in irrep.weights().items():
            result[w] += mult * m
    return result


@pytest.mark.parametrize("a, b", [(a, b) for a, b in PAIRS if a.dim * b.dim <= 20000],
                         ids=[i for (a, b), i in zip(PAIRS, PAIR_IDS) if a.dim * b.dim <= 20000])
def test_agrees_with_characters(a, b):
    assert character_sum(decompose(a, b)) == character_product(a, b)


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
    singlet = LieIrrep(a.algebra, (0,) * a.algebra.rank)
    assert dict(decompose(a, b)).get(singlet, 0) == (1 if b == a.conjugate() else 0)


@pytest.mark.parametrize("name", ['B2', 'C3', 'D4', 'G2'])
def test_associative(name):
    irreps = GRID[name][:6]
    for a, b, c in itertools.product(irreps, repeat=3):
        right = Counter()
        for irrep, mult in decompose(b, c):
            for result, m in decompose(a, irrep):
                right[result] += mult * m
        assert Counter(dict(decompose_many([a, b, c]))) == right


# ---- helpers ----

def test_format_product():
    a = LieIrrep('G2', (1, 0))
    assert format_product([a, a], decompose(a, a)) == \
        "(1, 0)[7] ⊗ (1, 0)[7] = (2, 0)[27] ⊕ (0, 1)[14] ⊕ (1, 0)[7] ⊕ (0, 0)[1]"


def test_mismatched_algebras():
    with pytest.raises(ValueError):
        decompose(LieIrrep('B2', (1, 0)), LieIrrep('C2', (1, 0)))
