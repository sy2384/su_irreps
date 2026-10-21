# Tests for the branching rules in branching.py
from collections import Counter
from fractions import Fraction

import pytest

from branching import (branch, embedding, describe, format_branching, dynkin_index, identify_diagram, extended_roots,
                       cartan_of_roots, principal_embedding, defining_rep_embedding, fundamental, number_of_weights,
                       is_u1, U1, A1, SubIrrep, EXCEPTIONAL, dimension_name, adjoint_labels, parse_factors)
from lie_algebra_CG import LieIrrep, lie_algebra, decompose


def strings(terms):
    return sorted(str(sub) if m == 1 else '{}·{}'.format(m, sub) for sub, m in terms)


def branching_of(group, subgroup, labels):
    e = embedding(group, subgroup)
    return strings(branch(LieIrrep(e.parent[0], labels), e))


def small_irreps(g, limit=3000):
    # the trivial irrep, the fundamental irreps and the adjoint, when small enough
    labels = {(0,) * g.rank, tuple(adjoint_labels(g))} | {fundamental(g, i + 1) for i in range(g.rank)}
    return [LieIrrep(g, x) for x in sorted(labels) if g.dimension(x) <= limit]


# Every named subalgebra, and a sample of the classical families
EMBEDDINGS = [(name, 'x'.join(factors)) for name, factors, _ in EXCEPTIONAL] + [
    ('SU(5)', 'SU(3)xSU(2)xU(1)'), ('SU(5)', 'SU(4)xU(1)'), ('SU(6)', 'SU(3)xSU(2)'), ('SU(4)', 'SU(2)xSU(2)'),
    ('SU(3)', 'SO(3)'), ('SU(6)', 'Sp(6)'), ('SU(6)', 'SO(6)'), ('SU(9)', 'SU(3)xSU(3)'),
    ('SO(10)', 'SU(5)xU(1)'), ('SO(10)', 'SU(4)xSU(2)xSU(2)'), ('SO(10)', 'SO(9)'), ('SO(10)', 'SO(7)xSO(3)'),
    ('SO(10)', 'SO(8)xU(1)'), ('SO(9)', 'SO(7)xU(1)'), ('SO(11)', 'SO(8)xSO(3)'), ('SO(7)', 'G2'), ('SO(8)', 'SO(7)'),
    ('SO(12)', 'Sp(6)xSU(2)'), ('SO(15)', 'SO(5)xSO(3)'), ('SO(16)', 'Sp(4)xSp(4)'),
    ('Sp(4)', 'SU(2)xSU(2)'), ('Sp(6)', 'SU(2)xSU(2)'), ('Sp(8)', 'SU(4)xU(1)'), ('Sp(8)', 'Sp(4)xSp(4)'),
    ('Sp(10)', 'Sp(6)xSp(4)'),
    ('E6', 'SU(2)'), ('E7', 'SU(2)'), ('E8', 'SU(2)'), ('F4', 'SU(2)'), ('G2', 'SU(2)'),
    ('E8', 'SO(14)xU(1)'), ('E7', 'SO(10)xSU(2)xU(1)'), ('E6', 'SO(10)'), ('SU(5)', 'SU(3)xSU(2)'),
    ('SO(10)', 'SU(5)xU(1) > SU(3)xSU(2)xU(1)xU(1)'), ('E6', 'SO(10)xU(1) > SU(5)xU(1)xU(1)'),
    ('E6', 'SU(3)xSU(3)xSU(3) > SU(3)xSU(2)xU(1)xSU(3)'),
]
IDS = ['{} > {}'.format(g, h) for g, h in EMBEDDINGS]


@pytest.fixture(scope='module', params=EMBEDDINGS, ids=IDS)
def any_embedding(request):
    return embedding(*request.param)


# ---- the Dynkin diagrams of the regular subalgebras ----

@pytest.mark.parametrize("name", ['A1', 'A5', 'B2', 'B4', 'C3', 'D4', 'D6', 'G2', 'F4', 'E6', 'E7', 'E8'])
def test_identify_own_diagram(name):
    g = lie_algebra(name)
    assert identify_diagram(g.cartan) == [(g.series, g.rank, tuple(range(g.rank)))]


def types(g, node):
    roots = extended_roots(g)
    kept = [roots[i] for i in range(g.rank + 1) if i != node]
    return sorted(s + str(r) for s, r, _ in identify_diagram(cartan_of_roots(g, kept)))


# Borel-de Siebenthal: deleting a node of the extended Dynkin diagram of E8, E7, E6, F4, G2
@pytest.mark.parametrize("name, expected", [
    ('E8', [['D8'], ['A8'], ['A1', 'A7'], ['A1', 'A2', 'A5'], ['A4', 'A4'], ['A3', 'D5'], ['A2', 'E6'], ['A1', 'E7']]),
    ('E7', [['A1', 'D6'], ['A7'], ['A2', 'A5'], ['A1', 'A3', 'A3'], ['A2', 'A5'], ['A1', 'D6'], ['E7']]),
    ('E6', [['E6'], ['A1', 'A5'], ['A1', 'A5'], ['A2', 'A2', 'A2'], ['A1', 'A5'], ['E6']]),
    ('F4', [['A1', 'C3'], ['A2', 'A2'], ['A1', 'A3'], ['B4']]),
    ('G2', [['A2'], ['A1', 'A1']]),
])
def test_borel_de_siebenthal(name, expected):
    g = lie_algebra(name)
    assert [types(g, node) for node in range(1, g.rank + 1)] == expected


# ---- consistency of every embedding ----

def test_dimensions_add_up(any_embedding):
    for irrep in small_irreps(any_embedding.parent[0]):
        assert sum(sub.dim * m for sub, m in branch(irrep, any_embedding)) == irrep.dim


def test_characters(any_embedding):
    # The projected weights of each irrep must be exactly the weights of the irreps it branches into: this fails
    # when the projection matrix is not the restriction of a real embedding (it is not Weyl-invariant then)
    for irrep in small_irreps(any_embedding.parent[0], limit=1000):
        projected = Counter()
        for w, m in irrep.weights().items():
            projected[any_embedding.project(w)] += m
        total = Counter()
        for sub, m in branch(irrep, any_embedding):
            for w, n in sub.weights().items():
                total[w] += m * n
        assert projected == total


def embedding_indices(e, irrep):
    # sum over the branching of the index of each factor (dim x charge^2 for a U(1)), divided by the index of irrep
    terms = branch(irrep, e)
    result = []
    for k, f in enumerate(e.factors):
        total = Fraction(0)
        for sub, m in terms:
            if is_u1(f):
                total += m * sub.dim * sub.parts[k] ** 2
            else:
                total += m * sub.dim // sub.parts[k].dim * dynkin_index(f, sub.parts[k].labels)
        result.append(total / dynkin_index(e.parent[0], irrep.labels))
    return tuple(result)


def test_index_sum_rule(any_embedding):
    # the embedding index of each factor is the same for every irrep
    irreps = [irrep for irrep in small_irreps(any_embedding.parent[0]) if irrep.dim > 1]
    assert len({embedding_indices(any_embedding, irrep) for irrep in irreps}) == 1


def test_conjugate_irreps_branch_to_conjugates(any_embedding):
    def conjugate(sub):
        return SubIrrep(sub.factors, [-p if is_u1(f) else p.conjugate() for f, p in zip(sub.factors, sub.parts)])

    for irrep in small_irreps(any_embedding.parent[0], limit=1000):
        assert Counter(dict(branch(irrep.conjugate(), any_embedding))) == \
            Counter({conjugate(sub): m for sub, m in branch(irrep, any_embedding)})


def test_adjoint_contains_subalgebra_adjoint(any_embedding):
    g = any_embedding.parent[0]
    terms = dict(branch(LieIrrep(g, adjoint_labels(g)), any_embedding))
    singlets = [Fraction(0) if is_u1(f) else LieIrrep(f, (0,) * f.rank) for f in any_embedding.factors]
    for k, f in enumerate(any_embedding.factors):
        if not is_u1(f):
            parts = list(singlets)
            parts[k] = LieIrrep(f, adjoint_labels(f))
            assert SubIrrep(any_embedding.factors, parts) in terms
    u1s = sum(1 for f in any_embedding.factors if is_u1(f))
    assert terms.get(SubIrrep(any_embedding.factors, singlets), 0) >= u1s


def product_of_subirreps(a, b):
    # (A1, ..., q) x (B1, ..., q') = sum of products of the factors, charges added (decomposed with lie_algebra_CG)
    terms = Counter({(): 1})
    for f, p, q in zip(a.factors, a.parts, b.parts):
        options = [(p + q, 1)] if is_u1(f) else decompose(p, q)
        terms = Counter({key + (x,): m * n for key, m in terms.items() for x, n in options})
    return Counter({SubIrrep(a.factors, key): m for key, m in terms.items()})


@pytest.mark.parametrize("group, subgroup, first, second", [
    ('SU(5)', 'SU(3)xSU(2)xU(1)', (1, 0, 0, 0), (0, 1, 0, 0)),
    ('SO(10)', 'SU(4)xSU(2)xSU(2)', (0, 0, 0, 1, 0), (0, 0, 0, 1, 0)),
    ('E6', 'SO(10)xU(1)', (1, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0)),
    ('E6', 'SU(3)xG2', (1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)),
    ('F4', 'SU(2)xG2', (0, 0, 0, 1), (0, 0, 0, 1)),
    ('E7', 'G2xSp(6)', (0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0, 1)),
    ('G2', 'SU(2)', (1, 0), (0, 1)),
])
def test_branching_commutes_with_tensor_products(group, subgroup, first, second):
    # branch(R1 x R2) = branch(R1) x branch(R2), with the tensor products done by lie_algebra_CG on both sides
    e = embedding(group, subgroup)
    g = e.parent[0]
    a, b = LieIrrep(g, first), LieIrrep(g, second)
    lhs = Counter()
    for irrep, m in decompose(a, b):
        for sub, n in branch(irrep, e):
            lhs[sub] += m * n
    rhs = Counter()
    for sa, m in branch(a, e):
        for sb, n in branch(b, e):
            for sub, k in product_of_subirreps(sa, sb).items():
                rhs[sub] += m * n * k
    assert lhs == rhs


# ---- embedding indices of the special subalgebras, from Dynkin's tables ----

@pytest.mark.parametrize("group, subgroup, indices", [
    ('E6', 'F4', (1,)), ('E6', 'Sp(8)', (1,)), ('E6', 'G2', (3,)), ('E6', 'SU(3)xG2', (2, 1)),
    ('E7', 'SU(2)xF4', (3, 1)), ('E7', 'G2xSp(6)', (1, 1)), ('E8', 'G2xF4', (1, 1)), ('F4', 'SU(2)xG2', (8, 1)),
    ('G2', 'SU(2)', (28,)), ('F4', 'SU(2)', (156,)), ('E6', 'SU(2)', (156,)), ('E7', 'SU(2)', (399,)),
    ('E8', 'SU(2)', (1240,)), ('SU(3)', 'SO(3)', (4,)), ('SO(7)', 'G2', (1,)), ('SU(6)', 'SU(3)xSU(2)', (2, 3)),
    ('E6', 'SO(10)xU(1)', (1, 12)), ('E8', 'SU(5)xSU(5)', (1, 1)),
])
def test_dynkin_embedding_index(group, subgroup, indices):
    e = embedding(group, subgroup)
    g = e.parent[0]
    assert embedding_indices(e, LieIrrep(g, adjoint_labels(g))) == indices


# Under the principal SU(2), the adjoint splits into spins equal to the exponents of the algebra
EXPONENTS = {'A4': [1, 2, 3, 4], 'B3': [1, 3, 5], 'C4': [1, 3, 5, 7], 'D5': [1, 3, 5, 7, 4], 'D4': [1, 3, 5, 3],
             'G2': [1, 5], 'F4': [1, 5, 7, 11], 'E6': [1, 4, 5, 7, 8, 11], 'E7': [1, 5, 7, 9, 11, 13, 17],
             'E8': [1, 7, 11, 13, 17, 19, 23, 29]}


@pytest.mark.parametrize("name", sorted(EXPONENTS))
def test_principal_su2_exponents(name):
    g = lie_algebra(name)
    terms = branch(LieIrrep(g, adjoint_labels(g)), principal_embedding(g))
    assert sorted(sub.dim for sub, m in terms for _ in range(m)) == sorted(2 * e + 1 for e in EXPONENTS[name])


# ---- known branching rules (Slansky, Phys. Rept. 79 (1981) 1, tables 15-58) ----

@pytest.mark.parametrize("group, subgroup, labels, expected", [
    # Georgi-Glashow SU(5), U(1) normalized as 6Y
    ('SU(5)', 'SU(3)xSU(2)xU(1)', (1, 0, 0, 0), ['(1, 2)(3)', '(3, 1)(-2)']),
    ('SU(5)', 'SU(3)xSU(2)xU(1)', (0, 0, 0, 1), ['(1, 2)(-3)', '(3bar, 1)(2)']),
    ('SU(5)', 'SU(3)xSU(2)xU(1)', (0, 1, 0, 0), ['(1, 1)(6)', '(3, 2)(1)', '(3bar, 1)(-4)']),
    ('SU(5)', 'SU(3)xSU(2)xU(1)', (1, 0, 0, 1),
     ['(1, 1)(0)', '(1, 3)(0)', '(3, 2)(-5)', '(3bar, 2)(5)', '(8, 1)(0)']),
    ('SU(5)', 'SU(4)xU(1)', (1, 0, 0, 0), ['1(4)', '4(-1)']),
    ('SU(6)', 'SU(5)xU(1)', (0, 1, 0, 0, 0), ['10(-2)', '5(4)']),
    ('SU(6)', 'SU(3)xSU(2)', (0, 0, 1, 0, 0), ['(1, 4)', '(8, 2)']),
    ('SU(6)', 'Sp(6)', (0, 0, 1, 0, 0), ["14'", '6']),
    ('SU(3)', 'SO(3)', (1, 1), ['3', '5']),
    # SO(10) GUTs
    ('SO(10)', 'SU(5)xU(1)', (1, 0, 0, 0, 0), ['5(2)', '5bar(-2)']),
    ('SO(10)', 'SU(5)xU(1)', (0, 0, 0, 1, 0), ['1(-5)', '10(-1)', '5bar(3)']),
    ('SO(10)', 'SU(5)xU(1)', (0, 1, 0, 0, 0), ['1(0)', '10(4)', '10bar(-4)', '24(0)']),
    ('SO(10)', 'SU(4)xSU(2)xSU(2)', (0, 0, 0, 1, 0), ['(4, 2, 1)', '(4bar, 1, 2)']),
    ('SO(10)', 'SU(4)xSU(2)xSU(2)', (1, 0, 0, 0, 0), ['(1, 2, 2)', '(6, 1, 1)']),
    ('SO(10)', 'SU(4)xSU(2)xSU(2)', (0, 1, 0, 0, 0), ['(1, 1, 3)', '(1, 3, 1)', '(15, 1, 1)', '(6, 2, 2)']),
    ('SO(10)', 'SU(4)xSU(2)xSU(2)', (0, 0, 0, 2, 0),
     ['(10, 3, 1)', '(10bar, 1, 3)', '(15, 2, 2)', '(6, 1, 1)']),            # the 126
    ('SO(10)', 'SO(9)', (0, 0, 0, 1, 0), ['16']),
    ('SO(9)', 'SO(8)', (0, 0, 0, 1), ["8'", "8''"]),
    ('SO(8)', 'SO(7)', (0, 0, 1, 0), ['8']),
    ('SO(7)', 'G2', (0, 0, 1), ['1', '7']),
    ('SO(7)', 'G2', (0, 1, 0), ['14', '7']),
    ('Sp(4)', 'SU(2)xSU(2)', (0, 1), ['(1, 1)', '(2, 2)']),
    ('Sp(6)', 'SU(2)xSU(2)', (1, 0, 0), ['(2, 3)']),
    # E6
    ('E6', 'SO(10)xU(1)', (1, 0, 0, 0, 0, 0), ['1(4)', '10(-2)', '16(1)']),
    ('E6', 'SO(10)xU(1)', (0, 1, 0, 0, 0, 0), ['1(0)', '16(-3)', '16bar(3)', '45(0)']),
    ('E6', 'SU(6)xSU(2)', (1, 0, 0, 0, 0, 0), ['(15, 1)', '(6bar, 2)']),
    ('E6', 'SU(6)xSU(2)', (0, 1, 0, 0, 0, 0), ['(1, 3)', '(20, 2)', '(35, 1)']),
    ('E6', 'SU(3)xSU(3)xSU(3)', (0, 1, 0, 0, 0, 0),
     ['(1, 1, 8)', '(1, 8, 1)', '(3, 3, 3bar)', '(3bar, 3bar, 3)', '(8, 1, 1)']),
    ('E6', 'F4', (1, 0, 0, 0, 0, 0), ['1', '26']),
    ('E6', 'F4', (0, 1, 0, 0, 0, 0), ['26', '52']),
    ('E6', 'Sp(8)', (1, 0, 0, 0, 0, 0), ['27']),
    ('E6', 'Sp(8)', (0, 1, 0, 0, 0, 0), ['36', '42']),
    ('E6', 'G2', (1, 0, 0, 0, 0, 0), ['27']),
    ('E6', 'G2', (0, 1, 0, 0, 0, 0), ['14', '64']),
    ('E6', 'SU(3)xG2', (1, 0, 0, 0, 0, 0), ['(3bar, 7)', '(6, 1)']),
    ('E6', 'SU(3)xG2', (0, 1, 0, 0, 0, 0), ['(1, 14)', '(8, 1)', '(8, 7)']),
    # E7
    ('E7', 'E6xU(1)', (0, 0, 0, 0, 0, 0, 1), ['1(-3)', '1(3)', '27(-1)', '27bar(1)']),
    ('E7', 'E6xU(1)', (1, 0, 0, 0, 0, 0, 0), ['1(0)', '27(2)', '27bar(-2)', '78(0)']),
    ('E7', 'SU(8)', (0, 0, 0, 0, 0, 0, 1), ['28', '28bar']),
    ('E7', 'SU(8)', (1, 0, 0, 0, 0, 0, 0), ['63', '70']),
    ('E7', 'SO(12)xSU(2)', (0, 0, 0, 0, 0, 0, 1), ['(12, 2)', '(32, 1)']),
    ('E7', 'SO(12)xSU(2)', (1, 0, 0, 0, 0, 0, 0), ['(1, 3)', "(32', 2)", '(66, 1)']),
    ('E7', 'SU(6)xSU(3)', (0, 0, 0, 0, 0, 0, 1), ['(20, 1)', '(6, 3)', '(6bar, 3bar)']),
    ('E7', 'SU(6)xSU(3)', (1, 0, 0, 0, 0, 0, 0), ['(1, 8)', '(15, 3bar)', '(15bar, 3)', '(35, 1)']),
    ('E7', 'SU(2)xF4', (0, 0, 0, 0, 0, 0, 1), ['(2, 26)', '(4, 1)']),
    ('E7', 'SU(2)xF4', (1, 0, 0, 0, 0, 0, 0), ['(1, 52)', '(3, 1)', '(3, 26)']),
    ('E7', 'G2xSp(6)', (0, 0, 0, 0, 0, 0, 1), ["(1, 14')", '(7, 6)']),
    ('E7', 'G2xSp(6)', (1, 0, 0, 0, 0, 0, 0), ['(1, 21)', '(14, 1)', '(7, 14)']),
    # E8
    ('E8', 'SO(16)', (0,) * 7 + (1,), ['120', '128']),
    ('E8', 'SO(16)', (1,) + (0,) * 7, ['135', '1820', "1920'"]),
    ('E8', 'SU(9)', (0,) * 7 + (1,), ['80', '84', '84bar']),
    ('E8', 'E7xSU(2)', (0,) * 7 + (1,), ['(1, 3)', '(133, 1)', '(56, 2)']),
    ('E8', 'E7xSU(2)', (1,) + (0,) * 7, ['(1, 1)', '(1539, 1)', '(133, 3)', '(56, 2)', '(912, 2)']),
    ('E8', 'E6xSU(3)', (0,) * 7 + (1,), ['(1, 8)', '(27, 3)', '(27bar, 3bar)', '(78, 1)']),
    ('E8', 'SU(5)xSU(5)', (0,) * 7 + (1,),
     ['(1, 24)', '(10, 5)', '(10bar, 5bar)', '(24, 1)', '(5, 10bar)', '(5bar, 10)']),
    ('E8', 'G2xF4', (0,) * 7 + (1,), ['(1, 52)', '(14, 1)', '(7, 26)']),
    # F4 and G2
    ('F4', 'SO(9)', (0, 0, 0, 1), ['1', '16', '9']),
    ('F4', 'SO(9)', (1, 0, 0, 0), ['16', '36']),
    ('F4', 'Sp(6)xSU(2)', (0, 0, 0, 1), ['(14, 1)', '(6, 2)']),
    ('F4', 'Sp(6)xSU(2)', (1, 0, 0, 0), ['(1, 3)', "(14', 2)", '(21, 1)']),
    ('F4', 'SU(3)xSU(3)', (0, 0, 0, 1), ['(3, 3)', '(3bar, 3bar)', '(8, 1)']),
    ('F4', 'SU(3)xSU(3)', (1, 0, 0, 0), ['(1, 8)', '(6, 3bar)', '(6bar, 3)', '(8, 1)']),
    ('F4', 'SU(2)xG2', (0, 0, 0, 1), ['(3, 7)', '(5, 1)']),
    ('F4', 'SU(2)xG2', (1, 0, 0, 0), ['(1, 14)', '(3, 1)', '(5, 7)']),
    ('G2', 'SU(3)', (1, 0), ['1', '3', '3bar']),
    ('G2', 'SU(3)', (0, 1), ['3', '3bar', '8']),
    ('G2', 'SU(2)xSU(2)', (1, 0), ['(1, 3)', '(2, 2)']),
    ('G2', 'SU(2)', (1, 0), ['7']),
])
def test_known_branchings(group, subgroup, labels, expected):
    assert branching_of(group, subgroup, labels) == sorted(expected)


def test_su_n_minus_one():
    # SU(N) > SU(N-1) x U(1): the Young diagram rule, e.g. the adjoint is (N-1)^2 - 1 + (N-1) + (N-1)bar + 1
    for n in range(3, 8):
        g = lie_algebra('SU({})'.format(n))
        e = embedding(g, 'SU({})xU(1)'.format(n - 1))
        terms = branch(LieIrrep(g, (1,) + (0,) * (n - 3) + (1,)), e)
        assert sorted(sub.dim for sub, m in terms) == sorted([(n - 1) ** 2 - 1, n - 1, n - 1, 1])


def test_so_n_minus_one_interlacing():
    # SO(2n+1) > SO(2n): the symmetric traceless tensors split as S^k = S^k + S^(k-1) + ... + S^0
    for name, sub in [('SO(7)', 'SO(6)'), ('SO(9)', 'SO(8)'), ('SO(11)', 'SO(10)')]:
        g = lie_algebra(name)
        e = embedding(g, sub)
        for k in range(4):
            labels = (k,) + (0,) * (g.rank - 1)
            assert sorted(s.parts[0].labels for s, m in branch(LieIrrep(g, labels), e)) == \
                sorted((j,) + (0,) * (g.rank - 1) for j in range(k + 1))


# ---- chains ----

def test_chain_equals_two_steps():
    # E6 > SO(10) x U(1) > SU(5) x U(1) x U(1), in one go and step by step
    chain = embedding('E6', 'SO(10)xU(1) > SU(5)xU(1)xU(1)')
    first = embedding('E6', 'SO(10)xU(1)')
    second = embedding('SO(10)', 'SU(5)xU(1)')
    for labels in [(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0)]:
        irrep = LieIrrep('E6', labels)
        steps = Counter()
        for sub, m in branch(irrep, first):
            so10, charge = sub.parts
            for subsub, n in branch(so10, second):
                su5, charge5 = subsub.parts
                steps[(su5.labels, charge5, charge)] += m * n
        direct = Counter({(s.parts[0].labels, s.parts[1], s.parts[2]): m for s, m in branch(irrep, chain)})
        assert direct == steps


def test_standard_model_from_so10():
    terms = branching_of('SO(10)', 'SU(5)xU(1) > SU(3)xSU(2)xU(1)xU(1)', (0, 0, 0, 1, 0))
    assert terms == sorted(['(3, 2)(1)(-1)', '(3bar, 1)(-4)(-1)', '(1, 1)(6)(-1)', '(3bar, 1)(2)(3)',
                            '(1, 2)(-3)(3)', '(1, 1)(0)(-5)'])


# ---- names and errors ----

def test_parse_factors():
    assert [f.name for f in parse_factors('SU(3) x SU(2) × U(1)')] == ['SU(3)', 'SU(2)', 'U(1)']
    assert [f.name for f in parse_factors('SO(4)xSO(2)')] == ['SU(2)', 'SU(2)', 'U(1)']
    assert [f.name for f in parse_factors('SO(3)*Sp(4)')] == ['SU(2)', 'Sp(4)']


def test_isomorphic_names():
    # SO(6) = SU(4) and SO(5) = Sp(4): the factor comes out under the name that was asked for
    assert branching_of('SO(10)', 'SO(6)xSU(2)xSU(2)', (1, 0, 0, 0, 0)) == ['(1, 2, 2)', '(6, 1, 1)']
    assert embedding('SO(10)', 'SO(6)xSU(2)xSU(2)').factors[0] == lie_algebra('SO(6)')
    assert branching_of('Sp(4)', 'SO(5)', (1, 0)) == ['4']


def test_factor_order_follows_request():
    assert branching_of('E8', 'SU(2)xE7', (0,) * 7 + (1,)) == ['(1, 133)', '(2, 56)', '(3, 1)']
    assert branching_of('SU(5)', 'U(1)xSU(2)xSU(3)', (1, 0, 0, 0)) == ['(1, 3)(-2)', '(2, 1)(3)']


@pytest.mark.parametrize("group, subgroup", [('SU(5)', 'E6'), ('SU(3)', 'SU(2)xSU(2)'), ('G2', 'SU(3)xU(1)'),
                                             ('SO(10)', 'SU(5)xU(1) > E6'), ('SU(5)', 'SU(3)xFoo')])
def test_unknown_subgroups(group, subgroup):
    with pytest.raises(ValueError):
        embedding(group, subgroup)


def test_bad_projection_is_caught():
    # an orthogonal irrep cannot be used for Sp(N), and the U(1) charges must add up to zero
    with pytest.raises(ValueError):
        defining_rep_embedding(lie_algebra('SU(3)'), [A1, U1()], [[(1,), Fraction(1)], [(0,), Fraction(1)]])
    with pytest.raises(ValueError):
        defining_rep_embedding(lie_algebra('Sp(4)'), [A1], [[(3,)], [(0,)]])


def test_number_of_weights():
    for name, labels in [('E6', (1, 0, 0, 0, 0, 0)), ('E8', (0,) * 7 + (1,)), ('SO(10)', (0, 1, 1, 0, 0)),
                         ('G2', (2, 1))]:
        g = lie_algebra(name)
        assert number_of_weights(g, labels) == len(LieIrrep(g, labels).weights())
    with pytest.raises(ValueError):
        branch(LieIrrep('E8', (1,) * 8), embedding('E8', 'SO(16)'), max_weights=10 ** 6)


def test_dimension_names():
    assert [dimension_name(LieIrrep('SU(3)', x)) for x in [(2, 1), (1, 2), (4, 0), (0, 4)]] == \
        ['15', '15bar', "15'", "15'bar"]
    assert [dimension_name(LieIrrep('SO(8)', x)) for x in [(1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]] == \
        ['8', "8'", "8''"]
    assert dimension_name(LieIrrep('Sp(6)', (0, 0, 1))) == "14'"


def test_describe_and_format():
    e = embedding('SU(5)', 'SU(3)xSU(2)xU(1)')
    assert describe(e) == ['5 = (3, 1)(-2) + (1, 2)(3)', '10 = (3, 2)(1) + (3bar, 1)(-4) + (1, 1)(6)']
    irrep = LieIrrep('E8', (0,) * 7 + (1,))
    assert format_branching(irrep, branch(irrep, embedding('E8', 'SU(2)'))) == \
        '248 = 59 + 47 + 39 + 35 + 27 + 23 + 15 + 3'
