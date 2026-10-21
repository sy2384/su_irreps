# Independent checks of lie_algebra_CG.py and sun_irrep_CG.py against programs and published tables that share no
# code with them. The data files in tests/data are made by tools/make_reference_data.py:
#   - lieart_tables.json: the LieART 2.0 tables of irreps and tensor products (Feger, Kephart, Saskowski,
#     Comput. Phys. Commun. 257 (2020) 107490, arXiv:1912.10969, appendix), which name irreps by dimension
#   - lie_reference.json: dimensions, conjugates, dominant weight multiplicities and tensor products from LiE 2.2.2
#   - gap_reference.json: dimensions and tensor products from GAP 4.12
# Each reference numbers the simple roots its own way, so its Dynkin labels are mapped to Bourbaki's first.
from collections import Counter
import itertools
import json
import os
import re

import pytest

from lie_algebra_CG import LieIrrep, lie_algebra, decompose, decompose_many
from sun_irrep_CG import IrrepSUN, decompose as decompose_sun

DATA = os.path.join(os.path.dirname(__file__), 'data')


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


LIEART = load('lieart_tables.json')
LIE = load('lie_reference.json')
GAP = load('gap_reference.json')


def as_counter(terms):
    return Counter({irrep.labels: mult for irrep, mult in terms})


def sun_rank(name):
    m = re.match(r'(?:SU\(|A)(\d+)', name)
    return m and (int(m.group(1)) - 1 if name.startswith('SU') else int(m.group(1)))


# ---- LieART ----

# LieART numbers E_n like Dynkin: a chain 1, 2, ..., n-1 with node n attached to node 3 (to node 5 for E8, counted
# from the other end), while Bourbaki's chain is 1, 3, 4, ..., n with node 2 attached to node 4. All other algebras,
# G2 and F4 included, are numbered as in Bourbaki.
def lieart_to_bourbaki(algebra, labels):
    if algebra.startswith('E'):
        return (labels[0], labels[-1]) + tuple(labels[1:-1])
    return tuple(labels)


# Irrep names of each algebra, e.g. "351bar'" -> Dynkin labels (Bourbaki). LieART lists one irrep of each conjugate
# pair; the conjugate of an irrep named X is named Xbar (or X without the bar).
def lieart_names(algebra):
    g = lie_algebra(algebra)
    names, clashes = {}, set()
    for labels, name in LIEART['irreps'][algebra]:
        labels = lieart_to_bourbaki(algebra, labels)
        conjugate = LieIrrep(g, labels).conjugate().labels
        other = name.replace('bar', '') if 'bar' in name else re.sub(r'^(\d+)', r'\1bar', name)
        for n, l in [(name, labels)] + ([(other, conjugate)] if conjugate != labels else []):
            if names.get(n, l) != l:
                clashes.add(n)  # LieART gives Sp(8)'s (4,0,0,1) and (0,1,0,2) the same name, 9009
            names[n] = l
    for n in clashes:
        del names[n]
    return names


def name_dim(name):
    return int(re.match(r'\d+', name).group())


@pytest.mark.parametrize("algebra", sorted(LIEART['irreps']))
def test_lieart_irrep_dimensions(algebra):
    g = lie_algebra(algebra)
    rows = LIEART['irreps'][algebra]
    assert rows
    wrong = [(labels, name) for labels, name in rows
             if LieIrrep(g, lieart_to_bourbaki(algebra, labels)).dim != name_dim(name)]
    assert wrong == []
    if sun_rank(algebra):
        assert all(IrrepSUN(g.rank + 1, labels).dim == name_dim(name) for labels, name in rows)


@pytest.mark.parametrize("algebra", sorted(LIEART['tensor_products']))
def test_lieart_tensor_products(algebra):
    g = lie_algebra(algebra)
    names = lieart_names(algebra)
    checked = 0
    for a, b, terms in LIEART['tensor_products'][algebra]:
        if a not in names or b not in names:
            continue  # a factor outside LieART's table of irreps; there are a few in the larger algebras
        irrep1, irrep2 = LieIrrep(g, names[a]), LieIrrep(g, names[b])
        result = as_counter(decompose(irrep1, irrep2))
        # every term by dimension, and by Dynkin labels whenever LieART's table of irreps names it
        assert sorted(LieIrrep(g, l).dim for l, m in result.items() for _ in range(m)) == \
            sorted(name_dim(n) for m, n in terms for _ in range(m)), (a, b)
        named = Counter()
        for m, n in terms:
            if n in names:
                named[names[n]] += m
        assert all(result[labels] == m for labels, m in named.items()), (a, b)
        if sun_rank(algebra):
            sun = decompose_sun(IrrepSUN(g.rank + 1, names[a]), IrrepSUN(g.rank + 1, names[b]))
            assert as_counter(sun) == result
        checked += 1
    assert checked >= len(LIEART['tensor_products'][algebra]) // 2


@pytest.mark.parametrize("algebra, lieart, name, bourbaki", [
    ('E6', (1, 0, 0, 0, 0, 0), '27', (1, 0, 0, 0, 0, 0)),
    ('E6', (0, 0, 0, 0, 0, 1), '78', (0, 1, 0, 0, 0, 0)),
    ('E6', (0, 0, 0, 1, 0, 0), '351', (0, 0, 0, 0, 1, 0)),
    ('E6', (0, 0, 0, 0, 2, 0), "351'", (0, 0, 0, 0, 0, 2)),
    ('E7', (0, 0, 0, 0, 0, 1, 0), '56', (0, 0, 0, 0, 0, 0, 1)),
    ('E7', (1, 0, 0, 0, 0, 0, 0), '133', (1, 0, 0, 0, 0, 0, 0)),
    ('E7', (0, 0, 0, 0, 0, 0, 1), '912', (0, 1, 0, 0, 0, 0, 0)),
    ('E8', (0, 0, 0, 0, 0, 0, 1, 0), '248', (0, 0, 0, 0, 0, 0, 0, 1)),
    ('E8', (1, 0, 0, 0, 0, 0, 0, 0), '3875', (1, 0, 0, 0, 0, 0, 0, 0)),
    ('E8', (0, 0, 0, 0, 0, 0, 0, 1), '147250', (0, 1, 0, 0, 0, 0, 0, 0)),
    ('F4', (0, 0, 0, 1), '26', (0, 0, 0, 1)),
    ('G2', (1, 0), '7', (1, 0)),
    ('SO(10)', (0, 0, 0, 0, 1), '16', (0, 0, 0, 0, 1)),
])
def test_lieart_numbering(algebra, lieart, name, bourbaki):
    assert [lieart] == [tuple(l) for l, n in LIEART['irreps'][algebra] if n == name]
    assert lieart_to_bourbaki(algebra, lieart) == bourbaki
    assert LieIrrep(algebra, bourbaki).dim == name_dim(name)


def test_lieart_paper_examples():
    # Quoted in the text of arXiv:1912.10969: 27 x 27bar = 1 + 78 + 650 and 27^3 = 1 + 2(78) + 3(650) + 2925 + 3003
    # + 2(5824)
    d27 = LieIrrep('E6', (1, 0, 0, 0, 0, 0))
    assert sorted(i.dim for i, m in decompose(d27, d27.conjugate()) for _ in range(m)) == [1, 78, 650]
    assert sorted(i.dim for i, m in decompose_many([d27] * 3) for _ in range(m)) == \
        [1, 78, 78, 650, 650, 650, 2925, 3003, 5824, 5824]


# ---- LiE (Bourbaki numbering, like this repository) ----

LIE_ALGEBRAS = sorted(LIE['dimensions'])


@pytest.mark.parametrize("algebra", LIE_ALGEBRAS)
def test_lie_dimensions_and_conjugates(algebra):
    g = lie_algebra(algebra)
    for labels, dim, conjugate in LIE['dimensions'][algebra]:
        irrep = LieIrrep(g, labels)
        assert (irrep.dim, irrep.conjugate().labels) == (dim, tuple(conjugate)), labels
        if sun_rank(algebra):
            assert IrrepSUN(g.rank + 1, labels).dim == dim


@pytest.mark.parametrize("algebra", LIE_ALGEBRAS)
def test_lie_dominant_weight_multiplicities(algebra):
    g = lie_algebra(algebra)
    for labels, character in LIE['dominant_characters'][algebra]:
        assert g.dominant_multiplicities(labels) == {tuple(w): m for w, m in character}, labels


@pytest.mark.parametrize("algebra", LIE_ALGEBRAS)
def test_lie_tensor_products(algebra):
    g = lie_algebra(algebra)
    for a, b, terms in LIE['tensor_products'][algebra]:
        expected = Counter({tuple(l): m for l, m in terms})
        assert as_counter(decompose(LieIrrep(g, a), LieIrrep(g, b))) == expected, (a, b)
        if sun_rank(algebra):
            assert as_counter(decompose_sun(IrrepSUN(g.rank + 1, a), IrrepSUN(g.rank + 1, b))) == expected


# ---- GAP ----

# GAP's numbering of the simple roots, found by matching its Cartan matrix: node i here is node perm[i] in GAP
def gap_permutation(algebra):
    g = lie_algebra(algebra)
    (cartan,) = GAP['cartan_matrices'][algebra]
    n = g.rank
    return next(p for p in itertools.permutations(range(n))
                if all(cartan[p[i]][p[j]] == g.cartan[i][j] for i in range(n) for j in range(n)))


def gap_to_bourbaki(algebra, labels):
    return tuple(labels[p] for p in gap_permutation(algebra))


def test_gap_f4_numbering():
    # GAP numbers F4 so that its fundamental irrep (1,0,0,0) is the 26, which is (0,0,0,1) in Bourbaki's numbering
    assert gap_permutation('F4') == (1, 3, 2, 0)
    assert gap_to_bourbaki('F4', (1, 0, 0, 0)) == (0, 0, 0, 1)
    assert [tuple(l) for l, d in GAP['dimensions']['F4'] if d == 26] == [(1, 0, 0, 0)]
    assert all(gap_permutation(name) == tuple(range(lie_algebra(name).rank)) for name in GAP['cartan_matrices']
               if name != 'F4')


@pytest.mark.parametrize("algebra", sorted(GAP['tensor_products']))
def test_gap(algebra):
    g = lie_algebra(algebra)
    for labels, dim in GAP['dimensions'][algebra]:
        assert LieIrrep(g, gap_to_bourbaki(algebra, labels)).dim == dim
    for a, b, terms in GAP['tensor_products'][algebra]:
        result = decompose(LieIrrep(g, gap_to_bourbaki(algebra, a)), LieIrrep(g, gap_to_bourbaki(algebra, b)))
        assert as_counter(result) == Counter({gap_to_bourbaki(algebra, l): m for l, m in terms}), (a, b)
