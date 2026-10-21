# Clebsch-Gordan decomposition of tensor products of SU(N) irreps (= irreducible representations), for any N >= 2.
# Generalizes su3_irrep_CG.py: the 'a', 'b', 'c', ... boxes of Georgi's method are the labels 1, 2, 3, ... of the
# Littlewood-Richardson rule, so the same Young tableau bookkeeping works for any number of rows.

from collections import Counter
from math import prod


class IrrepSUN():  # A class object for an SU(N) irrep, labelled by its N - 1 Dynkin coefficients
    def __init__(self, n, labels):
        labels = tuple(int(x) for x in labels)
        if n < 2:
            raise ValueError("SU(N) needs N >= 2, got N = {}".format(n))
        if len(labels) != n - 1:
            raise ValueError("SU({}) irreps need {} Dynkin coefficients, got {}".format(n, n - 1, len(labels)))
        if any(x < 0 for x in labels):
            raise ValueError("Dynkin coefficients must be nonnegative, got {}".format(labels))
        self.n = n
        self.labels = labels

    @classmethod
    def from_partition(cls, n, partition):
        # Young diagram with row lengths 'partition' (at most N rows); full columns of height N are dropped
        rows = [int(x) for x in partition if x]
        if len(rows) > n:
            raise ValueError("SU({}) Young diagrams have at most {} rows, got {}".format(n, n, len(rows)))
        if any(rows[i] < rows[i + 1] for i in range(len(rows) - 1)):
            raise ValueError("Row lengths must be weakly decreasing, got {}".format(tuple(rows)))
        rows += [0] * (n - len(rows))
        return cls(n, [rows[i] - rows[i + 1] for i in range(n - 1)])

    @property
    def partition(self):
        # Row lengths of the Young diagram (no full columns, trailing zeros dropped)
        rows = [sum(self.labels[i:]) for i in range(self.n - 1)]
        return tuple(r for r in rows if r)

    @property
    def n_boxes(self):
        return sum(self.partition)

    @property
    def dim(self):
        # Weyl dimension formula for SU(N), written with row lengths l_1 >= ... >= l_N = 0
        rows = list(self.partition) + [0] * (self.n - len(self.partition))
        pairs = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        return prod(rows[i] - rows[j] + j - i for i, j in pairs) // prod(j - i for i, j in pairs)

    # Same convention as su3_irrep_CG.Irrep, where (n, m) is 'bar' when n < m
    @property
    def bar(self):
        return self.labels < self.labels[::-1]

    def conjugate(self):
        return IrrepSUN(self.n, self.labels[::-1])

    def young_diagram(self, box='□'):
        return '\n'.join(box * r for r in self.partition)

    def __eq__(self, other):
        return isinstance(other, IrrepSUN) and (self.n, self.labels) == (other.n, other.labels)

    def __hash__(self):
        return hash((self.n, self.labels))

    def __repr__(self):
        display_dim = str(self.dim)
        if self.bar:
            display_dim += ' bar'
        return "({idx})[{dim}]".format(idx=', '.join(str(x) for x in self.labels), dim=display_dim)


# All ways to add a horizontal strip (no two new boxes in the same column) of 'size' boxes to the Young diagram
# 'shape' without going past 'max_rows' rows. Yields the number of new boxes in each row.
# If 'previous' (boxes per row of the previous label) is given, only strips that keep the lattice condition are kept:
# in each row the new label is read before the previous one, so the new boxes up to row r cannot outnumber the
# previous label's boxes strictly above row r (Georgi's "b_count <= a_count" check).
def horizontal_strips(shape, size, max_rows, previous=None):
    rows = list(shape) + ([0] if len(shape) < max_rows else [])

    def fill(r, remaining, above, seen):
        if r == len(rows):
            if remaining == 0:
                yield ()
            return
        room = remaining if r == 0 else min(remaining, rows[r - 1] - rows[r])
        if previous is not None:
            room = min(room, above - seen)
        next_above = above + (previous[r] if previous is not None and r < len(previous) else 0)
        for t in range(room, -1, -1):
            for rest in fill(r + 1, remaining - t, next_above, seen + t):
                yield (t,) + rest

    yield from fill(0, size, 0, 0)


# Littlewood-Richardson rule: add mu[0] boxes labelled 1, then mu[1] boxes labelled 2, ... to the diagram 'lam'.
# Each label forms a horizontal strip, and reading right to left, top to bottom, every prefix must contain at least as
# many (k-1)'s as k's. Returns a Counter {resulting row lengths: multiplicity}.
# What can still happen only depends on the current shape and on where the last label went, so the fillings are
# grouped by that state, one label at a time, and only the number of fillings in each state is kept.
def littlewood_richardson(lam, mu, max_rows):
    states = Counter({(tuple(lam), None): 1})
    for size in mu:
        new_states = Counter()
        for (shape, previous), mult in states.items():
            for strip in horizontal_strips(shape, size, max_rows, previous):
                new_shape = tuple(r + t for r, t in zip(list(shape) + [0], strip) if r + t)
                new_states[(new_shape, strip)] += mult
        states = new_states
    results = Counter()
    for (shape, _), mult in states.items():
        results[shape] += mult
    return results


# Tensor "multiply" 2 SU(N) irreps. Returns a list of (irrep, multiplicity), largest dimension first
def decompose(irrep1, irrep2):
    if irrep1.n != irrep2.n:
        raise ValueError("Cannot multiply an SU({}) irrep by an SU({}) irrep".format(irrep1.n, irrep2.n))
    # The work grows quickly with the number of boxes in the diagram that is cut into strips ('small'). The product is
    # commutative, and conjugating both factors conjugates the result, so pick whichever of the 4 equivalent products
    # has the fewest boxes there: e.g. in SU(N) a 10bar has many more boxes than a 10.
    options = []
    for conjugated in (False, True):
        a, b = (irrep1.conjugate(), irrep2.conjugate()) if conjugated else (irrep1, irrep2)
        options += [(b.n_boxes, a.n_boxes, conjugated, a, b), (a.n_boxes, b.n_boxes, conjugated, b, a)]
    _, _, conjugated, big, small = min(options, key=lambda option: option[:3])
    shapes = littlewood_richardson(big.partition, small.partition, big.n)
    terms = Counter()
    for shape, mult in shapes.items():
        irrep = IrrepSUN.from_partition(big.n, shape)
        terms[irrep.conjugate() if conjugated else irrep] += mult
    return sorted(terms.items(), key=lambda item: (-item[0].dim, tuple(-x for x in item[0].labels)))


# Same as decompose, but every irrep is repeated according to its multiplicity (like su3_irrep_CG.multiply_irreps)
def multiply_irreps(irrep1, irrep2):
    return [irrep for irrep, mult in decompose(irrep1, irrep2) for _ in range(mult)]


# Decompose a product of any number of irreps, one factor at a time
def decompose_many(irreps):
    irreps = list(irreps)
    if not irreps:
        raise ValueError("Need at least one irrep")
    terms = Counter({irreps[0]: 1})
    for factor in irreps[1:]:
        new_terms = Counter()
        for irrep, mult in terms.items():
            for result, m in decompose(irrep, factor):
                new_terms[result] += mult * m
        terms = new_terms
    return sorted(terms.items(), key=lambda item: (-item[0].dim, tuple(-x for x in item[0].labels)))


# Human readable equation, e.g. "(1, 0)[3] ⊗ (0, 1)[3 bar] = (1, 1)[8] ⊕ (0, 0)[1]"
def format_product(irreps, terms):
    lhs = ' ⊗ '.join(str(irrep) for irrep in irreps)
    rhs = ' ⊕ '.join((str(mult) + ' ' if mult > 1 else '') + str(irrep) for irrep, mult in terms)
    return lhs + ' = ' + rhs
