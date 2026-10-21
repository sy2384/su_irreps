# Clebsch-Gordan decomposition of tensor products of irreps of any simple Lie algebra:
# A_n = SU(n+1), B_n = SO(2n+1), C_n = Sp(2n), D_n = SO(2n), and the exceptional G2, F4, E6, E7, E8.
# Young tableaux only work for SU(N), so this uses weights instead:
#   - the weights of an irrep and their multiplicities come from Freudenthal's formula,
#   - the product is decomposed with the Racah-Speiser (Brauer-Klimyk) algorithm: add each weight of one factor to the
#     highest weight of the other and reflect the result into the dominant Weyl chamber.
# Irreps are labelled by Dynkin coefficients in the Bourbaki numbering of the simple roots.

from collections import Counter
from fractions import Fraction
from functools import lru_cache
from math import lcm
import re


# Simple roots as vectors in an orthonormal basis (Bourbaki, Planches I-IX). Only their inner products matter.
def simple_root_vectors(series, rank):
    def e(i, size, coefficient=1):
        v = [Fraction(0)] * size
        v[i] = Fraction(coefficient)
        return v

    def add(*vectors):
        return [sum(xs, Fraction(0)) for xs in zip(*vectors)]

    def scale(c, v):
        return [Fraction(c) * x for x in v]

    n = rank
    if series == 'A' and n >= 1:
        return [add(e(i, n + 1), e(i + 1, n + 1, -1)) for i in range(n)]
    if series == 'B' and n >= 2:
        return [add(e(i, n), e(i + 1, n, -1)) for i in range(n - 1)] + [e(n - 1, n)]
    if series == 'C' and n >= 2:
        return [add(e(i, n), e(i + 1, n, -1)) for i in range(n - 1)] + [e(n - 1, n, 2)]
    if series == 'D' and n >= 3:
        return [add(e(i, n), e(i + 1, n, -1)) for i in range(n - 1)] + [add(e(n - 2, n), e(n - 1, n))]
    if series == 'G' and n == 2:
        return [[Fraction(1), Fraction(-1), Fraction(0)], [Fraction(-2), Fraction(1), Fraction(1)]]
    if series == 'F' and n == 4:
        half = Fraction(1, 2)
        return [add(e(1, 4), e(2, 4, -1)), add(e(2, 4), e(3, 4, -1)), e(3, 4),
                [half, -half, -half, -half]]
    if series == 'E' and n in (6, 7, 8):
        half = Fraction(1, 2)
        roots = [[half, -half, -half, -half, -half, -half, -half, half], add(e(0, 8), e(1, 8))]
        roots += [add(e(i - 2, 8), e(i - 3, 8, -1)) for i in range(3, 9)]
        return roots[:n]
    raise ValueError("There is no simple Lie algebra {}{}".format(series, rank))


def dot(u, v):
    return sum((a * b for a, b in zip(u, v)), Fraction(0))


def invert(matrix):
    # Gauss-Jordan elimination over the rationals
    n = len(matrix)
    m = [[Fraction(x) for x in row] + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next(r for r in range(col, n) if m[r][col] != 0)
        m[col], m[pivot] = m[pivot], m[col]
        p = m[col][col]
        m[col] = [x / p for x in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    return [row[n:] for row in m]


class SimpleLieAlgebra():
    def __init__(self, series, rank):
        self.series = series
        self.rank = rank
        roots = simple_root_vectors(series, rank)
        products = [[dot(a, b) for b in roots] for a in roots]
        # Cartan matrix: row i holds the Dynkin labels of the simple root alpha_i
        self.cartan = tuple(tuple(int(2 * products[i][j] / products[j][j]) for j in range(rank)) for i in range(rank))
        # Inner products of the fundamental weights, scaled to integers (only ratios of inner products are used)
        inverse = invert(self.cartan)
        quadratic = [[sum(inverse[j][i] * inverse[k][l] * products[i][l] for i in range(rank) for l in range(rank))
                      for k in range(rank)] for j in range(rank)]
        factor = lcm(*(x.denominator for row in quadratic for x in row))
        self.metric = tuple(tuple(int(x * factor) for x in row) for row in quadratic)
        self.positive_roots = self._positive_roots()
        self.rho = (1,) * rank

    @property
    def name(self):
        n = self.rank
        physics = {'A': 'SU({})'.format(n + 1), 'B': 'SO({})'.format(2 * n + 1), 'C': 'Sp({})'.format(2 * n),
                   'D': 'SO({})'.format(2 * n)}
        return physics.get(self.series, self.series + str(n))

    def __repr__(self):
        return "{}{} = {}".format(self.series, self.rank, self.name)

    def __eq__(self, other):
        return isinstance(other, SimpleLieAlgebra) and (self.series, self.rank) == (other.series, other.rank)

    def __hash__(self):
        return hash((self.series, self.rank))

    # Positive roots in Dynkin coordinates, built up by height: for a root beta and simple root alpha_i, the
    # alpha_i-string through beta goes from beta - p alpha_i to beta + q alpha_i with p - q = <beta, alpha_i>
    def _positive_roots(self):
        n = self.rank
        simple = [tuple(int(i == j) for j in range(n)) for i in range(n)]
        found = set(simple)
        layer = list(simple)
        while layer:
            next_layer = []
            for beta in layer:
                labels = self.root_to_weight(beta)
                for i in range(n):
                    p = 0
                    lower = list(beta)
                    while True:
                        lower[i] -= 1
                        if tuple(lower) in found:
                            p += 1
                        else:
                            break
                    if p - labels[i] > 0:
                        higher = list(beta)
                        higher[i] += 1
                        higher = tuple(higher)
                        if higher not in found:
                            found.add(higher)
                            next_layer.append(higher)
            layer = next_layer
        return tuple(sorted((self.root_to_weight(beta) for beta in found), key=lambda w: w, reverse=True))

    def root_to_weight(self, coefficients):
        # sum_i c_i alpha_i in Dynkin coordinates
        return tuple(sum(c * self.cartan[i][j] for i, c in enumerate(coefficients)) for j in range(self.rank))

    def inner(self, u, v):
        m = self.metric
        return sum(u[j] * m[j][k] * v[k] for j in range(self.rank) for k in range(self.rank) if u[j] and v[k])

    def reflect(self, weight, i):
        c = weight[i]
        row = self.cartan[i]
        return tuple(w - c * a for w, a in zip(weight, row))

    # Reflect 'weight' into the dominant chamber; returns (dominant weight, number of reflections used)
    def dominant(self, weight):
        steps = 0
        weight = tuple(weight)
        while True:
            i = next((j for j, x in enumerate(weight) if x < 0), None)
            if i is None:
                return weight, steps
            weight = self.reflect(weight, i)
            steps += 1

    # Like dominant, for the Racah-Speiser step: returns None as soon as a coordinate is 0. Then a simple
    # reflection fixes the weight, so its dominant image lies on a wall too and it contributes nothing.
    def strictly_dominant(self, weight):
        weight = list(weight)
        rows = self.cartan
        steps = 0
        while True:
            i = -1
            for j, x in enumerate(weight):
                if x <= 0:
                    if x == 0:
                        return None
                    if i < 0:
                        i = j
            if i < 0:
                return tuple(weight), steps
            c = weight[i]
            weight = [w - c * a for w, a in zip(weight, rows[i])]
            steps += 1

    # All elements of the Weyl group orbit of a dominant weight, each exactly once, without remembering them:
    # every other element v has a unique parent s_j v, where j is the first negative coordinate of v
    def orbit(self, weight):
        rows = self.cartan
        rank = self.rank
        stack = [tuple(weight)]
        while stack:
            u = stack.pop()
            yield u
            for i in range(rank):
                c = u[i]
                if c > 0:
                    row = rows[i]
                    v = tuple(x - c * a for x, a in zip(u, row))
                    # v_i = -c < 0, so i must be the first negative coordinate of v
                    if all(v[k] >= 0 for k in range(i)):
                        stack.append(v)

    # Weyl dimension formula: product over positive roots of (lambda + rho, alpha) / (rho, alpha)
    def dimension(self, labels):
        shifted = tuple(x + 1 for x in labels)
        numerator = 1
        denominator = 1
        for alpha in self.positive_roots:
            numerator *= self.inner(shifted, alpha)
            denominator *= self.inner(self.rho, alpha)
        return numerator // denominator

    # Dominant weights of the irrep with highest weight 'labels' and their multiplicities (Freudenthal's formula).
    # Dominant weights below a dominant weight are reached by subtracting positive roots (Stembridge).
    def dominant_multiplicities(self, labels):
        highest = tuple(labels)
        depth = {highest: 0}
        layer = [highest]
        heights = [sum(self.weight_to_root(alpha)) for alpha in self.positive_roots]
        while layer:
            next_layer = []
            for mu in layer:
                for alpha, h in zip(self.positive_roots, heights):
                    nu = tuple(a - b for a, b in zip(mu, alpha))
                    if min(nu) >= 0 and nu not in depth:
                        depth[nu] = depth[mu] + h
                        next_layer.append(nu)
            layer = next_layer
        top = tuple(x + 1 for x in highest)
        top_norm = self.inner(top, top)
        mults = {}
        for mu in sorted(depth, key=depth.get):
            if mu == highest:
                mults[mu] = 1
                continue
            total = 0
            for alpha in self.positive_roots:
                k = 1
                while True:
                    nu = tuple(a + k * b for a, b in zip(mu, alpha))
                    m = mults.get(self.dominant(nu)[0], 0)
                    if m == 0:
                        break
                    total += m * self.inner(nu, alpha)
                    k += 1
            shifted = tuple(x + 1 for x in mu)
            mults[mu] = 2 * total // (top_norm - self.inner(shifted, shifted))
        return {mu: m for mu, m in mults.items() if m}

    def weight_to_root(self, weight):
        # inverse of root_to_weight; integer for elements of the root lattice
        inverse = self._cartan_inverse()
        coefficients = [sum(weight[j] * inverse[j][i] for j in range(self.rank)) for i in range(self.rank)]
        return tuple(int(c) for c in coefficients)

    def _cartan_inverse(self):
        if not hasattr(self, '_inverse'):
            self._inverse = invert(self.cartan)
        return self._inverse

    # All weights with multiplicities, as a Counter
    def weights(self, labels):
        result = Counter()
        for mu, m in self.dominant_multiplicities(labels).items():
            for w in self.orbit(mu):
                result[w] += m
        return result


ALGEBRA_PATTERN = re.compile(r'^\s*(?:(SU|SO|SP|USP)\s*\(?\s*(\d+)\s*\)?|([A-G])\s*_?\s*(\d+))\s*$', re.IGNORECASE)


# Parse 'SU(3)', 'SO(10)', 'Sp(4)', 'A2', 'E6', 'G_2', ...
@lru_cache(maxsize=None)
def lie_algebra(name):
    match = ALGEBRA_PATTERN.match(name)
    if not match:
        raise ValueError("Unknown Lie algebra '{}'; try e.g. SU(3), SO(10), Sp(4), G2, F4, E6".format(name))
    if match.group(1):
        group, n = match.group(1).upper(), int(match.group(2))
        if group == 'SU' and n >= 2:
            return SimpleLieAlgebra('A', n - 1)
        if group == 'SO' and n >= 5 and n % 2 == 1:
            return SimpleLieAlgebra('B', (n - 1) // 2)
        if group == 'SO' and n >= 6 and n % 2 == 0:
            return SimpleLieAlgebra('D', n // 2)
        if group in ('SP', 'USP') and n >= 4 and n % 2 == 0:
            return SimpleLieAlgebra('C', n // 2)
        raise ValueError("{}({}) is not a simple Lie group handled here (SU(N) needs N >= 2, SO(N) N >= 5, "
                         "Sp(N) even N >= 4)".format(match.group(1), n))
    return SimpleLieAlgebra(match.group(3).upper(), int(match.group(4)))


class LieIrrep():  # An irrep of a simple Lie algebra, labelled by its Dynkin coefficients
    def __init__(self, algebra, labels):
        if isinstance(algebra, str):
            algebra = lie_algebra(algebra)
        labels = tuple(int(x) for x in labels)
        if len(labels) != algebra.rank:
            raise ValueError("{} irreps need {} Dynkin coefficients, got {}".format(algebra.name, algebra.rank,
                                                                                 len(labels)))
        if any(x < 0 for x in labels):
            raise ValueError("Dynkin coefficients must be nonnegative, got {}".format(labels))
        self.algebra = algebra
        self.labels = labels

    @property
    def dim(self):
        return _dimension(self.algebra, self.labels)

    def conjugate(self):
        return LieIrrep(self.algebra, self.algebra.dominant(tuple(-x for x in self.labels))[0])

    # Same convention as the SU(N) code: barred when the labels are lexicographically smaller than the conjugate's
    @property
    def bar(self):
        return self.labels < self.conjugate().labels

    def weights(self):
        return _weights(self.algebra, self.labels)

    def __eq__(self, other):
        return isinstance(other, LieIrrep) and (self.algebra, self.labels) == (other.algebra, other.labels)

    def __hash__(self):
        return hash((self.algebra, self.labels))

    def __repr__(self):
        display_dim = str(self.dim)
        if self.bar:
            display_dim += ' bar'
        return "({idx})[{dim}]".format(idx=', '.join(str(x) for x in self.labels), dim=display_dim)


@lru_cache(maxsize=4096)
def _dimension(algebra, labels):
    return algebra.dimension(labels)


@lru_cache(maxsize=256)
def _weights(algebra, labels):
    return algebra.weights(labels)


@lru_cache(maxsize=256)
def _dominant_multiplicities(algebra, labels):
    return algebra.dominant_multiplicities(labels)


# Racah-Speiser: irrep1 x irrep2 = sum over weights nu of irrep2 of sign(w) [w(lambda + nu + rho) - rho].
# Returns a list of (irrep, multiplicity), largest dimension first.
# The work is one step per distinct weight of the smaller factor; past 'max_weights' steps a ValueError is raised.
def decompose(irrep1, irrep2, max_weights=None):
    if irrep1.algebra != irrep2.algebra:
        raise ValueError("Cannot multiply a {} irrep by a {} irrep".format(irrep1.algebra.name, irrep2.algebra.name))
    algebra = irrep1.algebra
    # The work is one step per weight of the factor whose weights are used, so use the smaller one
    big, small = sorted([irrep1, irrep2], key=lambda irrep: (irrep.dim, irrep.labels), reverse=True)
    shifted = tuple(x + 1 for x in big.labels)
    terms = Counter()
    count = 0
    for mu, m in _dominant_multiplicities(algebra, small.labels).items():
        for nu in algebra.orbit(mu):
            count += 1
            if max_weights is not None and count > max_weights:
                raise ValueError("{} has more than {} distinct weights; too large to multiply here"
                                 .format(small, max_weights))
            reflected = algebra.strictly_dominant([a + b for a, b in zip(shifted, nu)])
            if reflected is None:
                continue
            w, steps = reflected
            terms[tuple(x - 1 for x in w)] += -m if steps % 2 else m
    result = [(LieIrrep(algebra, labels), mult) for labels, mult in terms.items() if mult]
    assert all(mult > 0 for _, mult in result), "Racah-Speiser left a negative multiplicity"
    return sorted(result, key=lambda item: (-item[0].dim, tuple(-x for x in item[0].labels)))


def decompose_many(irreps, max_weights=None):
    irreps = list(irreps)
    if not irreps:
        raise ValueError("Need at least one irrep")
    terms = Counter({irreps[0]: 1})
    for factor in irreps[1:]:
        new_terms = Counter()
        for irrep, mult in terms.items():
            for result, m in decompose(irrep, factor, max_weights):
                new_terms[result] += mult * m
        terms = new_terms
    return sorted(terms.items(), key=lambda item: (-item[0].dim, tuple(-x for x in item[0].labels)))


def format_product(irreps, terms):
    lhs = ' ⊗ '.join(str(irrep) for irrep in irreps)
    rhs = ' ⊕ '.join((str(mult) + ' ' if mult > 1 else '') + str(irrep) for irrep, mult in terms)
    return lhs + ' = ' + rhs
