# Independent check of the decompositions via characters: the weights of an SU(N) irrep are the contents of the
# semistandard Young tableaux of its shape with entries 1..N. A decomposition is right exactly when the weights of the
# product equal the weights of the sum.

from collections import Counter


# Weights (as exponent tuples) of every semistandard Young tableau of shape 'partition' with entries 1..n
def ssyt_weights(partition, n):
    cells = [(r, c) for r, length in enumerate(partition) for c in range(length)]
    filling = {}
    content = [0] * n
    weights = Counter()

    def fill(i):
        if i == len(cells):
            weights[tuple(content)] += 1
            return
        r, c = cells[i]
        low = 1
        if c > 0:
            low = max(low, filling[(r, c - 1)])
        if r > 0:
            low = max(low, filling[(r - 1, c)] + 1)
        for value in range(low, n + 1):
            filling[(r, c)] = value
            content[value - 1] += 1
            fill(i + 1)
            content[value - 1] -= 1

    fill(0)
    return weights


# Since x_1 x_2 ... x_N = 1 in SU(N), exponent tuples are only defined up to adding (1, ..., 1)
def normalize(weight):
    low = min(weight)
    return tuple(x - low for x in weight)


def character(irrep):
    result = Counter()
    for weight, mult in ssyt_weights(irrep.partition, irrep.n).items():
        result[normalize(weight)] += mult
    return result


def character_product(char1, char2):
    result = Counter()
    for w1, m1 in char1.items():
        for w2, m2 in char2.items():
            result[normalize(tuple(a + b for a, b in zip(w1, w2)))] += m1 * m2
    return result


def character_sum(terms):
    result = Counter()
    for irrep, mult in terms:
        for weight, m in character(irrep).items():
            result[weight] += mult * m
    return result
