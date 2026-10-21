# Builds the reference data in tests/data from programs and published tables that share no code with this repository.
#   python tools/make_reference_data.py lie          needs the LiE program (Debian/Ubuntu package 'lie')
#   python tools/make_reference_data.py gap          needs GAP 4 (Debian/Ubuntu package 'gap')
#   python tools/make_reference_data.py lieart DIR   DIR = Documentation/English/Tables of a LieART checkout
#                                                    (https://github.com/rfeger/LieART)
# The tests only read the JSON files, so they need none of these programs.
# Only LiE and GAP are run here; nothing from lie_algebra_CG.py is used except to pick which irreps to ask about.

import itertools
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lie_algebra_CG import SimpleLieAlgebra  # noqa: E402  (only for dimensions, to keep the files small)

DATA = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data')

ALGEBRAS = ([('A', n) for n in range(1, 9)] + [('B', n) for n in range(2, 9)] + [('C', n) for n in range(2, 9)]
            + [('D', n) for n in range(3, 9)] + [('G', 2), ('F', 4), ('E', 6), ('E', 7), ('E', 8)])


def write_json(name, data):
    # one record per line, so that diffs stay readable
    with open(os.path.join(DATA, name), 'w') as f:
        f.write('{\n')
        items = list(data.items())
        for k, (key, value) in enumerate(items):
            f.write('"{}": '.format(key))
            if isinstance(value, dict):
                f.write('{\n')
                sub = list(value.items())
                for j, (alg, rows) in enumerate(sub):
                    f.write('  "{}": [\n'.format(alg))
                    f.write(',\n'.join('    ' + json.dumps(row, separators=(',', ':')) for row in rows))
                    f.write('\n  ]' + (',' if j < len(sub) - 1 else '') + '\n')
                f.write('}')
            else:
                f.write(json.dumps(value))
            f.write(',\n' if k < len(items) - 1 else '\n')
        f.write('}\n')


def labels_up_to(rank, total):
    return [l for l in itertools.product(range(total + 1), repeat=rank) if sum(l) <= total]


# ---- LiE ----

LIE_TERM = re.compile(r'(-?\d+)X\[([-\d,]*)\]')


def run_lie(commands):
    script = ''.join('{}\nprint("@@@")\n'.format(c) for c in commands)
    out = subprocess.run(['lie'], input=script, capture_output=True, text=True, check=True).stdout
    parts = out.split('@@@')[:-1]
    assert len(parts) == len(commands), out[-2000:]
    return [p.strip() for p in parts]


def lie_vec(labels):
    return '[' + ','.join(str(x) for x in labels) + ']'


def lie_poly(text):
    text = re.sub(r'\s+', '', text)
    return sorted([[int(x) for x in labels.split(',')], int(c)] for c, labels in LIE_TERM.findall(text))


# Products of these large irreps are checked too
LIE_EXTRA_PRODUCTS = [
    ('E8', (1, 0, 0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0, 0)), ('E8', (0, 0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 0, 1)),
    ('E8', (0, 0, 0, 0, 0, 0, 0, 2), (0, 0, 0, 0, 0, 0, 0, 1)), ('E7', (1, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1, 0)),
    ('E7', (0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1, 0)), ('E7', (0, 0, 0, 0, 0, 0, 2), (0, 0, 0, 0, 0, 0, 3)),
    ('E6', (0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 1, 0)), ('E6', (1, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 1)),
    ('E6', (2, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 2)), ('F4', (0, 1, 0, 0), (0, 0, 0, 1)), ('F4', (0, 0, 1, 0), (0, 0, 1, 0)),
    ('F4', (1, 0, 0, 1), (0, 0, 0, 2)), ('G2', (5, 3), (4, 2)), ('G2', (0, 7), (6, 0)),
    ('B5', (0, 0, 0, 1, 1), (1, 0, 0, 0, 1)), ('D6', (0, 0, 0, 0, 2, 0), (0, 0, 0, 0, 1, 1)),
    ('C5', (0, 1, 0, 1, 0), (0, 0, 1, 0, 0)), ('A5', (1, 2, 0, 1, 0), (0, 1, 0, 2, 1)),
]


def make_lie():
    dimensions, characters, products = {}, {}, {}
    for series, rank in ALGEBRAS:
        g = SimpleLieAlgebra(series, rank)
        name = series + str(rank)
        level = {1: 10, 2: 6, 3: 4, 4: 4, 5: 3, 6: 3}.get(rank, 2)
        labels = labels_up_to(rank, level)
        out = run_lie(['dim({},{})'.format(lie_vec(l), name) for l in labels] +
                      ['contragr({},{})'.format(lie_vec(l), name) for l in labels])
        dimensions[name] = [[list(l), int(d), [int(x) for x in c.strip('[]').split(',')]]
                            for l, d, c in zip(labels, out, out[len(labels):])]

        small = [l for l in labels_up_to(rank, 2 if rank > 2 else 4) if g.dimension(l) <= 3000]
        out = run_lie(['dom_char({},{})'.format(lie_vec(l), name) for l in small])
        characters[name] = [[list(l), lie_poly(o)] for l, o in zip(small, out)]

        level = {1: 6, 2: 3, 3: 2, 4: 2}.get(rank, 1)
        factors = [l for l in labels_up_to(rank, level) if sum(l) > 0]
        pairs = [(a, b) for a, b in itertools.combinations_with_replacement(factors, 2)
                 if min(g.dimension(a), g.dimension(b)) <= 5000 and g.dimension(a) * g.dimension(b) <= 2 * 10 ** 6]
        pairs += [(a, b) for alg, a, b in LIE_EXTRA_PRODUCTS if alg == name]
        out = run_lie(['tensor({},{},{})'.format(lie_vec(a), lie_vec(b), name) for a, b in pairs])
        products[name] = [[list(a), list(b), lie_poly(o)] for (a, b), o in zip(pairs, out)]
    write_json('lie_reference.json', {
        'source': 'LiE 2.2.2 (van Leeuwen, Cohen, Lisser), Bourbaki numbering like this repository; generated by '
                  'tools/make_reference_data.py lie',
        'dimensions': dimensions, 'dominant_characters': characters, 'tensor_products': products})


# ---- GAP ----

GAP_CASES = [('G', 2, 3), ('F', 4, 1), ('E', 6, 1), ('E', 7, 1), ('B', 3, 2), ('C', 3, 2), ('D', 4, 2), ('A', 3, 2)]


def run_gap(script):
    return subprocess.run(['gap', '-q', '-o', '4g'], input=script + '\nQUIT;\n', capture_output=True, text=True,
                          check=True).stdout


def gap_value(text):
    return json.loads(re.sub(r'\s+', '', text.replace('\\\n', '')))


def make_gap():
    cartan, dimensions, products = {}, {}, {}
    for series, rank, level in GAP_CASES:
        name = series + str(rank)
        start = 'L:=SimpleLieAlgebra("{}",{},Rationals);;'.format(series, rank)
        cartan[name] = gap_value(run_gap(start + 'Print(CartanMatrix(RootSystem(L)));'))
        # labels in GAP's own numbering of the simple roots
        labels = [l for l in labels_up_to(rank, level) if sum(l) > 0]
        dims = [int(x) for x in run_gap(start + ''.join(
            'Print(DimensionOfHighestWeightModule(L,{}),"\\n");'.format(list(l)) for l in labels)).split()]
        dim = {l: d for l, d in zip(labels, dims) if d <= 3000}
        labels = list(dim)
        dimensions[name] = [[list(l), d] for l, d in dim.items()]
        pairs = [(a, b) for a, b in itertools.combinations_with_replacement(labels, 2) if dim[a] * dim[b] <= 400000]
        out = run_gap(start + ''.join('Print(DecomposeTensorProduct(L,{},{}),"\\n");'.format(list(a), list(b))
                                      for a, b in pairs))
        rows = re.split(r'\n(?=\[ \[ \[)', out.strip())
        assert len(rows) == len(pairs)
        products[name] = []
        for (a, b), row in zip(pairs, rows):
            weights, mults = gap_value(row)
            products[name].append([list(a), list(b), sorted([w, m] for w, m in zip(weights, mults))])
    version = run_gap('Print(GAPInfo.Version);').strip()
    write_json('gap_reference.json', {
        'source': 'GAP {} (DimensionOfHighestWeightModule, DecomposeTensorProduct), labels in GAP\'s numbering of '
                  'the simple roots; generated by tools/make_reference_data.py gap'.format(version),
        'cartan_matrices': {k: [v] for k, v in cartan.items()}, 'dimensions': dimensions, 'tensor_products': products})


# ---- LieART tables ----
# The LieART 2.0 tables of irreps and tensor products (Feger, Kephart, Saskowski, Comput. Phys. Commun. 257 (2020)
# 107490, arXiv:1912.10969, appendix) are Mathematica notebooks in Documentation/English/Tables. Irreps are written
# by name there: the dimension, a bar, primes and, for SO(8), a subscript, e.g. 351bar', 8_s.

PRIME = r'(?:\\\[Prime\])'


def flat_notebook(path):
    text = open(path).read().replace('\\\n', '')
    return re.sub(r'\s+', '', text)


def table_algebra(chunk):
    chunk = chunk[:400].replace('\\', '')
    m = re.search(r'SubscriptBox\["*([EFG])"*,"*(\d)"*\]', chunk)
    if m:
        return m.group(1) + m.group(2)
    m = re.search(r'(SU|SO|Sp)\((\d+)\)', chunk)
    return '{}({})'.format(m.group(1), m.group(2))


def simplify_names(text):
    text = re.sub(r'StyleBox\[OverscriptBox\["(\d+)","_"\],FontWeight->Bold\]', r'«\1bar»', text)
    text = re.sub(r'StyleBox\["(\d+)",FontWeight->Bold\]', r'«\1»', text)
    for _ in range(3):
        text = re.sub(r'SuperscriptBox\[«([^»]*)»,"(' + PRIME + r'+)"\]',
                      lambda m: '«' + m.group(1) + "'" * m.group(2).count('Prime') + '»', text)
        text = re.sub(r'SubscriptBox\[«([^»]*)»,"(\w+)"\]', r'«\1_\2»', text)
        text = re.sub(r'SubsuperscriptBox\[«([^»]*)»,"(\w+)","(' + PRIME + r'+)"\]',
                      lambda m: '«' + m.group(1) + "'" * m.group(3).count('Prime') + '_' + m.group(2) + '»', text)
        text = re.sub(r'FormBox\[«([^»]*)»,TraditionalForm\]', r'«\1»', text)
    return re.sub(r'RowBox\[\{"(\d+)","\(",«([^»]*)»,"\)"\}\]', r'«\1*\2»', text)


def make_lieart(directory):
    irreps = {}
    text = flat_notebook(os.path.join(directory, 'IrrepProperties.nb'))
    for chunk in text.split('Table\\[MediumSpace]')[1:]:
        rows = re.findall(r'\{FormBox\["\\<\\"\(([\d,]+)\)\\"\\>",TraditionalForm\],(.*?)DisplayForm\],'
                          r'TraditionalForm\]', chunk)
        algebra = table_algebra(chunk)
        table = []
        for labels, box in rows:
            dim = re.search(r'OverscriptBox\["(\d+)"|RowBox\[\{"(\d+)"', box)
            name = (dim.group(1) or dim.group(2)) + ('bar' if 'OverscriptBox' in box else '')
            name += "'" * box.count('Prime')
            sub = re.search(r'SubsuperscriptBox\["\\\[AlignmentMarker\]","(\w+)"', box)
            if sub:
                name += '_' + sub.group(1)
            # one digit per label, except for SU(2) and where a label has two digits (then they are comma-separated)
            digits = labels.split(',') if ',' in labels or algebra == 'SU(2)' else labels
            table.append([[int(x) for x in digits], name])
        irreps.setdefault(algebra, []).extend(table)

    products = {}
    text = flat_notebook(os.path.join(directory, 'TensorProducts.nb'))
    for chunk in text.split('Table\\[MediumSpace]')[1:]:
        s = simplify_names(chunk)
        rows = re.findall(r'RowBox\[\{«([^»]*)»,"\\\[CircleTimes\]",«([^»]*)»\}\],"\\<\\"=\\"\\>",'
                          r'(RowBox\[\{[^\[\]]*\}\]|«[^»]*»)', s)
        assert len(rows) == s.count('"\\<\\"=\\"\\>"'), 'a row was not understood'
        table = []
        for a, b, rhs in rows:
            terms = [[int(t.split('*')[0]), t.split('*')[1]] if '*' in t else [1, t]
                     for t in re.findall(r'«([^»]*)»', rhs)]
            table.append([a, b, terms])
        products.setdefault(table_algebra(chunk), []).extend(table)
    write_json('lieart_tables.json', {
        'source': 'LieART 2.1.1 (github.com/rfeger/LieART, commit 3dd8a2c), Documentation/English/Tables: the tables '
                  'of arXiv:1912.10969. Labels in LieART\'s numbering of the simple roots; generated by '
                  'tools/make_reference_data.py lieart',
        'irreps': irreps, 'tensor_products': products})


if __name__ == '__main__':
    if sys.argv[1:2] == ['lie']:
        make_lie()
    elif sys.argv[1:2] == ['gap']:
        make_gap()
    elif sys.argv[1:2] == ['lieart'] and len(sys.argv) == 3:
        make_lieart(sys.argv[2])
    else:
        sys.exit('usage: python tools/make_reference_data.py lie | gap | lieart DIR')
