# Web app for Clebsch-Gordan decompositions of tensor products of irreps of simple Lie algebras, and for branching
# rules (branching.py). Run with:
#   python app.py
# (also works as app.py of a Hugging Face Space, together with requirements.txt)
# SU(N) products use Young tableaux (sun_irrep_CG.py); SO(N), Sp(N), G2, F4, E6, E7, E8 use weights (lie_algebra_CG.py).

import html
import re

import gradio as gr

import branching
import lie_algebra_CG
import sun_irrep_CG
from lie_algebra_CG import LieIrrep, lie_algebra
from sun_irrep_CG import IrrepSUN

MAX_N = 12
MAX_BOXES = 40       # largest Young diagram accepted for a single SU(N) factor
MAX_SMALL_BOXES = 24  # every factor after the largest is cut into strips, so keep those smaller
MAX_RANK = 12        # for the other algebras
MAX_WEIGHTS = 400000  # distinct weights of a factor, for the other algebras
MAX_FACTORS = 4
MAX_ROWS_SHOWN = 400
MAX_BRANCH_WEIGHTS = 300000  # distinct weights of the irrep being branched

DYNKIN = "Dynkin labels"
PARTITION = "Young diagram rows"


# Parse '1, 1' or '(1 1)' or '[2,1]' into a list of nonnegative integers
def parse_numbers(text):
    text = text.strip()
    if not text or text in ('0', '()', '[]', '-'):
        return []
    parts = [p for p in re.split(r'[\s,;()\[\]]+', text) if p]
    try:
        numbers = [int(p) for p in parts]
    except ValueError:
        raise ValueError("'{}' is not a list of integers".format(text))
    if any(x < 0 for x in numbers):
        raise ValueError("'{}' has a negative entry".format(text))
    return numbers


def parse_irrep(algebra, text, mode):
    numbers = parse_numbers(text)
    if algebra.series == 'A':
        n = algebra.rank + 1
        if mode == PARTITION:
            return IrrepSUN.from_partition(n, numbers)
        if not numbers:
            return IrrepSUN(n, [0] * (n - 1))
        if len(numbers) != n - 1:
            raise ValueError("SU({}) irreps need {} Dynkin labels, '{}' has {}".format(n, n - 1, text, len(numbers)))
        return IrrepSUN(n, numbers)
    if mode == PARTITION:
        raise ValueError("Young diagram input only works for SU(N); enter {} irreps as Dynkin labels"
                         .format(algebra.name))
    if not numbers:
        return LieIrrep(algebra, [0] * algebra.rank)
    if len(numbers) != algebra.rank:
        raise ValueError("{} irreps need {} Dynkin labels, '{}' has {}".format(algebra.name, algebra.rank, text,
                                                                              len(numbers)))
    return LieIrrep(algebra, numbers)


def parse_algebra(group):
    algebra = lie_algebra(str(group).strip())
    if algebra.series == 'A' and algebra.rank + 1 > MAX_N:
        raise ValueError("SU(N) is limited to N <= {} here".format(MAX_N))
    if algebra.rank > MAX_RANK:
        raise ValueError("Rank is limited to {} here".format(MAX_RANK))
    return algebra


def parse_factors(group, mode, texts):
    algebra = parse_algebra(group)
    factors = [parse_irrep(algebra, text, mode) for text in texts if text is not None and text.strip()]
    if len(factors) < 2:
        raise ValueError("Enter at least two irreps")
    if len(factors) > MAX_FACTORS:
        raise ValueError("At most {} factors".format(MAX_FACTORS))
    if algebra.series == 'A':
        boxes = sorted((irrep.n_boxes for irrep in factors), reverse=True)
        if boxes[0] > MAX_BOXES or any(b > MAX_SMALL_BOXES for b in boxes[1:]):
            raise ValueError("These Young diagrams are too large for the web app (at most {} boxes for the largest "
                             "factor and {} for the others). Use sun_irrep_CG.py directly for bigger cases."
                             .format(MAX_BOXES, MAX_SMALL_BOXES))
    return factors


def decompose_factors(factors):
    if isinstance(factors[0], IrrepSUN):
        return sun_irrep_CG.decompose_many(factors)
    return lie_algebra_CG.decompose_many(factors, max_weights=MAX_WEIGHTS)


# Young diagram as an inline SVG; 'currentColor' keeps it readable in light and dark themes
def young_svg(irrep, size=12):
    if not isinstance(irrep, IrrepSUN):
        return ''
    rows = irrep.partition
    if not rows:
        return '<span title="trivial irrep">•</span>'
    width = rows[0] * size + 2
    height = len(rows) * size + 2
    rects = ''.join('<rect x="{}" y="{}" width="{}" height="{}"/>'.format(1 + c * size, 1 + r * size, size, size)
                    for r, length in enumerate(rows) for c in range(length))
    return ('<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="Young diagram {p}">'
            '<g fill="none" stroke="currentColor" stroke-width="1">{rects}</g></svg>'
            .format(w=width, h=height, p=irrep.partition, rects=rects))


def irrep_name(irrep):
    labels = '(' + ', '.join(str(x) for x in irrep.labels) + ')'
    dim = '<b>{}</b>'.format(irrep.dim)
    if irrep.bar:
        dim = '<span style="text-decoration: overline"><b>{}</b></span>'.format(irrep.dim)
    return dim, labels


def render(factors, terms):
    total_dim = 1
    for irrep in factors:
        total_dim *= irrep.dim
    count = sum(mult for _, mult in terms)
    lhs = ' ⊗ '.join('<span title="{}">{}</span>'.format(html.escape(str(f)), irrep_name(f)[0]) for f in factors)
    rhs = ' ⊕ '.join((str(mult) + '·' if mult > 1 else '') + irrep_name(irrep)[0] for irrep, mult in terms)
    if len(terms) > 60:
        rhs = ' ⊕ '.join((str(mult) + '·' if mult > 1 else '') + irrep_name(irrep)[0]
                              for irrep, mult in terms[:60]) + ' ⊕ …'
    inputs = ''.join('<td style="padding:4px 12px;text-align:center">{}<br>{}<br><code>{}</code></td>'
                     .format(young_svg(f), irrep_name(f)[0], irrep_name(f)[1]) for f in factors)
    # Young diagrams only for SU(N)
    diagrams = isinstance(factors[0], IrrepSUN)
    rows = ''.join(
        '<tr><td style="padding:2px 8px">{dim}</td><td style="padding:2px 8px"><code>{labels}</code></td>'
        '<td style="padding:2px 8px;text-align:right">{mult}</td>{svg}</tr>'
        .format(dim=irrep_name(irrep)[0], labels=irrep_name(irrep)[1], mult=mult,
                svg='<td style="padding:2px 8px">{}</td>'.format(young_svg(irrep)) if diagrams else '')
        for irrep, mult in terms[:MAX_ROWS_SHOWN])
    more = ''
    if len(terms) > MAX_ROWS_SHOWN:
        more = '<p>Showing the {} largest of {} distinct irreps.</p>'.format(MAX_ROWS_SHOWN, len(terms))
    check = sum(irrep.dim * mult for irrep, mult in terms)
    header_diagram = '<th style="text-align:left;padding:2px 8px">Young diagram</th>' if diagrams else ''
    return (
        '<div style="font-size:1.1em;line-height:1.8">{lhs} = {rhs}</div>'
        '<p>{count} irreps in total ({distinct} distinct). Dimension check: {total} = {check} {ok}</p>'
        '<table><tr>{inputs}</tr></table>'
        '<table style="border-collapse:collapse;margin-top:8px">'
        '<tr><th style="text-align:left;padding:2px 8px">Dim</th><th style="text-align:left;padding:2px 8px">'
        'Dynkin labels</th><th style="padding:2px 8px">Multiplicity</th>'
        '{header_diagram}</tr>{rows}</table>{more}'
    ).format(lhs=lhs, rhs=rhs, count=count, distinct=len(terms), total=total_dim, check=check,
             ok='✔' if check == total_dim else '✘', inputs=inputs, rows=rows, more=more,
             header_diagram=header_diagram)


# Returns (html result, plain text equation) for the web page
def compute(group, mode, *texts):
    try:
        factors = parse_factors(group, mode, texts)
        terms = decompose_factors(factors)
    except ValueError as error:
        return '<p style="color:#c0392b">{}</p>'.format(html.escape(str(error))), ''
    return render(factors, terms), lie_algebra_CG.format_product(factors, terms)


# ---- branching rules ----

def parse_group_irrep(algebra, text):
    numbers = parse_numbers(text)
    if not numbers:
        numbers = [0] * algebra.rank
    if len(numbers) != algebra.rank:
        raise ValueError("{} irreps need {} Dynkin labels, '{}' has {}".format(algebra.name, algebra.rank, text,
                                                                              len(numbers)))
    return LieIrrep(algebra, numbers)


def dimension_html(irrep):
    level = branching.prime_level(irrep.algebra, irrep.labels)
    text = '<b>{}</b>{}'.format(irrep.dim, '′' * level)
    if irrep.bar:
        text = '<span style="text-decoration: overline">{}</span>'.format(text)
    return text


# An irrep of the subalgebra in Slansky's notation, e.g. (3, 2)(1)
def sub_html(sub):
    names = [dimension_html(irrep) for irrep in sub.irreps]
    text = names[0] if len(names) == 1 else '(' + ', '.join(names) + ')' if names else ''
    return text + ''.join('({})'.format(q) for q in sub.charges)


def render_branching(irrep, e, terms):
    cell = '<td style="padding:2px 8px">{}</td>'
    header = '<th style="text-align:left;padding:2px 8px">{}</th>'
    rhs = ' ⊕ '.join((str(m) + '·' if m > 1 else '') + sub_html(sub) for sub, m in terms[:60])
    if len(terms) > 60:
        rhs += ' ⊕ …'
    u1s = [i for i, f in enumerate(e.factors) if branching.is_u1(f)]
    heads = ''.join(header.format(html.escape(f.name if i not in u1s else
                                              'U(1)' if len(u1s) == 1 else 'U(1) #{}'.format(u1s.index(i) + 1)))
                    for i, f in enumerate(e.factors))
    rows = ''.join('<tr>' + cell.format(m) + cell.format(sub_html(sub)) +
                   ''.join(cell.format(p if isinstance(p, branching.Fraction) else
                                       '<code>({})</code>'.format(', '.join(str(x) for x in p.labels)))
                           for p in sub.parts) + '</tr>'
                   for sub, m in terms[:MAX_ROWS_SHOWN])
    more = ''
    if len(terms) > MAX_ROWS_SHOWN:
        more = '<p>Showing the {} largest of {} distinct irreps.</p>'.format(MAX_ROWS_SHOWN, len(terms))
    check = sum(sub.dim * m for sub, m in terms)
    conventions = ''.join('<li>{}</li>'.format(html.escape(line)) for line in branching.describe(e))
    return (
        '<div style="font-size:1.1em;line-height:1.8">{lhs} = {rhs}</div>'
        '<p>{count} irreps in total ({distinct} distinct). Dimension check: {dim} = {check} {ok}</p>'
        '<p>{g} ⊃ {h}: {description}. Conventions:</p><ul>{conventions}</ul>'
        '<table style="border-collapse:collapse;margin-top:8px"><tr>{mult}{irrep}{heads}</tr>{rows}</table>{more}'
    ).format(lhs=dimension_html(irrep), rhs=rhs, count=sum(m for _, m in terms), distinct=len(terms), dim=irrep.dim,
             check=check, ok='✔' if check == irrep.dim else '✘', g=html.escape(e.parent[0].name),
             h=html.escape(e.name), description=html.escape(e.description), conventions=conventions,
             mult=header.format('Multiplicity'), irrep=header.format('Irrep'), heads=heads, rows=rows, more=more)


# Returns (html result, plain text) for the branching rules tab
def compute_branching(group, subgroup, irrep_text):
    try:
        algebra = parse_algebra(group)
        e = branching.embedding(algebra, str(subgroup or ''))
        irrep = parse_group_irrep(algebra, irrep_text or '')
        terms = branching.branch(irrep, e, max_weights=MAX_BRANCH_WEIGHTS)
    except ValueError as error:
        return '<p style="color:#c0392b">{}</p>'.format(html.escape(str(error))), ''
    return render_branching(irrep, e, terms), branching.format_branching(irrep, terms)


def subgroup_choices(group):
    try:
        return branching.known_subgroups(parse_algebra(str(group).strip()))
    except ValueError:
        return []


# When the group changes: new suggestions, and the first of them replaces the subgroup if that does not fit any more
def update_subgroups(group, subgroup):
    choices = subgroup_choices(group)
    if choices:
        try:
            branching.embedding(parse_algebra(group), subgroup)
        except ValueError:
            subgroup = choices[0]
    return gr.Dropdown(choices=choices, value=None), subgroup


BRANCH_EXAMPLES = [
    ["SU(5)", "SU(3)xSU(2)xU(1)", "0, 1, 0, 0"],                       # the 10 of SU(5)
    ["SO(10)", "SU(5)xU(1)", "0, 0, 0, 1, 0"],                         # the 16 of SO(10)
    ["SO(10)", "SU(4)xSU(2)xSU(2)", "0, 0, 0, 2, 0"],                  # the 126 of SO(10)
    ["SO(10)", "SU(5)xU(1) > SU(3)xSU(2)xU(1)xU(1)", "0, 0, 0, 1, 0"],
    ["E6", "SO(10)xU(1)", "1, 0, 0, 0, 0, 0"],                         # the 27 of E6
    ["E6", "SU(3)xSU(3)xSU(3)", "0, 1, 0, 0, 0, 0"],                   # the 78 of E6
    ["E7", "SU(2)xF4", "0, 0, 0, 0, 0, 0, 1"],
    ["E8", "SU(2)", "0, 0, 0, 0, 0, 0, 0, 1"],                          # principal SU(2): spins = exponents
    ["G2", "SU(3)", "1, 0"],
]


EXAMPLES = [
    ["SU(3)", DYNKIN, "1, 0", "0, 1", ""],        # 3 x 3bar = 8 + 1
    ["SU(3)", DYNKIN, "1, 1", "1, 1", ""],        # 8 x 8
    ["SU(3)", DYNKIN, "1, 0", "1, 0", "1, 0"],    # 3 x 3 x 3
    ["SU(2)", DYNKIN, "1", "2", ""],              # spin 1/2 x spin 1
    ["SU(5)", DYNKIN, "0, 1, 0, 0", "0, 1, 0, 0", ""],  # 10 x 10 in SU(5)
    ["SU(6)", PARTITION, "2, 1", "2, 1", ""],
    ["SO(10)", DYNKIN, "0, 0, 0, 1, 0", "0, 0, 0, 1, 0", ""],  # 16 x 16
    ["Sp(4)", DYNKIN, "1, 0", "0, 1", ""],
    ["G2", DYNKIN, "1, 0", "1, 0", ""],           # 7 x 7
    ["F4", DYNKIN, "0, 0, 0, 1", "0, 0, 0, 1", ""],  # 26 x 26
    ["E6", DYNKIN, "1, 0, 0, 0, 0, 0", "0, 0, 0, 0, 0, 1", ""],  # 27 x 27bar
    ["E8", DYNKIN, "0, 0, 0, 0, 0, 0, 0, 1", "0, 0, 0, 0, 0, 0, 0, 1", ""],  # 248 x 248
]


def build_demo():
    with gr.Blocks(title="Clebsch-Gordan decomposition") as demo:
        with gr.Tab("Tensor products"):
            gr.Markdown(
                "# Clebsch-Gordan decomposition\n"
                "Decompose a tensor product of irreducible representations of SU(N), SO(N), Sp(N), G2, F4, E6, E7 or "
                "E8 into irreducible ones. SU(N) uses Young tableaux (the Littlewood-Richardson rule; see Georgi, *Lie "
                "Algebras in Particle Physics*); the other groups use weights (Freudenthal's formula and the "
                "Racah-Speiser algorithm).\n\n"
                "Enter each irrep as its Dynkin labels in the Bourbaki numbering, e.g. `1, 1` for the SU(3) octet or "
                "`0, 0, 0, 1, 0` for the 16 of SO(10). SU(N) irreps can also be entered as the row lengths of their "
                "Young diagram, e.g. `2, 1`. An irrep is marked with a bar when its Dynkin labels are "
                "lexicographically smaller than those of its conjugate.")
            with gr.Row():
                n = gr.Textbox(value="SU(3)", label="Group",
                               info="SU(N), SO(N), Sp(N), G2, F4, E6, E7, E8 (or A3, B4, C2, D5, ...)")
                mode = gr.Radio([DYNKIN, PARTITION], value=DYNKIN, label="Enter irreps as")
            with gr.Row():
                first = gr.Textbox(value="1, 1", label="Irrep 1")
                second = gr.Textbox(value="1, 1", label="Irrep 2")
                third = gr.Textbox(value="", label="Irrep 3 (optional)")
                fourth = gr.Textbox(value="", label="Irrep 4 (optional)")
            button = gr.Button("Decompose", variant="primary")
            result = gr.HTML()
            text = gr.Textbox(label="As text", lines=2)
            inputs = [n, mode, first, second, third, fourth]
            gr.Examples([example + [""] for example in EXAMPLES], inputs=inputs)
            button.click(compute, inputs=inputs, outputs=[result, text])
            for box in [first, second, third, fourth]:
                box.submit(compute, inputs=inputs, outputs=[result, text])
        with gr.Tab("Branching rules"):
            gr.Markdown(
                "# Branching rules\n"
                "Decompose an irrep of a group G into irreps of a subgroup H, e.g. SU(5) ⊃ SU(3)×SU(2)×U(1) or "
                "E6 ⊃ SO(10)×U(1). Pick a subgroup from the list or type one: a product of factors such as "
                "`SU(3)xSU(2)xU(1)`, or a chain such as `SU(5)xU(1) > SU(3)xSU(2)xU(1)xU(1)`, where each step breaks "
                "one factor and the new factors take its place. Regular subgroups (from the extended Dynkin diagram, "
                "with a U(1) for each deleted node) are found automatically; `SU(2)` alone is the principal SU(2) "
                "unless a regular or classical one is known.\n\n"
                "Irreps of H are written as in Slansky's tables: dimensions of the simple factors (primes tell apart "
                "irreps of the same dimension, by their index), then the U(1) charges, normalized to the smallest "
                "integers. Signs and normalizations of U(1) charges are conventions; the ones used are shown with "
                "each result. Hypercharge in SU(5) ⊃ SU(3)×SU(2)×U(1) is the U(1) charge divided by 6.")
            with gr.Row():
                b_group = gr.Textbox(value="SU(5)", label="Group G", info="SU(N), SO(N), Sp(N), G2, F4, E6, E7, E8")
                b_subgroup = gr.Textbox(value="SU(3)xSU(2)xU(1)", label="Subgroup H",
                                        info="e.g. SU(3)xSU(2)xU(1), or a chain A > B")
                b_known = gr.Dropdown(choices=subgroup_choices("SU(5)"), value=None, label="Known subgroups of G",
                                      info="pick one to fill in H")
                b_irrep = gr.Textbox(value="0, 1, 0, 0", label="Irrep of G (Dynkin labels)")
            b_button = gr.Button("Branch", variant="primary")
            b_result = gr.HTML()
            b_text = gr.Textbox(label="As text", lines=2)
            b_inputs = [b_group, b_subgroup, b_irrep]
            gr.Examples(BRANCH_EXAMPLES, inputs=b_inputs)
            b_group.change(update_subgroups, inputs=[b_group, b_subgroup], outputs=[b_known, b_subgroup])
            b_known.input(lambda choice, current: choice or current, inputs=[b_known, b_subgroup],
                          outputs=b_subgroup)
            b_button.click(compute_branching, inputs=b_inputs, outputs=[b_result, b_text])
            for box in [b_subgroup, b_irrep]:
                box.submit(compute_branching, inputs=b_inputs, outputs=[b_result, b_text])
        demo.load(compute, inputs=inputs, outputs=[result, text])
        demo.load(compute_branching, inputs=b_inputs, outputs=[b_result, b_text])
    return demo


demo = build_demo()

if __name__ == '__main__':
    demo.launch()
