# Web app for Clebsch-Gordan decompositions of tensor products of irreps of simple Lie algebras. Run with:
#   python app.py
# (also works as app.py of a Hugging Face Space, together with requirements.txt)
# SU(N) products use Young tableaux (sun_irrep_CG.py); SO(N), Sp(N), G2, F4, E6, E7, E8 use weights (lie_algebra_CG.py).

import html
import re

import gradio as gr

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
        gr.Markdown(
            "# Clebsch-Gordan decomposition\n"
            "Decompose a tensor product of irreducible representations of SU(N), SO(N), Sp(N), G2, F4, E6, E7 or E8 "
            "into irreducible ones. SU(N) uses Young tableaux (the Littlewood-Richardson rule; see Georgi, *Lie "
            "Algebras in Particle Physics*); the other groups use weights (Freudenthal's formula and the "
            "Racah-Speiser algorithm).\n\n"
            "Enter each irrep as its Dynkin labels in the Bourbaki numbering, e.g. `1, 1` for the SU(3) octet or "
            "`0, 0, 0, 1, 0` for the 16 of SO(10). SU(N) irreps can also be entered as the row lengths of their "
            "Young diagram, e.g. `2, 1`. An irrep is marked with a bar when its Dynkin labels are lexicographically "
            "smaller than those of its conjugate.")
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
        demo.load(compute, inputs=inputs, outputs=[result, text])
    return demo


demo = build_demo()

if __name__ == '__main__':
    demo.launch()
