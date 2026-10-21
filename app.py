# Web app for Clebsch-Gordan decompositions of SU(N) tensor products. Run with:  python app.py
# (also works as app.py of a Hugging Face Space, together with requirements.txt)

import html
import re

import gradio as gr

from sun_irrep_CG import IrrepSUN, decompose_many, format_product

MAX_N = 12
MAX_BOXES = 40       # largest Young diagram accepted for a single factor
MAX_SMALL_BOXES = 24  # every factor after the largest is cut into strips, so keep those smaller
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


def parse_irrep(n, text, mode):
    numbers = parse_numbers(text)
    if mode == PARTITION:
        return IrrepSUN.from_partition(n, numbers)
    if not numbers:
        return IrrepSUN(n, [0] * (n - 1))
    if len(numbers) != n - 1:
        raise ValueError("SU({}) irreps need {} Dynkin labels, '{}' has {}".format(n, n - 1, text, len(numbers)))
    return IrrepSUN(n, numbers)


def parse_factors(n, mode, texts):
    n = int(n)
    if not 2 <= n <= MAX_N:
        raise ValueError("N must be between 2 and {}".format(MAX_N))
    factors = [parse_irrep(n, text, mode) for text in texts if text is not None and text.strip()]
    if len(factors) < 2:
        raise ValueError("Enter at least two irreps")
    if len(factors) > MAX_FACTORS:
        raise ValueError("At most {} factors".format(MAX_FACTORS))
    boxes = sorted((irrep.n_boxes for irrep in factors), reverse=True)
    if boxes[0] > MAX_BOXES or any(b > MAX_SMALL_BOXES for b in boxes[1:]):
        raise ValueError("These Young diagrams are too large for the web app (at most {} boxes for the largest factor "
                         "and {} for the others). Use sun_irrep_CG.py directly for bigger cases."
                         .format(MAX_BOXES, MAX_SMALL_BOXES))
    return factors


# Young diagram as an inline SVG; 'currentColor' keeps it readable in light and dark themes
def young_svg(irrep, size=12):
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
    rows = ''.join(
        '<tr><td style="padding:2px 8px">{dim}</td><td style="padding:2px 8px"><code>{labels}</code></td>'
        '<td style="padding:2px 8px;text-align:right">{mult}</td><td style="padding:2px 8px">{svg}</td></tr>'
        .format(dim=irrep_name(irrep)[0], labels=irrep_name(irrep)[1], mult=mult, svg=young_svg(irrep))
        for irrep, mult in terms[:MAX_ROWS_SHOWN])
    more = ''
    if len(terms) > MAX_ROWS_SHOWN:
        more = '<p>Showing the {} largest of {} distinct irreps.</p>'.format(MAX_ROWS_SHOWN, len(terms))
    check = sum(irrep.dim * mult for irrep, mult in terms)
    return (
        '<div style="font-size:1.1em;line-height:1.8">{lhs} = {rhs}</div>'
        '<p>{count} irreps in total ({distinct} distinct). Dimension check: {total} = {check} {ok}</p>'
        '<table><tr>{inputs}</tr></table>'
        '<table style="border-collapse:collapse;margin-top:8px">'
        '<tr><th style="text-align:left;padding:2px 8px">Dim</th><th style="text-align:left;padding:2px 8px">'
        'Dynkin labels</th><th style="padding:2px 8px">Multiplicity</th>'
        '<th style="text-align:left;padding:2px 8px">Young diagram</th></tr>{rows}</table>{more}'
    ).format(lhs=lhs, rhs=rhs, count=count, distinct=len(terms), total=total_dim, check=check,
             ok='✔' if check == total_dim else '✘', inputs=inputs, rows=rows, more=more)


# Returns (html result, plain text equation) for the web page
def compute(n, mode, *texts):
    try:
        factors = parse_factors(n, mode, texts)
        terms = decompose_many(factors)
    except ValueError as error:
        return '<p style="color:#c0392b">{}</p>'.format(html.escape(str(error))), ''
    return render(factors, terms), format_product(factors, terms)


EXAMPLES = [
    [3, DYNKIN, "1, 0", "0, 1", ""],        # 3 x 3bar = 8 + 1
    [3, DYNKIN, "1, 1", "1, 1", ""],        # 8 x 8
    [3, DYNKIN, "1, 0", "1, 0", "1, 0"],    # 3 x 3 x 3
    [2, DYNKIN, "1", "2", ""],              # spin 1/2 x spin 1
    [4, DYNKIN, "1, 0, 1", "1, 0, 1", ""],  # 15 x 15 in SU(4)
    [5, DYNKIN, "0, 1, 0, 0", "0, 1, 0, 0", ""],  # 10 x 10 in SU(5)
    [6, PARTITION, "2, 1", "2, 1", ""],
]


def build_demo():
    with gr.Blocks(title="SU(N) Clebsch-Gordan decomposition") as demo:
        gr.Markdown(
            "# SU(N) Clebsch-Gordan decomposition\n"
            "Decompose a tensor product of SU(N) irreducible representations into irreducible ones, using Young "
            "tableaux (the Littlewood-Richardson rule; see Georgi, *Lie Algebras in Particle Physics*).\n\n"
            "Enter each irrep as its N−1 Dynkin labels, e.g. `1, 1` for the SU(3) octet, or as the row lengths "
            "of its Young diagram, e.g. `2, 1`. Irreps whose Dynkin labels read smaller left to right than right to "
            "left are marked with a bar.")
        with gr.Row():
            n = gr.Number(value=3, label="N", minimum=2, maximum=MAX_N, precision=0)
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
