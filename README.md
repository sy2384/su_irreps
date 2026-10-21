# lie-tensor-products

Compute the Clebsch-Gordan decomposition of tensor products of irreducible representations of simple Lie algebras:
SU(N) with Young tableaux, and SO(N), Sp(2N), G2, F4, E6, E7, E8 (as well as SU(N)) with weights.

The algorithm is the one described in Howard Georgi's book *Lie Algebras in Particle Physics*: the boxes of the second
Young diagram are labelled a, b, c, ... by row and added to the first diagram, keeping only the tableaux that pass
Georgi's checks. For SU(N) this is the Littlewood-Richardson rule.

![Web app](docs/app_screenshot.png)

## Files

| File | What it does |
| --- | --- |
| `su3_irrep_CG.py` | The original SU(3) implementation (`Irrep(n, m)`, `multiply_irreps`) |
| `sun_irrep_CG.py` | The SU(N) generalization for any N >= 2 (`IrrepSUN`, `decompose`, `decompose_many`) |
| `lie_algebra_CG.py` | Any simple Lie algebra, using weights (`LieIrrep`, `lie_algebra`, `decompose`, `decompose_many`) |
| `app.py` | Gradio web app for products of 2 to 4 irreps |
| `front_end_1.py` | Command-line prompt for SU(3) |
| `write_irrep_to_file.py` | Reads SU(3) cases from `test_cases.csv`, writes `result.txt` |
| `tests/` | pytest suite |

## Usage

```python
from sun_irrep_CG import IrrepSUN, decompose, format_product

octet = IrrepSUN(3, (1, 1))           # SU(3), Dynkin labels (1, 1)
terms = decompose(octet, octet)       # [(irrep, multiplicity), ...], largest first
print(format_product([octet, octet], terms))
# (1, 1)[8] ⊗ (1, 1)[8] = (2, 2)[27] ⊕ (3, 0)[10] ⊕ (0, 3)[10 bar] ⊕ 2 (1, 1)[8] ⊕ (0, 0)[1]

IrrepSUN.from_partition(5, (1, 1))     # build from Young diagram row lengths: the 10 of SU(5)
```

For other algebras, `lie_algebra_CG.py` has the same interface. Algebras are named `SU(N)`, `SO(N)`, `Sp(N)` (N even),
`G2`, `F4`, `E6`, `E7`, `E8`, or by Cartan type (`A3`, `B4`, `C2`, `D5`, ...), and Dynkin labels follow Bourbaki's
numbering of the simple roots:

```python
from lie_algebra_CG import LieIrrep, decompose, format_product

spinor = LieIrrep('SO(10)', (0, 0, 0, 1, 0))
print(format_product([spinor, spinor], decompose(spinor, spinor)))
# (0, 0, 0, 1, 0)[16] ⊗ (0, 0, 0, 1, 0)[16] = (0, 0, 0, 2, 0)[126] ⊕ (0, 0, 1, 0, 0)[120] ⊕ (1, 0, 0, 0, 0)[10]
```

It finds the weights of an irrep and their multiplicities with Freudenthal's formula, then decomposes the product
with the Racah-Speiser algorithm: each weight of the smaller factor is added to the highest weight of the other plus
rho, and the result is reflected into the dominant Weyl chamber, with a sign for each reflection.

An irrep is written `(Dynkin labels)[dimension]`, with `bar` when its labels are lexicographically smaller than those
of its conjugate irrep. For SU(N) the conjugate has the labels reversed, so (n, m) is barred when n < m, as in the
original SU(3) code.

## Web app

```
pip install -r requirements.txt
python app.py
```

then open http://127.0.0.1:7860. `app.py` and `requirements.txt` can also be uploaded as they are to a Hugging Face
Space (Gradio SDK).

## Tests

```
pip install pytest gradio
python -m pytest
```

The tests check known decompositions (SU(2) angular momentum, SU(3), SU(4), SU(5) tables), that dimensions add up,
commutativity, associativity and conjugation symmetry, that the original SU(3) code and the SU(N) code agree, and,
independently of the Young tableau method, that the characters (weights counted from semistandard tableaux) of each
product equal the characters of its decomposition.

For the other algebras they check the root systems and Cartan matrices, the dimensions of all fundamental irreps,
known products (SO(7), SO(8), SO(9), SO(10), Sp(4), Sp(6), G2, F4, E6, E7, E8), that the weight method agrees with
the Young tableau code for SU(N), the isomorphisms SO(5) = Sp(4) and SO(6) = SU(4), that weight multiplicities add up
to the Weyl dimension, and the same character, dimension, symmetry and singlet checks as for SU(N).
