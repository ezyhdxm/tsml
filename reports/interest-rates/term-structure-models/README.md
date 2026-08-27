# Rates term-structure models

A bilingual, editable report on no-arbitrage interest-rate term-structure modeling, with every included stochastic differential equation stated together with its measure, Brownian drivers, correlations, drift and diffusion, parameter restrictions, boundary behavior, pricing implications, and simulation scheme.

## Read

- [中文 HTML](index.html) · [中文 Markdown index](SOURCE.md)
- [English HTML](index.en.html) · [English Markdown index](SOURCE.en.md)

## Scope

The report develops probability measures and numeraires; Girsanov changes and pricing PDEs; Vasicek, CIR, Hull–White, and G2++; affine Riccati equations; HJM drift restrictions and Musiela dynamics; LMM/BGM under forward and terminal measures; SOFR/OIS multi-curve modeling; DNS, Gaussian ATSM, and AFNS; calibration, exact transitions, stable discretizations, and implementation tests.

## Manuscript layout

The editable source is split under `manuscript/zh/` and `manuscript/en/`. `SOURCE.md` and `SOURCE.en.md` are the corresponding chapter indexes. Markdown is the source of truth.

Render the Chinese report from the report directory with:

```bash
pandoc manuscript/zh/*.md \
  --standalone --toc --toc-depth=3 \
  --citeproc --bibliography=references.bib \
  --mathml --css=style.css --embed-resources \
  -o index.full.html
```

Replace `zh` by `en` for the English report. The checked-in `index.html` and `index.en.html` are compact browser loaders containing the gzip-compressed standalone HTML through local payload files; the uncompressed output is reproduced by the command above.

## Validation

```bash
python validate_report.py
```

The deterministic test harness checks exact-transition moments, CIR nonnegativity, G2++ endpoint covariance, Vasicek/CIR pricing PDE residuals, the HJM drift identity, and exact initial-curve fitting for Hull–White and G2++. The report contains only public literature, abstract mathematics, and synthetic parameters.
