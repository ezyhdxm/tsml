#!/usr/bin/env python3
"""Render the published tutorial from frozen results. Does NOT train a model.

The AST loader imports only deterministic surface generation and original
plotting functions from the companion experiment, without importing PyTorch.
"""
from __future__ import annotations
import ast
import hashlib
import json
import re
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from html.parser import HTMLParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reproduction'
CHAPTERS = [
    '00-surface-foundations.md', '00a-why-surface.md',
    '00b-classical-methods.md', '01-vae-foundations.md',
    '02a-literature.md', '02b-literature.md', '02c-literature.md',
    '03-comparison-and-reproducibility.md', '04-experiment.md',
    '05-implementation-and-research.md', '06-references-and-appendices.md',
]


def load_original_functions() -> dict:
    source = (ROOT / 'reproduce_vol_surface_vae.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    wanted = {'Config', 'ssvi', 'calls_np', 'arb_flags', 'make_data', 'save_plots'}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise RuntimeError('Original experiment function inventory changed')
    days = np.array([14, 30, 60, 90, 120, 180], dtype=float)
    k = np.array([-.3, -.2, -.1, 0., .1, .2, .3], dtype=float)
    scope = {'__name__': '__main__', 'np': np, 'pd': pd, 'plt': plt,
             'ndtr': ndtr, 'dataclass': dataclass, 'TENOR_DAYS': days,
             'TENORS': days / 365., 'KLOG': k, 'NT': 6, 'NK': 7, 'P': 42}
    exec(compile(ast.Module(body=nodes, type_ignores=[]),
                 'selected_original_functions', 'exec'), scope)
    return scope


def frozen_figures() -> None:
    OUT.mkdir(exist_ok=True)
    results = json.loads((OUT / 'results.json').read_text())
    # round_trip preserves every saved binary float, not rounded table values.
    hist = pd.read_csv(OUT / 'training_history.csv', float_precision='round_trip')
    cells = pd.read_csv(OUT / 'sample_completion.csv', float_precision='round_trip')
    truth = cells.true_iv.to_numpy().reshape(6, 7)
    sample = {'truth': truth,
              'observed': np.where(cells.observed.to_numpy().reshape(6, 7), truth, np.nan)}
    for name in ('PCA-8', 'MLP-VAE', 'ConvVAE', 'ConvVAE+NA'):
        sample[name] = cells[name].to_numpy().reshape(6, 7)
    scope = load_original_functions()
    scope['save_plots'](OUT, truth, hist, results['completion'], results['generation'],
                       np.asarray(results['latent_factor_audit']['correlation_matrix']), sample)
    # Reuse the checked-in frozen data byte for byte. Regenerating even a
    # deterministic sample can change floating-point bytes across CPU/library
    # builds; rendering must not replace the published experimental dataset.
    provenance = json.loads((ROOT / 'publication.json').read_text())
    for rel, expected in provenance['frozen_sha256'].items():
        actual = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Frozen artifact checksum mismatch: {rel}: {actual}')
    runpy_source = (ROOT / 'make_tutorial_figures.py').read_text()
    exec(compile(runpy_source, 'make_tutorial_figures.py', 'exec'), {'__file__': str(ROOT / 'make_tutorial_figures.py')})


class HTMLAudit(HTMLParser):
    def __init__(self):
        super().__init__(); self.math = 0; self.images = []; self.errors = 0
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'math': self.math += 1
        if tag == 'img': self.images.append(a.get('src', ''))
        if tag == 'merror': self.errors += 1


def main() -> None:
    frozen_figures()
    import classical_demo
    classical_demo.run(ROOT / 'classical_examples')
    text = ''.join((ROOT / 'manuscript' / name).read_text(encoding='utf-8') for name in CHAPTERS)
    (ROOT / 'report.md').write_text(text, encoding='utf-8')
    cmd = ['pandoc', 'report.md', '-f', 'markdown+tex_math_dollars+raw_html+markdown_in_html_blocks',
           '-t', 'html5', '--standalone', '--template', 'report_template.html5', '--mathml',
           '--toc', '--toc-depth=3', '--embed-resources', '--css', 'report.css',
           '--include-in-header', 'header.html', '--include-before-body', 'before.html',
           '--include-after-body', 'after.html', '-o', 'index.html']
    run = subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)
    if 'Could not convert' in run.stderr or 'Could not fetch' in run.stderr:
        raise RuntimeError(run.stderr)
    html = (ROOT / 'index.html').read_text(encoding='utf-8')
    html = re.sub(r'(<math display="block"[^>]*>.*?</math>)', r'<div class="math-scroll">\1</div>', html, flags=re.S)
    html = re.sub(r'<table>(.*?)</table>',
                  r'<div class="table-wrap"><table class="data-table">\1</table></div>',
                  html, flags=re.S)
    (ROOT / 'index.html').write_text(html, encoding='utf-8')
    audit = HTMLAudit(); audit.feed(html)
    ast_run = subprocess.run(['pandoc', 'report.md', '-f',
        'markdown+tex_math_dollars+raw_html+markdown_in_html_blocks', '-t', 'json'],
        cwd=ROOT, check=True, capture_output=True, text=True)
    def count_math(node):
        if isinstance(node, dict):
            return int(node.get('t') == 'Math') + sum(count_math(v) for v in node.values())
        if isinstance(node, list):
            return sum(count_math(v) for v in node)
        return 0
    expected_math = count_math(json.loads(ast_run.stdout))
    body_audit = HTMLAudit()
    body = re.search(r'<main\b[^>]*>(.*?)</main>', html, flags=re.S)
    assert body is not None, 'Main content element missing'
    body_audit.feed(body.group(1))
    assert body_audit.math == expected_math, f'MathML/AST mismatch: {body_audit.math}/{expected_math}'
    assert len(audit.images) == 14, f'Expected 14 image occurrences, found {len(audit.images)}'
    assert not audit.errors, 'MathML error element found'
    assert all(src.startswith('data:image/') for src in audit.images), 'Non-embedded image found'
    summary = {'mathml_equations': audit.math, 'body_mathml_equations': body_audit.math, 'image_occurrences': len(audit.images),
               'all_images_embedded': True, 'models_retrained': False,
               'revision': 'v2-classical-motivation',
               'teaching_examples_validated': True,
               'new_sections': ['3A: why surfaces and models', '3B: classical methods and VAE value'],
               'mathml_matches_pandoc_ast': True,
               'html_sha256': hashlib.sha256((ROOT / 'index.html').read_bytes()).hexdigest()}
    (ROOT / 'build_validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    # Repository bundle intentionally excludes original model checkpoints.
    names = [p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
             and '.transport' not in p.parts and p.suffix not in {'.zip', '.pt', '.pyc'}]
    (ROOT / 'MANIFEST.sha256').write_text(''.join(
        f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n'
        for p in sorted(names) if p.name != 'MANIFEST.sha256'))
    with zipfile.ZipFile(ROOT / 'public_reproduction_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(names + [ROOT / 'MANIFEST.sha256'])):
            z.write(p, p.relative_to(ROOT))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
