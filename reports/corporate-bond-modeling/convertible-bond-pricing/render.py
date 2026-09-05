"""Render core Markdown plus foundations into offline, self-contained HTML.
Python standard library and Pandoc >=2.17 only; rendering does not run simulations.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parent
CHAPTERS = [
    '01-black-scholes.md', '02-hedging-frictions.md',
    '03-variable-volatility.md', '04-stochastic-volatility.md',
    '05-experiments-and-reading.md',
]


def render():
    source = (ROOT/'SOURCE.md').read_text(encoding='utf-8')
    marker = '## 05 · 提前转股、回售与赎回：一个节点的决策 {#exercise}'
    assert source.count(marker) == 1, 'Core chapter insertion point changed'
    foundations = '\n\n'.join((ROOT/'foundations'/name).read_text(encoding='utf-8') for name in CHAPTERS)
    source = source.replace(marker, foundations+'\n\n'+marker, 1)
    route = '### 阅读路线\n'
    addition = '\n**新增基础专题**：[BS1：自融资复制](#bs-replication) → [BS3：离散对冲](#bs-discrete) → [BS4：bid–ask 与交易成本](#bs-costs) → [BS5–BS7：时变、局部和随机波动率](#bs-time-vol) → [BS9：实际运行的对冲实验](#bs-experiments)。专题放在第 4 节之后，原有章节与链接保留。\n'
    assert route in source
    source = source.replace(route, route+addition, 1)
    output = subprocess.run([
        'pandoc', '-f', 'markdown+tex_math_dollars+fenced_divs+raw_html',
        '-t', 'html5', '--standalone', '--mathml', '--toc', '--toc-depth=2',
        '--template', str(ROOT/'template.html'), '--metadata', 'lang=zh-CN',
        '--metadata', 'title=Convertible Bond Pricing and Black-Scholes Foundations'
    ], input=source, check=True, text=True, capture_output=True)
    text = output.stdout
    if output.stderr.strip():
        print(output.stderr)
    js = (ROOT/'interactive.js').read_text(encoding='utf-8').replace(
        '@@RESULTS_JSON@@', (ROOT/'results.json').read_text(encoding='utf-8'))
    js += '\n' + (ROOT/'hedging_lab.js').read_text(encoding='utf-8').replace(
        '@@HEDGING_RESULTS@@', (ROOT/'hedging_results.json').read_text(encoding='utf-8'))
    text = text.replace('@@STYLE@@', (ROOT/'style.css').read_text(encoding='utf-8')).replace('@@INTERACTIVE@@', js)
    text = re.sub(r'<table(\s[^>]*)?>', lambda m: '<div class="tablewrap">'+m.group(), text).replace('</table>', '</table></div>')
    text = re.sub(r'\s+id="toc-[^"]*"', '', text)
    # Normalize unary signs across Pandoc/texmath builds.
    text = text.replace('<mi>−</mi>', '<mo>−</mo>').replace('<mi>+</mi>', '<mo>+</mo>')
    text = re.sub(r'(<math display="block".*?</math>)', r'<span class="math display">\1</span>', text, flags=re.S)
    markup = text.split('<script>')[0]
    ids = re.findall(r'\bid="([^"]+)"', markup)
    links = re.findall(r'href="#([^"]+)"', markup)
    checks = {
        'mathml_count': len(re.findall(r'<math\b', text)),
        'foundation_source_files': CHAPTERS,
        'foundation_section_count': len(re.findall(r'^## BS[1-9] ', foundations, re.M)),
        'duplicate_ids': sorted({x for x in ids if ids.count(x)>1}),
        'missing_local_targets': sorted(set(links)-set(ids)),
        'unresolved_placeholders': re.findall(r'@@[A-Z_]+@@', text),
        'external_runtime_assets': re.findall(r'<(?:script|link|img)\b[^>]*(?:src|href)="https?://[^\"]*"', text),
        'pandoc_version': subprocess.check_output(['pandoc','--version'], text=True).splitlines()[0],
        'sha256': hashlib.sha256(text.encode()).hexdigest(),
    }
    (ROOT/'index.html').write_text(text, encoding='utf-8')
    (ROOT/'build_validation.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    assert checks['mathml_count'] > 400 and checks['foundation_section_count'] == 9, checks
    assert not any(checks[k] for k in ['duplicate_ids','missing_local_targets','unresolved_placeholders','external_runtime_assets']), checks
    assert '<merror' not in text, 'MathML parse error'
    assert 'Could not convert TeX' not in output.stderr, output.stderr
    hedging = json.loads((ROOT/'hedging_results.json').read_text())
    assert len(hedging['checks']) == 9 and all(hedging['checks'].values())
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    render()
