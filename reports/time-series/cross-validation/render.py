"""Render the editable manuscript to a network-free HTML report.
Requires Pandoc 3.1.11.1 (tested); Python standard library only.
"""
from __future__ import annotations
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent

class Audit(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []; self.anchors = []; self.math = 0; self.external_assets = []
    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if 'id' in d: self.ids.append(d['id'])
        if tag == 'a' and d.get('href','').startswith('#'): self.anchors.append(d['href'][1:])
        if tag == 'math': self.math += 1
        if tag in ('script','img','iframe') and 'src' in d: self.external_assets.append(d['src'])
        if tag == 'link' and d.get('rel') == 'stylesheet': self.external_assets.append(d.get('href'))

def main() -> None:
    pandoc = shutil.which('pandoc')
    if not pandoc: raise RuntimeError('Install Pandoc before rendering')
    template = (ROOT/'template.html').read_text(encoding='utf-8').replace('/*INLINE_STYLE*/', (ROOT/'style.css').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'template.html'
        path.write_text(template, encoding='utf-8')
        result = subprocess.run([pandoc, str(ROOT/'SOURCE.md'), '--from=markdown+raw_html+fenced_divs',
            '--to=html5', '--standalone', '--toc', '--toc-depth=2', '--mathml',
            '--highlight-style=pygments', '--template='+str(path)],
            text=True, capture_output=True, check=True)
    if 'Could not convert TeX math' in result.stderr or '[ERROR]' in result.stderr:
        raise RuntimeError(result.stderr)
    text = re.sub(r'(<table(?:\s[^>]*)?>.*?</table>)', r'<div class="table-wrap">\1</div>',result.stdout, flags=re.S)
    text = re.sub(r'(<math(?=[^>]*display="block")[^>]*>.*?</math>)', r'<span class="math display">\1</span>', text, flags=re.S)
    # Some Pandoc builds classify unary minus as an identifier.
    text = text.replace('<mi>−</mi>', '<mo>−</mo>')
    (ROOT/'index.html').write_text(text,encoding='utf-8')
    audit=Audit();audit.feed(text)
    missing=sorted(set(audit.anchors)-set(audit.ids))
    assert not missing, f'Broken internal anchors: {missing}'
    assert len(audit.ids)==len(set(audit.ids)), 'Duplicate element IDs'
    assert audit.math > 100, 'Missing MathML rendering'
    assert not audit.external_assets, f'Unexpected remote assets: {audit.external_assets}'
    assert 'math-error' not in text.lower()
    info={'pandoc':subprocess.check_output([pandoc,'--version'],text=True).splitlines()[0],
          'pandoc_warnings':result.stderr.strip(),
          'mathml_elements':audit.math,'internal_links':len(audit.anchors),
          'missing_anchors':missing,'external_runtime_assets':audit.external_assets,
          'source_sha256':hashlib.sha256((ROOT/'SOURCE.md').read_bytes()).hexdigest(),
          'html_sha256':hashlib.sha256((ROOT/'index.html').read_bytes()).hexdigest(),
          'html_bytes':len(text.encode('utf-8'))}
    (ROOT/'build_validation.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))

if __name__=='__main__': main()
