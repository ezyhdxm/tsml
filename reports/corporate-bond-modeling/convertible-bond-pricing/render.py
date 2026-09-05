"""Render editable Markdown to a self-contained HTML report with offline MathML.
Requires Pandoc >= 2.17; Python standard library only. No network access.
"""
from pathlib import Path
import hashlib,json,re,subprocess
ROOT=Path(__file__).resolve().parent

def render():
    output=subprocess.run(['pandoc',str(ROOT/'SOURCE.md'),'-f','markdown+tex_math_dollars+fenced_divs+raw_html','-t','html5','--standalone','--mathml','--toc','--toc-depth=2','--template',str(ROOT/'template.html'),'--metadata','lang=en','--metadata','title=Convertible Bond Pricing'],check=True,text=True,capture_output=True)
    text=output.stdout
    if output.stderr.strip(): print(output.stderr)
    js=(ROOT/'interactive.js').read_text(encoding='utf-8').replace('@@RESULTS_JSON@@',(ROOT/'results.json').read_text(encoding='utf-8'))
    text=text.replace('@@STYLE@@',(ROOT/'style.css').read_text(encoding='utf-8')).replace('@@INTERACTIVE@@',js)
    text=re.sub(r'<table(\s[^>]*)?>',lambda m:'<div class="tablewrap">'+m.group(),text).replace('</table>','</table></div>')
    text=re.sub(r' id="toc-[^"]*"','',text)
    # Normalize unary signs across Pandoc/texmath builds.
    text=text.replace('<mi>−</mi>','<mo>−</mo>').replace('<mi>+</mi>','<mo>+</mo>')
    # Pandoc MathML output is bare <math>; constrain each long equation locally.
    text=re.sub(r'(<math display="block".*?</math>)', r'<span class="math display">\1</span>', text, flags=re.S)
    markup=text.split('<script>')[0]
    ids=re.findall(r'\bid="([^"]+)"',markup)
    links=re.findall(r'href="#([^"]+)"',markup)
    checks={'mathml_count':len(re.findall(r'<math\b',text)), 'duplicate_ids':sorted({x for x in ids if ids.count(x)>1}),'missing_local_targets':sorted(set(links)-set(ids)), 'unresolved_placeholders':re.findall(r'@@[A-Z_]+@@',text), 'external_runtime_assets':re.findall(r'<(?:script|link|img)\b[^>]*(?:src|href)="https?://[^\"]*"',text), 'pandoc_version':subprocess.check_output(['pandoc','--version'],text=True).splitlines()[0]}
    checks['sha256']=hashlib.sha256(text.encode()).hexdigest()
    (ROOT/'index.html').write_text(text,encoding='utf-8')
    (ROOT/'build_validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    assert checks['mathml_count']>150, checks
    assert not any(checks[k] for k in ['duplicate_ids','missing_local_targets','unresolved_placeholders','external_runtime_assets']),checks
    assert '<merror' not in text,'MathML parse error'
    assert 'Could not convert TeX' not in output.stderr,output.stderr
    print(json.dumps(checks,ensure_ascii=False,indent=2))

if __name__=='__main__':render()
