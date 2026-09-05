"""Validate the expanded offline report with Playwright/Chromium.
CHROMIUM_PATH can specify an installed Chromium binary.
Screenshots are temporary local review aids, not publication assets.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT/'hedging_results.json').read_text())
    shots = Path(tempfile.mkdtemp(prefix='cb-foundations-review-'))
    record = {'html_sha256': hashlib.sha256((ROOT/'index.html').read_bytes()).hexdigest(),
              'viewports': [], 'score_comparisons': 0}
    with sync_playwright() as p:
        options = {'headless': True, 'args': ['--no-sandbox']}
        if os.environ.get('CHROMIUM_PATH'):
            options['executable_path'] = os.environ['CHROMIUM_PATH']
        browser = p.chromium.launch(**options)
        record['browser_version'] = browser.version
        for name, width, height in [('desktop',1440,1100), ('mobile',390,844)]:
            page = browser.new_page(viewport={'width':width,'height':height}, device_scale_factor=1)
            errors, external = [], []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: external.append(request.url) if request.url.startswith(('http://','https://')) else None)
            page.goto((ROOT/'index.html').as_uri(), wait_until='load')
            page.wait_for_function('window.BSHedgingTeaching !== undefined')
            metrics = page.evaluate('''() => ({
                viewport: innerWidth, document: document.documentElement.scrollWidth,
                mathml: document.querySelectorAll('math').length,
                math_errors: document.querySelectorAll('merror').length,
                foundation_headers: [...document.querySelectorAll('h2')].filter(e=>/^BS[1-9]/.test(e.textContent)).length,
                old_exercise_section: !!document.getElementById('exercise')
            })''')
            assert metrics['document'] <= width+1, metrics
            assert metrics['mathml'] > 400 and metrics['math_errors'] == 0, metrics
            assert metrics['foundation_headers'] == 9 and metrics['old_exercise_section'], metrics
            assert page.locator('#bsBestGrid').inner_text() == '256'
            assert page.locator('#bsBestScore').inner_text() == '0.5730'
            for mode in ['constant','revelation']:
                for spread in [0,5,20]:
                    for alpha in [0,0.5,2]:
                        actual = page.evaluate('(p)=>window.BSHedgingTeaching.scoreRows(...p)', [mode,spread,alpha])
                        for a, row in zip(actual, data[mode]['rows']):
                            cost = row['mean_cost_pv']*spread/(data['parameters']['half_spread']*10000)
                            assert abs(a['cost']-cost) < 1e-12
                            assert abs(a['score']-(cost+alpha*row['gross_pv']['sd'])) < 1e-12
                        record['score_comparisons'] += len(actual)
            # Exercise actual DOM events, rather than testing only pure functions.
            page.locator('#bsSpread').evaluate('(e)=>{e.value="0";e.dispatchEvent(new Event("input"));}')
            assert page.locator('#bsBestGrid').inner_text() == '1024'
            page.locator('#bsMode').select_option('revelation')
            assert page.locator('#bsBestGrid').inner_text() == '1024'
            page.locator('#bsSpread').evaluate('(e)=>{e.value="5";e.dispatchEvent(new Event("input"));}')
            page.locator('#bsMode').select_option('constant')
            assert page.locator('#bsBestGrid').inner_text() == '256'
            canvases = page.evaluate('''() => ['bsCostChart','bsVolRiskChart'].map(id=>{
                const c=document.getElementById(id), a=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
                let visible=0; for(let i=3;i<a.length;i+=4) if(a[i]) visible++;
                return {id,width:c.width,height:c.height,visible_pixels:visible};
            })''')
            assert all(c['visible_pixels'] > 2000 for c in canvases), canvases
            page.locator('#bs-replication').scroll_into_view_if_needed()
            page.screenshot(path=str(shots/f'{name}-replication.png'))
            page.locator('#bsHedgeLab').screenshot(path=str(shots/f'{name}-hedging-lab.png'))
            page.locator('#bs-stochastic-vol').scroll_into_view_if_needed()
            page.screenshot(path=str(shots/f'{name}-stochastic-vol.png'))
            assert not errors and not external, (errors, external)
            record['viewports'].append({'name':name, **metrics, 'canvases':canvases,
                                       'javascript_errors':errors,'external_requests':external,
                                       'interactive_controls_passed':True})
            page.close()
        browser.close()
    record['passed'] = True
    (ROOT/'hedging_browser_validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False,indent=2))
    print('Review screenshots:', shots)


if __name__ == '__main__':
    main()
