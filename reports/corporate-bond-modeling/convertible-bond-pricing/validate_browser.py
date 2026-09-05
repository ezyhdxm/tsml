"""Validate rendered MathML, responsive layout, and synthetic browser pricing."""
import json, os, tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parent
SCREENSHOTS=Path(tempfile.mkdtemp(prefix='cb-review-'))
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path=os.environ.get('CHROMIUM_PATH'),args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.set_content((R/'index.html').read_text());page.wait_for_timeout(600)
    page.screenshot(path=str(SCREENSHOTS/'desktop.png'))
    evals=page.evaluate('''()=>({math:document.querySelectorAll('math').length,mathHeight:document.querySelector('math[display="block"]').getBoundingClientRect().height,base:CBTeaching.treePrice(),closed:CBTeaching.exactPrice(),all:['european','american_zero','american','call','put','both'].map(t=>[t,CBTeaching.treePrice(50,.3,.025,t,600),CBTeaching.treePrice(50,.3,.025,t,1200)]),width:document.documentElement.scrollWidth,viewport:innerWidth,canvasCount:[...document.querySelectorAll('canvas')].map(c=>[c.id,c.width,c.height]),rawMath:[...document.querySelectorAll('span.math')].filter(x=>!x.querySelector('math')).length})''')
    page.locator('#default').scroll_into_view_if_needed();page.evaluate('scrollBy(0,500)');page.screenshot(path=str(SCREENSHOTS/'math.png'))
    page.locator('#pricingLab').scroll_into_view_if_needed();page.screenshot(path=str(SCREENSHOTS/'lab.png'))
    page.select_option('#contractInput','both');page.locator('#hazInput').evaluate('(e)=>{e.value=10;e.dispatchEvent(new Event("input"))}');page.wait_for_timeout(300)
    evals['stress_display']=page.locator('#labPrice').inner_text()
    page.set_viewport_size({'width':390,'height':844});page.goto('about:blank');page.set_content((R/'index.html').read_text());page.wait_for_timeout(300);page.screenshot(path=str(SCREENSHOTS/'mobile.png'))
    evals['mobile_width']=page.evaluate('document.documentElement.scrollWidth');evals['mobile_viewport']=page.evaluate('innerWidth')
    page.locator('#pricingLab').scroll_into_view_if_needed();page.screenshot(path=str(SCREENSHOTS/'mobile_lab.png'))
    evals['mobile_table_overflow_is_contained']=page.evaluate("[...document.querySelectorAll('.tablewrap')].every(e=>e.getBoundingClientRect().right<=innerWidth+1)")
    # Numerical extreme corners in the intended interactive range.
    evals['corner_prices']=page.evaluate("[15,100].flatMap(s=>[.1,.6].flatMap(v=>[0,.1].flatMap(h=>['european','american_zero','american','call','put','both'].map(t=>CBTeaching.treePrice(s,v,h,t)))))")
    evals['js_errors']=errors
    assert not errors,errors
    assert abs(evals['base']-111.64964856142917)<1e-9
    assert abs(evals['closed']-111.66558266333283)<2e-5
    expected=json.loads((R/'results.json').read_text())['scenarios']
    evals['python_javascript_max_abs_difference']=max(abs(row[i+1]-expected[j][key]) for j,row in enumerate(evals['all']) for i,key in enumerate(['price_600','price_1200']))
    assert evals['python_javascript_max_abs_difference']<1e-9
    assert evals['rawMath']==0
    assert evals['mobile_width']==evals['mobile_viewport']
    assert evals['width']==evals['viewport']
    assert all(x>0 for x in evals['corner_prices'])
    (R/'browser_validation.json').write_text(json.dumps(evals,indent=2))
    print(json.dumps(evals,indent=2))
    print('Review screenshots:', SCREENSHOTS)
    browser.close()
