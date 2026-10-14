from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch();pg=b.new_page(viewport={'width':1280,'height':720})
    errs=[];pg.on('pageerror',lambda e:errs.append(str(e)));pg.on('console',lambda m:errs.append(m.text) if m.type=='error' else None)
    pg.goto('http://127.0.0.1:5055/');pg.wait_for_timeout(800)
    pg.evaluate("startDemo(8)")
    for t,name in [(40,'a'),(100,'b'),(180,'c')]:
        pg.wait_for_function(f"window.simT>={t}",timeout=60000);pg.screenshot(path=f'/home/claude/img/dash_{name}.png')
    pg.evaluate("showOverlay('res')");pg.wait_for_timeout(500);pg.screenshot(path='/home/claude/img/dash_res.png')
    pg.evaluate("showOverlay('arch')");pg.wait_for_timeout(300);pg.screenshot(path='/home/claude/img/dash_arch.png')
    print('errors',errs or 'none');b.close()
