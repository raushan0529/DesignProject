import time, glob, os, shutil
from playwright.sync_api import sync_playwright
os.makedirs('/home/claude/video_raw',exist_ok=True)
for f in glob.glob('/home/claude/video_raw/*'): os.remove(f)
if os.path.exists('alerts.jsonl'): os.remove('alerts.jsonl')
SPEED=2.5
with sync_playwright() as p:
    b=p.chromium.launch()
    ctx=b.new_context(viewport={'width':1280,'height':720},record_video_dir='/home/claude/video_raw',record_video_size={'width':1280,'height':720})
    pg=ctx.new_page();errs=[];pg.on('pageerror',lambda e:errs.append(str(e)))
    pg.goto('http://127.0.0.1:5055/');pg.wait_for_timeout(500)
    pg.evaluate("showOverlay('title')");pg.wait_for_timeout(7000)
    pg.evaluate("showOverlay('arch')");pg.wait_for_timeout(14000)
    pg.evaluate("showOverlay(null)");pg.evaluate(f"startDemo({SPEED})")
    pg.wait_for_function("window.finished===true",timeout=240000);pg.wait_for_timeout(3000)
    pg.evaluate("showOverlay('res')");pg.wait_for_timeout(18000)
    pg.evaluate("showOverlay('end')");pg.wait_for_timeout(11000)
    print('errors',errs or 'none')
    ctx.close();b.close()
v=glob.glob('/home/claude/video_raw/*.webm')[0];print(v,os.path.getsize(v))
shutil.copy(v,'/home/claude/video_raw/demo.webm')
