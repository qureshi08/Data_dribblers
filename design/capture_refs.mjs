// Full page screenshots of the reference products, for the design study (FootballCV/design/REFERENCES.md).
import fs from 'fs'; import { createRequire } from 'module';
const require = createRequire('D:/Anas Personal Projects/Anas_Qureshi.ai/Quitnafs/Instagram/RealisticPosts/package.json');
const puppeteer = require('puppeteer');
const OUT = 'D:/Anas Personal Projects/Anas_Qureshi.ai/FootballCV/design/references';
fs.mkdirSync(OUT, { recursive: true });
const PAGES = {
  veo_analytics: 'https://www.veo.com/product/veo-analytics-2',
  veo_editor: 'https://www.veo.com/product/veo-editor',
  veo_home: 'https://www.veo.com/',
  veo_appstore: 'https://apps.apple.com/us/app/veo-camera/id1446591213',
  soccerment_atlas: 'https://soccerment.com/products/sics-plus/atlas',
  soccerment_xvalue: 'https://blog.soccerment.com/xvalue/',
};
const b = await puppeteer.launch({ headless: 'new' });
for (const [name, url] of Object.entries(PAGES)) {
  const p = await b.newPage();
  await p.setViewport({ width: 1440, height: 900 });
  try {
    await p.goto(url, { waitUntil: 'networkidle2', timeout: 60000 });
    // scroll through so lazy images and scroll animations load
    const h = await p.evaluate(() => document.body.scrollHeight);
    for (let y = 0; y < h; y += 700) { await p.evaluate(y => window.scrollTo(0, y), y); await new Promise(r => setTimeout(r, 350)); }
    await p.evaluate(() => window.scrollTo(0, 0)); await new Promise(r => setTimeout(r, 800));
    await p.evaluate(() => { for (const el of document.querySelectorAll('[class*="cookie" i],[id*="cookie" i],[class*="consent" i],[id*="onetrust" i]')) el.remove(); });
    await p.screenshot({ path: `${OUT}/${name}.png`, fullPage: true });
    console.log('ok', name, h);
  } catch (e) { console.log('fail', name, e.message); }
  await p.close();
}
await b.close();
