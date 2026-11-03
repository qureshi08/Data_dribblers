// Screenshots the match page on every tab, plus a player spotlight, for review: node snap.mjs <run> [width]
// Needs the `footballcv` preview (port 4810). Writes runs/<run>/_ui/*.png and prints any page errors.
import fs from 'fs'; import { createRequire } from 'module';
const require = createRequire('D:/Anas Personal Projects/Anas_Qureshi.ai/Quitnafs/Instagram/RealisticPosts/package.json');
const puppeteer = require('puppeteer');
const run = process.argv[2] || 'sample_720', width = +process.argv[3] || 1440, at = +process.argv[4] || 1.5;
const out = `D:/Anas Personal Projects/Anas_Qureshi.ai/FootballCV/runs/${run}/_ui`; fs.mkdirSync(out, { recursive: true });
const b = await puppeteer.launch({ headless: 'new' }), p = await b.newPage(), errors = [];
p.on('pageerror', e => errors.push(e.message));
await p.setViewport({ width, height: width < 800 ? 900 : 1000, deviceScaleFactor: 1 });
await p.goto(`http://localhost:4810/${run}/report.html`, { waitUntil: 'networkidle2' });
await p.evaluate(async t => { const v = document.getElementById('vid'); await new Promise(r => v.readyState >= 2 ? r() : v.addEventListener('loadeddata', r, { once: true }));
  await new Promise(r => { v.addEventListener('seeked', r, { once: true }); v.currentTime = t; }); await new Promise(r => setTimeout(r, 700)); }, at);
await p.screenshot({ path: `${out}/overview.png` });
for (const tab of await p.evaluate(() => TABS)) {
  await p.evaluate(t => show(t), tab); await new Promise(r => setTimeout(r, 300));
  await (await p.$('.panel')).screenshot({ path: `${out}/tab_${tab}.png` });
}
await p.evaluate(() => { const i = idxAt(document.getElementById('vid').currentTime); SPOT = holder(i) ?? FR[i][1].find(q => ROLE[q[0]] && ROLE[q[0]][1] !== 'referee')[0]; });
await new Promise(r => setTimeout(r, 400));
await (await p.$('.player')).screenshot({ path: `${out}/spotlight.png` });
console.log('errors', JSON.stringify(errors));
await b.close();
