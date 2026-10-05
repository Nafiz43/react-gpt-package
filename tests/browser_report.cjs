const puppeteer = require('puppeteer-core');
const assert = require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const path=require('node:path');
if(!process.argv[2] || !process.env.CHROME_BIN) throw new Error('Usage: CHROME_BIN=/path/to/chrome node tests/browser_report.cjs /path/to/report.html');
(async()=>{
const browser=await puppeteer.launch({executablePath:process.env.CHROME_BIN,headless:true,args:['--no-sandbox']});
try{
const page=await browser.newPage();await page.setViewport({width:1440,height:1080});
const errors=[],remote=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))remote.push(r.url());});
await page.goto(pathToFileURL(path.resolve(process.argv[2])).href);
await page.waitForSelector('.action-node');
const totals=await page.evaluate(()=>({actions:data.rows.length,best:data.rows.filter(best).length}));
const counts=await page.evaluate(()=>({nodes:document.querySelectorAll('.action-node').length,edges:document.querySelectorAll('#graph line').length,expected:data.rows.slice(0,300).reduce((n,r)=>n+labels(r).length,0)}));assert.equal(counts.nodes,Math.min(300,totals.actions));assert.equal(counts.edges,counts.expected);
await page.click('#list-tab');assert.equal(await page.$eval('#list-panel',e=>e.hidden),false);assert.equal(await page.$$eval('.list-action',es=>es.length),totals.actions);
await page.select('#quality','best');assert.equal(await page.$$eval('.list-action',es=>es.length),totals.best);
await page.type('#search','nonexistent_search_phrase');assert.equal(await page.$$eval('.list-action',es=>es.length),0);
await page.click('#clear-filters');await page.click('#graph-tab');
await page.click('.action-node');assert.match(await page.$eval('#inspector',e=>e.textContent),/Evidence from the paper/);
await page.click('#zoom-in');assert.match(await page.$eval('#graph > g',e=>e.getAttribute('transform')),/scale\(1.2\)/);await page.click('#fit');
await page.click('.category-node');assert.match(await page.$eval('#inspector',e=>e.textContent),/Inclusion criteria/);await page.click('#clear-filters');
assert.equal(await page.$eval('#inspector a.source-link',e=>e.getAttribute('href').startsWith('#source-')),true);
if(process.env.SCREENSHOT_PATH)await page.screenshot({path:process.env.SCREENSHOT_PATH,fullPage:true});
await page.setViewport({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);

assert.deepEqual(errors,[]);assert.deepEqual(remote,[]);console.log(JSON.stringify({counts,errors,remote,checks:'graph, categories, filters, list, details, zoom, mobile layout'}));
}finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
