const { chromium } = require('@playwright/test');
const fs = require('fs');
(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', args: ['--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
  try {
    for (const size of [{width:1440,height:900}, {width:390,height:844}]) {
      const page = await browser.newPage({ viewport: size });
      const errors = []; page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:47631');
      await page.waitForSelector('#image-file');
      const bytes = fs.readFileSync('../demo/product-cutout.png');
      await page.locator('#image-file').setInputFiles({name:'input.png',mimeType:'image/png',buffer:bytes});
      await page.waitForFunction(() => !document.querySelector('#generate').disabled);
      await page.evaluate(async () => {
        const file = new File([new Uint8Array([137,80,78,71])], 'pasted.png', {type:'image/png'});
        const data = new DataTransfer(); data.items.add(file);
        document.dispatchEvent(new ClipboardEvent('paste', {clipboardData:data, bubbles:true}));
      });
      await page.waitForFunction(() => document.querySelector('#selected').textContent.includes('pasted.png'));
      await page.evaluate(() => {
        const data = new DataTransfer(); data.items.add(new File([new Uint8Array([137,80,78,71])], 'dropped.png', {type:'image/png'}));
        document.querySelector('#drop-zone').dispatchEvent(new DragEvent('drop', {dataTransfer:data, bubbles:true, cancelable:true}));
      });
      await page.waitForFunction(() => document.querySelector('#selected').textContent.includes('dropped.png'));
      await page.locator('#image-file').focus();
      if (await page.evaluate(() => document.activeElement.id) !== 'image-file') throw Error('keyboard focus failed');
      await page.screenshot({path:`studio-${size.width}.png`,fullPage:true});
      if (errors.length) throw Error(errors.join('; '));
      await page.close();
    }
    console.log('Desktop/mobile image select, paste, drop and keyboard focus pass; no generation invoked.');
  } finally { await browser.close(); }
})().catch(e => {console.error(e);process.exitCode=1});
