const { chromium } = require('@playwright/test');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawn, execFileSync } = require('child_process');
const zlib = require('zlib');
const { pathToFileURL } = require('url');
const [input, outArg] = process.argv.slice(2);
if (!input || !outArg) {
  console.error('Usage: node proof.cjs <file.glb|--synthetic> <output-dir>');
  process.exit(1);
}
function png() {
  function crc(buffer) {
    let value = 0xffffffff;
    for (const byte of buffer) { value ^= byte; for (let i = 0; i < 8; i++) value = (value >>> 1) ^ (0xedb88320 & -(value & 1)); }
    return (value ^ 0xffffffff) >>> 0;
  }
  function chunk(type, bytes) {
    const label = Buffer.from(type), out = Buffer.alloc(bytes.length + 12);
    out.writeUInt32BE(bytes.length); label.copy(out, 4); bytes.copy(out, 8);
    out.writeUInt32BE(crc(Buffer.concat([label, bytes])), bytes.length + 8); return out;
  }
  const header = Buffer.alloc(13); header.writeUInt32BE(2); header.writeUInt32BE(2, 4);
  header[8] = 8; header[9] = 6;
  const row = Buffer.from([0, 240, 35, 15, 255, 240, 35, 15, 255]);
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]), chunk('IHDR', header), chunk('IDAT', zlib.deflateSync(Buffer.concat([row,row]))), chunk('IEND', Buffer.alloc(0))]);
}
function fixture(uri, sceneExtras) {
  const image = png();
  const vertices = Buffer.alloc(60 + Math.ceil(image.length / 4) * 4);
  image.copy(vertices, 60);
  [0, 1, 0, -1, -1, 0, 1, -1, 0, .5, 1, 0, 0, 1, 0].forEach((v, i) => vertices.writeFloatLE(v, i * 4));
  const doc = {
    asset: { version: '2.0', generator: 'synthetic textured test fixture' }, scene: 0,
    scenes: [{ nodes: [0], extras: sceneExtras }], nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, material: 0 }] }],
    materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 }, metallicFactor: 0, roughnessFactor: 1 }, doubleSided: true }],
    textures: [{ source: 0 }], images: [uri ? { uri } : { bufferView: 2, mimeType: 'image/png' }],
    buffers: [{ byteLength: vertices.length }], bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 36 }, { buffer: 0, byteOffset: 36, byteLength: 24 }, { buffer: 0, byteOffset: 60, byteLength: image.length }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', min: [-1, -1, 0], max: [1, 1, 0] }, { bufferView: 1, componentType: 5126, count: 3, type: 'VEC2' }]
  };
  let json = Buffer.from(JSON.stringify(doc));
  json = Buffer.concat([json, Buffer.alloc((4 - json.length % 4) % 4, 0x20)]);
  const result = Buffer.alloc(12 + 8 + json.length + 8 + vertices.length);
  result.write('glTF'); result.writeUInt32LE(2, 4); result.writeUInt32LE(result.length, 8);
  result.writeUInt32LE(json.length, 12); result.write('JSON', 16); json.copy(result, 20);
  const offset = 20 + json.length;
  result.writeUInt32LE(vertices.length, offset); result.write('BIN\0', offset + 4);
  vertices.copy(result, offset + 8); return result;
}
function waitForServer(server) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(Error('server timeout')), 10000);
    const finish = callback => value => { clearTimeout(timer); callback(value); };
    server.stdout.once('data', finish(chunk => {
      const match = chunk.toString().match(/http:\/\/127\.0\.0\.1:\d+/);
      if (!match) reject(Error('server did not announce port')); else resolve(match[0]);
    }));
    server.once('error', finish(reject));
    server.once('exit', finish(code => reject(Error('server exited ' + code))));
  });
}
(async () => {
  const synthetic = input === '--synthetic';
  const data = synthetic ? fixture() : fs.readFileSync(input);
  const sha = crypto.createHash('sha256').update(data).digest('hex');
  const out = path.resolve(outArg);
  fs.mkdirSync(out, { recursive: true });
  const server = spawn(process.execPath, ['server.cjs'], { cwd: __dirname, stdio: 'pipe', env: { ...process.env, PORT: '0' } });
  let browser;
  try {
    const base = await waitForServer(server);
    const options = { headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] };
    const installed = ['C:/Program Files/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'].find(fs.existsSync);
    if (installed) options.executablePath = installed;
    browser = await chromium.launch(options);
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const errors = [], external = [], failed = [];
    page.on('requestfailed', request => failed.push(request.url() + ': ' + request.failure()?.errorText));
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    page.on('request', request => { if (!request.url().startsWith(base) && !request.url().startsWith('blob:') && !request.url().startsWith('data:')) external.push(request.url()); });
    await page.goto(base);
    const denied = await page.request.get(base + '/package.json');
    if (denied.status() !== 404) throw Error('server exposed non-allowlisted file');
    await page.locator('#file').setInputFiles({ name: 'external.glb', mimeType: 'model/gltf-binary', buffer: fixture('https://example.invalid/tracker.png') });
    await page.waitForFunction(() => document.querySelector('#status').textContent.includes('外部 URI'));
    await page.locator('#file').setInputFiles({ name: synthetic ? 'synthetic-fixture.glb' : path.basename(input), mimeType: 'model/gltf-binary', buffer: data });
    await page.waitForFunction(expected => window.viewerState?.sha === expected, sha, { timeout: 30000 });
    const canvas = page.locator('canvas'), box = await canvas.boundingBox();
    const textureCount = await page.evaluate(() => window.viewerTextureCount());
    if (textureCount < 1) throw Error('expected texture not loaded');
    const materialImage = await canvas.screenshot();
    if (synthetic) {
      const dessert = fixture(undefined, { imageTo3dProfile: 'layered-dessert' });
      await page.locator('#file').setInputFiles({ name: 'dessert.glb', mimeType: 'model/gltf-binary', buffer: dessert });
      await page.waitForFunction(() => window.viewerState?.view === 'home');
      const home = await page.evaluate(() => window.viewerCamera());
      if (Math.abs(home.azimuth - Math.atan(.13)) > .00001
        || Math.abs(home.polar - Math.atan2(Math.hypot(1, .13), .31)) > .00001) throw Error('incorrect dessert home camera');
      await page.locator('[data-view=front]').click();
      if (Math.abs((await page.evaluate(() => window.viewerCamera())).azimuth) > .00001) throw Error('front preset failed');
      await page.locator('#reset').click();
      if ((await page.evaluate(() => window.viewerState.view)) !== 'home') throw Error('dessert reset failed');
      await page.locator('#file').setInputFiles({ name: 'synthetic-fixture.glb', mimeType: 'model/gltf-binary', buffer: data });
      await page.waitForFunction(expected => window.viewerState?.sha === expected && window.viewerState.view === 'front', sha);
      if (!(await canvas.screenshot()).equals(materialImage)) throw Error('generic backdrop or camera changed after dessert');
    }
    await page.locator('[data-mode=clay]').click();
    const clayImage = await canvas.screenshot();
    const clayMetrics = await page.evaluate(() => {
      const source = document.querySelector('canvas');
      const copy = document.createElement('canvas'); copy.width = source.width; copy.height = source.height;
      const context = copy.getContext('2d'); context.drawImage(source, 0, 0);
      const pixels = context.getImageData(0, 0, copy.width, copy.height).data;
      const bg = [pixels[0], pixels[1], pixels[2]];
      let count = 0, sum = 0, square = 0, bright = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        const delta = Math.abs(pixels[i] - bg[0]) + Math.abs(pixels[i+1] - bg[1]) + Math.abs(pixels[i+2] - bg[2]);
        if (delta < 25) continue;
        const l = .2126 * pixels[i] + .7152 * pixels[i+1] + .0722 * pixels[i+2];
        count++; sum += l; square += l*l; if (l > 45) bright++;
      }
      const mean = sum / count;
      return { count, mean, standardDeviation: Math.sqrt(Math.max(0, square / count - mean * mean)), brightFraction: bright / count };
    });
    if (!synthetic && (clayMetrics.count < 100 || clayMetrics.mean < 45 || clayMetrics.brightFraction < .5 || clayMetrics.standardDeviation < 2)) throw Error('clay appears black or indistinguishable: ' + JSON.stringify(clayMetrics));
    if (synthetic && materialImage.equals(clayImage)) throw Error('textured material did not render differently from clay');
    await page.locator('[data-mode=material]').click();
    const state = () => page.evaluate(() => window.viewerCamera());
    const before = await state();
    let previous = before.azimuth, total = 0;
    for (let i = 0; i < 8; i++) {
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.mouse.down();
      await page.mouse.move(box.x + box.width / 2 + 110, box.y + box.height / 2, { steps: 8 });
      await page.mouse.up();
      const next = (await state()).azimuth;
      let delta = next - previous;
      if (delta > Math.PI) delta -= 2 * Math.PI;
      if (delta < -Math.PI) delta += 2 * Math.PI;
      total += delta; previous = next;
    }
    if (Math.abs(total) < 2 * Math.PI) throw Error('drag did not complete a full rotation: ' + total);
    await page.mouse.wheel(0, -300);
    const afterZoom = await state();
    if (Math.abs(afterZoom.zoom - before.zoom) < .01) throw Error('wheel zoom unchanged');
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down({ button: 'right' });
    await page.mouse.move(box.x + box.width / 2 + 80, box.y + box.height / 2, { steps: 5 });
    await page.mouse.up({ button: 'right' });
    const panned = await state();
    await page.locator('#reset').click();
    const reset = await state();
    const dist = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));
    if (dist(panned.target, before.target) < .001 || dist(reset.target, before.target) > .00001 || Math.abs(reset.zoom - 1) > .00001) throw Error('pan/reset failed');
    await page.locator('[data-light="side"]').click();
    if ((await page.evaluate(() => window.viewerState.lighting)) !== 'side') throw Error('lighting failed');
    await page.locator('[data-light="studio"]').click();
    const downloadPromise = page.waitForEvent('download');
    await page.locator('#download').click();
    const download = await downloadPromise, temp = path.join(out, 'download-check.glb');
    await download.saveAs(temp);
    if (!fs.readFileSync(temp).equals(data)) throw Error('download differs');
    fs.unlinkSync(temp);
    const views = ['front', 'back', 'left', 'right', 'top', 'bottom', 'front-right', 'front-left', 'back-right', 'back-left'];
    const files = [];
    for (const mode of ['material', 'clay']) {
      await page.locator(`[data-mode="${mode}"]`).click();
      for (const view of views) {
        await page.locator(`[data-view="${view}"]`).click();
        const name = `${sha}_${view}_${mode}.png`;
        await canvas.screenshot({ path: path.join(out, name) }); files.push(name);
      }
    }
    if (errors.length || external.length || failed.length) throw Error(JSON.stringify({ errors, external, failed }));
    {
      const glbPath = synthetic ? path.join(out, 'synthetic-textured.glb') : path.resolve(input);
      const htmlPath = path.join(out, synthetic ? 'synthetic-offline.html' : 'offline-preview.html');
      if (synthetic) fs.writeFileSync(glbPath, data);
      execFileSync(process.execPath, ['make-preview.cjs', glbPath, htmlPath], { cwd: __dirname });
      const offline = await browser.newPage({ viewport: { width: 1280, height: 900 } });
      const requests = [], offlineErrors = [], offlineFailed = [];
      offline.on('request', request => requests.push(request.url()));
      offline.on('requestfailed', request => offlineFailed.push(request.url()));
      offline.on('pageerror', error => offlineErrors.push(error.message));
      offline.on('console', message => { if (message.type() === 'error') offlineErrors.push(message.text()); });
      await offline.goto(pathToFileURL(htmlPath).href);
      await offline.waitForFunction(expected => window.viewerState?.sha === expected, sha, { timeout: 30000 });
      const offlineTextureCount = await offline.evaluate(() => window.viewerTextureCount());
      if (synthetic && offlineTextureCount < 1) throw Error('offline texture did not load');
      const actual = await offline.locator('canvas').screenshot();
      await offline.locator('[data-mode=clay]').click();
      const clay = await offline.locator('canvas').screenshot();
      if (synthetic && actual.equals(clay)) throw Error('offline textured material not visible');
      const network = requests.filter(url => !url.startsWith('file:') && !url.startsWith('blob:') && !url.startsWith('data:'));
      if (network.length || offlineErrors.length || offlineFailed.length) throw Error(JSON.stringify({ network, offlineErrors, offlineFailed }));
      await offline.close();
    }
    const manifest = { synthetic, input: synthetic ? 'synthetic fixture' : path.basename(input), sha256: sha, front: '+Z', projection: 'orthographic', lighting: 'studio', textureCount, clayMetrics, dragRadians: total, zoomBefore: before.zoom, zoomAfter: afterZoom.zoom, panTarget: panned.target, resetTarget: reset.target, downloadByteIdentical: true, externalURIRejected: true, externalRequests: external, failedRequests: failed, offlinePreviewVerified: true, texturedFixtureVerified: synthetic, files };
    fs.writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(manifest, null, 2));
    console.log(JSON.stringify({ ...manifest, files: files.length, output: out }));
  } finally {
    if (browser) await browser.close();
    server.kill();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
