// Run against a previously generated Studio job. No generation is submitted.
// node viewer/test-local-paint.cjs <loopback-port> <job-id> <mode>
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { readFile } = require('node:fs/promises');
const { chromium } = require('@playwright/test');

async function main() {
  const [port, id, mode] = process.argv.slice(2);
  assert(/^\d{1,5}$/.test(port) && Number(port) > 0 && Number(port) <= 65535);
  assert(/^[\da-f]{8}(?:-[\da-f]{4}){3}-[\da-f]{12}$/.test(id));
  assert(['local_continuity', 'native_material', 'review_hold', 'chart_local', 'chart_generated', 'chart_hold'].includes(mode));
  const origin = 'http://127.0.0.1:' + port;
  const browser = await chromium.launch({
    channel: 'chrome', headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
  });
  try {
    const page = await browser.newPage({ isMobile: true, hasTouch: true, viewport: {width: 390, height: 740} });
    const path = '/api/jobs/' + id;
    const read = async suffix => {
      const response = await page.request.get(origin + path + suffix);
      assert(response.ok()); return response.json();
    };
    const { job } = await read('');
    const chart = mode.startsWith('chart_');
    const hold = mode === 'review_hold' || mode === 'chart_hold';
    const local = mode === 'local_continuity';
    assert.equal(job.status, hold ? 'review_hold' : 'needs_review');
    const settings = hold ? null : await read('/artifacts/settings.json');
    if (settings && !chart) assert.equal(settings.appearance_mode, mode);
    if (chart) {
      assert.equal(job.config.pipeline, 'triposg-codex-surface-chart-v1');
      if (!hold) assert.equal(job.surface_chart.decision, mode === 'chart_local' ? 'local_continuity' : 'inferred_chart');
    }
    const expected = hold ? job.hold.preview_sha256 : settings.glb_sha256;
    const errors = [];
    page.on('pageerror', err => errors.push(err.message));
    await page.goto(origin + '/?job=' + id);
    await page.waitForFunction(sha => window.viewerState?.sha === sha, expected);
    if (!hold) await page.waitForFunction(() => document.querySelector('#job-state').textContent.includes('形式検査'));
    const links = page.locator('#local-paint-files');
    assert.equal(await links.isVisible(), local);
    const notice = await page.locator('#job-state').innerText();
    assert.equal(notice.includes('塗装用の外部画像生成は0回'), local || mode === 'chart_local');
    assert.equal(await page.locator('#chart-result').isVisible(), chart && !hold);
    if (chart && !hold) {
      assert.equal(await page.locator('#chart-result').getAttribute('href'), path + '/artifacts/chart-result.json');
      const report = await read('/artifacts/chart-result.json');
      assert.equal(report.output_sha256, expected);
      assert.equal(report.paint_attempts, mode === 'chart_local' ? 0 : 1);
    }
    if (mode === 'chart_hold') {
      assert(notice.includes(job.hold.reason));
      assert.equal((await page.request.get(origin + path + '/artifacts/model.glb')).status(), 404);
    }
    if (local) {
      assert((await page.locator('.inspection-title').innerText()).includes('元写真から模様を補ったモデル'));
      for (const [i, name] of ['continuity-paint.json', 'continuity-native.glb'].entries()) {
        const href = await links.locator('a').nth(i).getAttribute('href');
        assert.equal(href, path + '/artifacts/' + name);
        const data = await (await page.request.get(origin + href)).body();
        assert.equal(createHash('sha256').update(data).digest('hex'), settings.artifact_sha256[name]);
      }
    }
    const pending = page.waitForEvent('download');
    await page.locator('#download').click();
    const bytes = await readFile(await (await pending).path());
    assert.equal(createHash('sha256').update(bytes).digest('hex'), expected);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({id, mode, sha256: expected}));
  } finally {
    await browser.close();
  }
}
main().catch(err => { console.error(err); process.exitCode = 1; });
