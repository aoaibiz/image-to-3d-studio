const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const esbuild = require('esbuild');
const [input, output] = process.argv.slice(2);
if (!input || !output) {
  console.error('Usage: node make-preview.cjs <model.glb> <preview.html>');
  process.exit(1);
}
(async () => {
  const data = fs.readFileSync(input);
  if (data.readUInt32LE(0) !== 0x46546c67) throw Error('Expected GLB');
  const sha = crypto.createHash('sha256').update(data).digest('hex');
  const result = await esbuild.build({ entryPoints: [path.join(__dirname, 'viewer.js')], bundle: true, format: 'iife', platform: 'browser', write: false, legalComments: 'inline' });
  const script = result.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
  const embedded = JSON.stringify({ name: path.basename(input), base64: data.toString('base64') }).replace(/</g, '\\u003c');
  let html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
  // This bundle contains Three.js only; retain its complete distribution notice.
  const license = fs.readFileSync(path.join(__dirname, 'node_modules/three/LICENSE'), 'utf8');
  html = html.replace('<!doctype html>', '<!doctype html>\n<!-- Bundled Three.js license:\n' + license.replace(/--/g, '- -') + '\n-->');
  html = html.replace('<meta charset="utf-8">', `<meta charset="utf-8">\n  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src blob: data:; connect-src blob: data:; base-uri 'none'; form-action 'none'">`);
  html = html.replace(/  <script type="importmap">[\s\S]*?<\/script>\s*<script type="module" src="\/viewer\.js"><\/script>/, `<script>window.embeddedGLB=${embedded};</script>\n<script>${script}</script>`);
  fs.writeFileSync(output, html);
  console.log(JSON.stringify({ output: path.resolve(output), sha256: sha, glbBytes: data.length, htmlBytes: Buffer.byteLength(html) }));
})().catch(error => { console.error(error); process.exitCode = 1; });
