const http = require('http');
const fs = require('fs');
const path = require('path');
const routes = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/viewer.js', ['viewer.js', 'text/javascript; charset=utf-8']],
  ['/node_modules/three/build/three.module.js', ['node_modules/three/build/three.module.js', 'text/javascript; charset=utf-8']],
  ['/node_modules/three/build/three.core.js', ['node_modules/three/build/three.core.js', 'text/javascript; charset=utf-8']],
  ['/node_modules/three/examples/jsm/loaders/GLTFLoader.js', ['node_modules/three/examples/jsm/loaders/GLTFLoader.js', 'text/javascript; charset=utf-8']],
  ['/node_modules/three/examples/jsm/controls/OrbitControls.js', ['node_modules/three/examples/jsm/controls/OrbitControls.js', 'text/javascript; charset=utf-8']],
  ['/node_modules/three/examples/jsm/utils/BufferGeometryUtils.js', ['node_modules/three/examples/jsm/utils/BufferGeometryUtils.js', 'text/javascript; charset=utf-8']]
]);
const port = process.env.PORT === undefined ? 4173 : Number(process.env.PORT);
http.createServer((req, res) => {
  if (req.method !== 'GET') return res.writeHead(405).end();
  const route = routes.get(new URL(req.url, 'http://localhost').pathname);
  if (!route) return res.writeHead(404).end();
  fs.readFile(path.join(__dirname, route[0]), (error, data) => {
    if (error) return res.writeHead(404).end();
    res.writeHead(200, {
      'Content-Type': route[1], 'X-Content-Type-Options': 'nosniff',
      'Content-Security-Policy': "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'unsafe-inline'; img-src blob: data:; connect-src blob: data:; base-uri 'none'; form-action 'none'"
    });
    res.end(data);
  });
}).listen(port, '127.0.0.1', function () {
  console.log(`http://127.0.0.1:${this.address().port}`);
});
