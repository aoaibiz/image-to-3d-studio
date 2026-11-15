import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const viewport = document.querySelector('#viewport');
const status = document.querySelector('#status');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0xdce1dc);
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.append(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, .001, 10000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = false;
const hemisphere = new THREE.HemisphereLight(0xffffff, 0x6b756d, 2);
const key = new THREE.DirectionalLight(0xffffff, 2.2);
const fill = new THREE.DirectionalLight(0xffffff, .8);
key.position.set(4, 8, 6);
fill.position.set(-4, 2, -5);
scene.add(hemisphere, key, fill);
let model = null, radius = 1, source = null, sourceURL = null, sha = '';
let view = 'front', mode = 'material', lighting = 'studio', selection = 0;
let dessertEnvironment = null, contactShadow = null;
const center = new THREE.Vector3();
const directions = {
  home: [.13, .31, 1],
  front: [0, 0, 1], back: [0, 0, -1], left: [-1, 0, 0], right: [1, 0, 0],
  top: [0, 1, 0], bottom: [0, -1, 0], 'front-right': [1, .35, 1],
  'front-left': [-1, .35, 1], 'back-right': [1, .35, -1], 'back-left': [-1, .35, -1]
};
function setupBackdrop(box, size) {
  const dessert = model.userData.imageTo3dProfile === 'layered-dessert';
  if (dessert && !dessertEnvironment) {
    const room = new RoomEnvironment(), generator = new THREE.PMREMGenerator(renderer);
    try {
      dessertEnvironment = generator.fromScene(room, .025, .1, 100, { size: 64 });
    } finally {
      generator.dispose();
      room.dispose();
    }
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 128;
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createRadialGradient(64, 64, 2, 64, 64, 64);
    gradient.addColorStop(0, 'rgba(72,65,57,.25)');
    gradient.addColorStop(.45, 'rgba(72,65,57,.12)');
    gradient.addColorStop(1, 'rgba(72,65,57,0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 128, 128);
    const material = new THREE.MeshBasicMaterial({
      map: new THREE.CanvasTexture(canvas), transparent: true, depthWrite: false, toneMapped: false,
    });
    contactShadow = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
    contactShadow.rotation.x = -Math.PI / 2;
    scene.add(contactShadow);
  }
  if (dessert) {
    scene.environment = dessertEnvironment.texture;
    contactShadow.position.set(center.x, box.min.y - size.y * .001, center.z);
    contactShadow.scale.set(size.x * .48, size.z * .48, 1);
  } else if (dessertEnvironment) {
    scene.environment = null;
    dessertEnvironment.dispose();
    dessertEnvironment = null;
    scene.remove(contactShadow);
    disposeObject(contactShadow);
    contactShadow = null;
  }
  scene.environmentIntensity = dessert ? .17 : 1;
  hemisphere.groundColor.setHex(dessert ? 0xb3aaa0 : 0x6b756d);
  renderer.setClearColor(dessert ? 0xeee9e1 : 0xdce1dc);
  setLighting(lighting);
}
function homeView() { return dessertEnvironment ? 'home' : 'front'; }
function render() { renderer.render(scene, camera); }
function resize() {
  const width = viewport.clientWidth, height = viewport.clientHeight;
  renderer.setSize(width, height, false);
  const span = radius * (dessertEnvironment ? 1.15 : 1.5), aspect = width / height;
  camera.left = -span * aspect; camera.right = span * aspect;
  camera.top = span; camera.bottom = -span;
  camera.updateProjectionMatrix(); render();
}
window.addEventListener('resize', resize);
controls.addEventListener('change', render);
function setView(name) {
  view = name;
  controls.target.copy(center);
  const dir = new THREE.Vector3(...directions[name]).normalize();
  camera.up.set(0, 1, 0);
  if (name === 'top') camera.up.set(0, 0, -1);
  if (name === 'bottom') camera.up.set(0, 0, 1);
  camera.position.copy(center).addScaledVector(dir, radius * 3.8);
  camera.zoom = 1; camera.near = Math.max(radius / 1000, .00001);
  camera.far = radius * 100; resize(); controls.update(); render();
  updateState();
}
function setMode(next) {
  mode = next;
  model?.traverse(object => {
    if (!object.isMesh) return;
    if (next !== 'material' && !object.geometry.getAttribute('normal')) object.geometry.computeVertexNormals();
    if (!object.userData.originalMaterial) object.userData.originalMaterial = object.material;
    object.userData.override?.dispose(); object.userData.override = null;
    if (next === 'material') object.material = object.userData.originalMaterial;
    else {
      object.userData.override = new THREE.MeshStandardMaterial({ color: next === 'clay' ? 0xf5f5f2 : 0x252d28, roughness: 1, wireframe: next === 'wire', side: THREE.DoubleSide });
      object.material = object.userData.override;
    }
  });
  render(); updateState();
}
function setLighting(next) {
  lighting = next;
  const studio = dessertEnvironment ? [1.42, 1.68, .65] : [2, 2.2, .8];
  const values = next === 'soft' ? [2.6, .7, .5] : next === 'side' ? [.6, 3, .2] : studio;
  [hemisphere.intensity, key.intensity, fill.intensity] = values;
  key.position.set(...(next === 'side' ? [6, 2, 0] : [4, 8, 6]));
  render(); updateState();
}
function updateState() {
  if (source) window.viewerState = { sha, file: source.name, view, mode, lighting };
}
// Numeric camera observations are exposed for repeatable browser testing.
window.viewerTextureCount = () => { let count = 0; model?.traverse(object => { const materials = object.userData.originalMaterial || object.material; for (const material of (Array.isArray(materials) ? materials : [materials])) if (material?.map?.image) count++; }); return count; };
window.viewerCamera = () => ({ azimuth: controls.getAzimuthalAngle(), polar: controls.getPolarAngle(), zoom: camera.zoom, target: controls.target.toArray() });
function disposeObject(root) {
  root?.traverse(object => {
    object.geometry?.dispose(); object.userData.override?.dispose();
    const materials = object.userData.originalMaterial || object.material;
    for (const material of (Array.isArray(materials) ? materials : [materials])) {
      if (!material) continue;
      for (const value of Object.values(material)) if (value?.isTexture) value.dispose();
      material.dispose();
    }
  });
}
function checkEmbeddedGLB(bytes) {
  const data = new DataView(bytes);
  if (data.byteLength < 20 || data.getUint32(0, true) !== 0x46546c67 || data.getUint32(4, true) !== 2 || data.getUint32(8, true) !== data.byteLength || data.getUint32(16, true) !== 0x4e4f534a) throw Error('GLB v2 形式ではありません');
  const length = data.getUint32(12, true);
  if (20 + length > bytes.byteLength) throw Error('GLB の JSON が不正です');
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, 20, length)));
  function check(value) {
    if (!value || typeof value !== 'object') return;
    for (const [key, child] of Object.entries(value)) {
      if (key.toLowerCase() === 'uri' && (typeof child !== 'string' || !child.startsWith('data:'))) throw Error('外部 URI を含む GLB は開けません');
      check(child);
    }
  }
  check(json);
}
async function openFile(file, expectedSha) {
  if (!file) return;
  const current = ++selection;
  status.textContent = '読み込み中…';
  try {
    const bytes = await file.arrayBuffer();
    checkEmbeddedGLB(bytes);
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    const nextHash = Array.from(new Uint8Array(digest), x => x.toString(16).padStart(2, '0')).join('');
    if (expectedSha !== undefined && nextHash !== expectedSha) throw Error('検品記録とGLBが一致しません');
    const manager = new THREE.LoadingManager();
    manager.setURLModifier(url => {
      if (!url.startsWith('blob:') && !url.startsWith('data:')) throw Error('外部リソースは禁止されています');
      return url;
    });
    const gltf = await new GLTFLoader(manager).parseAsync(bytes, '');
    if (current !== selection) { disposeObject(gltf.scene); return; }
    const box = new THREE.Box3().setFromObject(gltf.scene), size = box.getSize(new THREE.Vector3());
    if (box.isEmpty() || !Number.isFinite(size.length()) || size.length() === 0) { disposeObject(gltf.scene); throw Error('表示可能な形状がありません'); }
    if (model) { scene.remove(model); disposeObject(model); }
    model = gltf.scene; scene.add(model); box.getCenter(center);
    radius = Math.max(size.x, size.y, size.z) / 2;
    setupBackdrop(box, size);
    source = file; sha = nextHash;
    if (sourceURL) URL.revokeObjectURL(sourceURL);
    sourceURL = URL.createObjectURL(file);
    const link = document.querySelector('#download');
    link.href = sourceURL; link.download = file.name; link.hidden = false;
    document.querySelector('#hash').textContent = 'SHA-256: ' + sha;
    setMode(mode); setView(homeView());
    status.textContent = `${file.name} を表示中。正面は +Z。SHA-256: ${sha}`;
    return sha;
  } catch (error) {
    if (current === selection) status.textContent = '読み込めません: ' + error.message;
  }
}
window.openViewerFile = openFile;
document.querySelector('#file').addEventListener('change', event => openFile(event.target.files[0]));
document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => setView(button.dataset.view)));
document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => setMode(button.dataset.mode)));
document.querySelectorAll('[data-light]').forEach(button => button.addEventListener('click', () => setLighting(button.dataset.light)));
document.querySelector('#reset').addEventListener('click', () => setView(homeView()));
function markCustom() { view = 'custom'; updateState(); }
renderer.domElement.addEventListener('pointerdown', markCustom);
renderer.domElement.addEventListener('wheel', markCustom);
document.querySelector('#capture').addEventListener('click', () => {
  if (!source) { status.textContent = '先に GLB を選択してください。'; return; }
  render(); const link = document.createElement('a');
  link.download = `${sha}_${view}_${mode}_${lighting}.png`;
  link.href = renderer.domElement.toDataURL('image/png'); link.click();
});
resize();

if (window.embeddedGLB) {
  const { name, base64 } = window.embeddedGLB;
  const binary = atob(base64);
  const bytes = Uint8Array.from(binary, character => character.charCodeAt(0));
  openFile(new File([bytes], name, { type: 'model/gltf-binary' }));
}
