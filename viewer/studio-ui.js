// Loaded only by the loopback studio server. The offline GLB viewer does not load this file.
document.body.classList.add('studio');
document.querySelector('header h1').textContent = '画像から 3D';
const app = document.createElement('main');
app.id = 'studio-app';
app.innerHTML = `
  <div class="workbench">
    <h2>画像を立体にする</h2>
    <p class="intro">画像を1枚選び、生成後に回転して形を確認できます。見えない面は推定されます。生成結果の +Z は入力画像を見た方向で、被写体の正面とは限りません。画像と結果はこの端末の jobs/studio 内に保存されます。</p>
    <div class="drop" id="drop-zone" tabindex="0" aria-label="画像のドロップまたは貼り付け領域">
      <p><strong>画像をここにドロップ、または貼り付け</strong></p>
      <label class="pick">画像を選ぶ <input id="image-file" type="file" accept="image/png,image/jpeg,image/webp"></label>
      <p><small>PNG / JPEG / WebP・20 MB 以下</small></p>
    </div>
    <p id="selected" aria-live="polite">画像はまだ選ばれていません。</p>
    <div class="actions"><button class="primary" id="generate" disabled>3D を生成</button><button id="cancel" hidden>処理を中止</button><button id="resume" hidden>処理を再開</button></div>
    <div id="job-state" role="status" aria-live="polite"><p>画像を選んでください。</p></div>
    <a id="appearance-sheet" target="_blank" rel="noopener" hidden>入力画像と立体の形・色を見比べる</a>
    <figure id="pose-preview" hidden><img width="96" height="96" alt="自動で選んだポーズ"><figcaption>使用したポーズです。別の対象を作るには、その部分を切り抜いて送ってください。</figcaption></figure>
  </div>`;
document.querySelector('header').after(app);
document.querySelector('[data-view=front]').textContent = '入力側 +Z';
const title = document.createElement('h2'); title.className = 'inspection-title'; title.textContent = '3D を確認';
app.after(title);
const input = document.querySelector('#image-file');
const drop = document.querySelector('#drop-zone');
const selected = document.querySelector('#selected');
const generate = document.querySelector('#generate');
const cancel = document.querySelector('#cancel');
const resume = document.querySelector('#resume');
const state = document.querySelector('#job-state');
const lastKey = 'image3d:last-job';
let file = null, imageHash = '', jobId = '', stream = null, loadingJob = '', busy = false, jobActive = false, choice = 0, scrollOnResult = false;
const terminal = new Set(['needs_review', 'failed', 'cancelled', 'interrupted']);
const stages = { queued: '順番待ち', starting: '生成の準備中', 'initializing model': 'モデルを準備中', preprocessing: '画像を整えています', 'selecting source': '対象を選んでいます', 'finishing model': '形と塗装を仕上げています', 'resuming finish': '保存した立体から仕上げを再開しています', 'running model': '立体を推定中', 'extracting mesh': '形状を取り出しています', 'baking texture': '表面の色を作成中', inference: '立体を生成中', mesh: '形状を作成中', texture: 'テクスチャを作成中', export: 'GLB を出力中', inspection: '形式を検査中', rendering: '確認画像を作成中', appearance: '入力画像と立体を比較中', needs_review: '確認待ち', failed: '処理に失敗', cancelled: '中止', interrupted: '中断' };
function message(text) { state.textContent = text; }
function idKey(hash) { return 'image3d:finish-v2:input:' + hash; }
function showControls(job) {
  const active = job.status === 'running' || job.status === 'queued';
  jobActive = active;
  cancel.hidden = !active; resume.hidden = !['failed', 'cancelled', 'interrupted'].includes(job.status);
  generate.disabled = busy || !file || active;
}
async function choose(candidate) {
  if (!candidate) return;
  const selection = ++choice;
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(candidate.type) || candidate.size > 20 * 1024 * 1024 || !candidate.size) {
    file = null; generate.disabled = true; selected.textContent = 'PNG・JPEG・WebP の画像を20 MB以下で選んでください。'; return;
  }
  const digest = await crypto.subtle.digest('SHA-256', await candidate.arrayBuffer());
  if (selection !== choice) return;
  file = candidate; imageHash = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
  selected.textContent = `選択中: ${candidate.name || '貼り付け画像'} (${(candidate.size / 1024 / 1024).toFixed(2)} MB)`;
  document.querySelector('#pose-preview').hidden = true;
  generate.disabled = busy || jobActive;
  if (!jobId) message('画像を確認したら「3D を生成」を押してください。');
}
input.addEventListener('change', () => choose(input.files[0]));
drop.addEventListener('dragover', event => { event.preventDefault(); drop.classList.add('dragging'); });
drop.addEventListener('dragleave', () => drop.classList.remove('dragging'));
drop.addEventListener('drop', event => { event.preventDefault(); drop.classList.remove('dragging'); if (event.dataTransfer.files.length !== 1) { selected.textContent = '画像は1枚だけ選んでください。'; return; } choose(event.dataTransfer.files[0]); });
document.addEventListener('paste', event => { const images = [...event.clipboardData.files].filter(item => item.type.startsWith('image/')); if (images.length === 1) { event.preventDefault(); choose(images[0]); } else if (images.length > 1) selected.textContent = '画像は1枚だけ貼り付けてください。'; });
async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw Error(data.error || `通信エラー (${response.status})`);
  return data;
}
async function loadResult(job) {
  if (loadingJob === job.id) return;
  loadingJob = job.id;
  try {
    if (!job.artifacts.includes('model.glb') || !job.artifacts.includes('inspection.json')) throw Error('結果ファイルが揃っていません。');
    const [inspection, settings, response] = await Promise.all([
      api(`/api/jobs/${job.id}/artifacts/inspection.json`),
      api(`/api/jobs/${job.id}/artifacts/settings.json`),
      fetch(`/api/jobs/${job.id}/artifacts/model.glb`)
    ]);
    if (!response.ok) throw Error('GLB を読み込めませんでした。');
    const bytes = await response.arrayBuffer();
    if (jobId !== job.id) return;
    const result = new File([bytes], 'model.glb', { type: 'model/gltf-binary' });
    const actualHash = await window.openViewerFile(result);
    if (jobId !== job.id) return;
    if (!actualHash || actualHash !== inspection.sha256) throw Error('表示した GLB と検査結果の SHA が一致しません。');
    const labels = { pass: '合格', fail: '不合格', reject: '不合格', review: '要確認', needs_review: '要確認', pending_review: '人の確認待ち', unknown: '未確認' };
    const status = part => labels[part?.status] || '未確認';
    const manual = job.review;
    const manualText = manual?.sha256 === actualHash && ['pass', 'fail', 'review'].includes(manual.decision)
      ? '。人の確認: ' + (labels[manual.decision] || '未確認') : '';
    const finishes = {'standing-figure': '立ち姿の輪郭から造形・塗装しました。', 'wheeled-vehicle': '車輪と窓の位置から車体を造形しました。', 'layered-dessert': '写真からグラスの幅、クリームの層、果物の位置を計測して仕上げました。', 'image-guided-general': '入力の輪郭を照合し、推定した立体に汎用仕上げを適用しました。', 'inferred-mesh': '元の推定形状を保持しました。入力画像と見比べてください。'};
    const comparison = document.querySelector('#appearance-sheet');
    const sheet = inspection.image_comparison?.contact_sheet;
    comparison.hidden = !(sheet === actualHash + '_appearance-comparison.jpg' && job.artifacts.includes(sheet));
    if (!comparison.hidden) comparison.href = '/api/jobs/' + job.id + '/artifacts/' + sheet;
    const selectedPose = settings.source_selection?.detected_count ? `全${settings.source_selection.detected_count}個の候補から上段の中央に近い1個を選びました。` : '';
    const pose = document.querySelector('#pose-preview');
    pose.hidden = !selectedPose || !job.artifacts.includes('input-processed.png');
    if (!pose.hidden) pose.querySelector('img').src = `/api/jobs/${job.id}/artifacts/input-processed.png`;
    message(`${selectedPose}${finishes[settings.finish_profile] || ''}形式検査: ${status(inspection.format)}。形状: ${status(inspection.geometry)}。見た目: ${status(inspection.appearance)}${manualText}。ドラッグで全方向を確認してください。`);
    if (scrollOnResult) { scrollOnResult = false; title.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  } catch (error) { if (jobId === job.id) message(`結果を表示できません: ${error.message}`); }
  finally { if (loadingJob === job.id) loadingJob = ''; }
}
async function refresh(id) {
  const { job } = await api(`/api/jobs/${id}`);
  if (id !== jobId) return;
  showControls(job);
  if (job.status !== 'needs_review') document.querySelector('#appearance-sheet').hidden = true;
  const label = stages[job.stage] || job.stage || job.status;
  message(`${label}（ジョブ ${job.id}）${job.error ? ' — ' + job.error : ''}`);
  if (job.status === 'needs_review') await loadResult(job);
  if (terminal.has(job.status)) { stream?.close(); stream = null; }
}
function watch(id) {
  stream?.close(); jobId = id; localStorage.setItem(lastKey, id);
  stream = new EventSource(`/api/jobs/${id}/events`);
  stream.onmessage = () => refresh(id).catch(error => message('状態を確認できません: ' + error.message));
  stream.onerror = () => { if (jobId === id) message('接続が切れました。自動で再接続します。'); };
  refresh(id).catch(error => message('状態を確認できません: ' + error.message));
}
generate.addEventListener('click', async () => {
  if (!file || busy) return;
  busy = true; generate.disabled = true;
  const chosen = file, hash = imageHash;
  let key = localStorage.getItem(idKey(hash));
  if (!key) { key = crypto.randomUUID(); localStorage.setItem(idKey(hash), key); }
  message('画像を送信しています。同じ画像の再送には同じ識別子を使います。');
  try {
    const { job } = await api('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/octet-stream', 'X-Studio-Client': '1', 'Idempotency-Key': key }, body: chosen });
    scrollOnResult = true; watch(job.id);
  } catch (error) { message(`送信結果を確認できません: ${error.message}。再度「3D を生成」を押すと同じ依頼を照会します。`); }
  finally { busy = false; generate.disabled = !file || jobActive; }
});
async function action(name) {
  if (!jobId) return;
  const id = jobId; cancel.disabled = resume.disabled = true;
  try { await api(`/api/jobs/${id}/${name}`, { method: 'POST', headers: { 'X-Studio-Client': '1' } }); watch(id); }
  catch (error) { message(`操作できません: ${error.message}`); }
  finally { cancel.disabled = resume.disabled = false; }
}
cancel.addEventListener('click', () => action('cancel'));
resume.addEventListener('click', () => action('resume'));
const previous = localStorage.getItem(lastKey);
if (previous && /^[0-9a-f-]{36}$/.test(previous)) watch(previous);
