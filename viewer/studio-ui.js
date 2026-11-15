// Loaded only by the loopback studio server. The offline GLB viewer does not load this file.
document.body.classList.add('studio');
document.querySelector('header h1').textContent = '画像から 3D';
const app = document.createElement('main');
app.id = 'studio-app';
app.innerHTML = `
  <div class="workbench">
    <h2>画像を立体にする</h2>
    <p class="intro">画像を1枚選び、生成後に回転して形を確認できます。見えない面は推定されます。</p>
    <p id="engine-info" class="intro" role="status">この端末の生成方式を確認しています。</p>
    <div class="drop" id="drop-zone" tabindex="0" aria-label="画像のドロップまたは貼り付け領域">
      <p><strong>画像をここにドロップ、または貼り付け</strong></p>
      <label class="pick">画像を選ぶ <input id="image-file" type="file" accept="image/png,image/jpeg,image/webp"></label>
      <p><small>PNG / JPEG / WebP・20 MB 以下</small></p>
    </div>
    <p id="selected" aria-live="polite">画像はまだ選ばれていません。</p>
    <div class="actions"><button class="primary" id="generate" disabled>3D を生成</button><button id="cancel" hidden>処理を中止</button><button id="resume" hidden>処理を再開</button></div>
    <div id="job-state" role="status" aria-live="polite"><p>画像を選んでください。</p></div>
    <a id="appearance-sheet" target="_blank" rel="noopener" hidden>入力画像と立体の形・色を見比べる</a>
    <a id="held-sheet" target="_blank" rel="noopener" hidden>元の立体の検品画像を開く</a>
    <div id="local-paint-files" hidden>
      <a target="_blank" rel="noopener">推定塗装の検品記録</a>
      <a download="continuity-native.glb">推定塗装のGLBを保存</a>
    </div>
    <figure id="pose-preview" hidden><img width="96" height="96" alt="自動で選んだポーズ"><figcaption>使用したポーズです。別の対象を作るには、その部分を切り抜いて送ってください。</figcaption></figure>
  </div>`;
document.querySelector('header').after(app);
const front = document.querySelector('[data-view=front]');
front.textContent = '+Z 視点';
const title = document.createElement('h2'); title.className = 'inspection-title'; title.textContent = '3D を確認';
app.after(title);
const shapeLink = document.createElement('a');
shapeLink.className = 'shape-variant'; shapeLink.hidden = true;
title.after(shapeLink);
const requested = new URLSearchParams(location.search);
const input = document.querySelector('#image-file');
const drop = document.querySelector('#drop-zone');
const selected = document.querySelector('#selected');
const generate = document.querySelector('#generate');
const cancel = document.querySelector('#cancel');
const resume = document.querySelector('#resume');
const state = document.querySelector('#job-state');
const lastKey = 'image3d:last-job';
const tripoSR = 'triposr-auto-finish-v2';
const tripoSG = 'triposg-local-v1';
const guidedSG = 'triposg-codex-guided-v1';
let pipeline = null;
let file = null, imageHash = '', jobId = '', stream = null, loadingJob = '', busy = false, jobActive = false, choice = 0, scrollOnResult = false;
let heldInput = '', heldDisplay = '';
const terminal = new Set(['needs_review', 'review_hold', 'failed', 'cancelled', 'interrupted']);
const stages = { queued: '順番待ち', starting: '生成の準備中', 'initializing model': 'モデルを準備中', preprocessing: '画像を整えています', 'selecting source': '対象を選んでいます', 'finishing model': '形と塗装を仕上げています', 'resuming finish': '保存した立体から仕上げを再開しています', 'running model': '立体を推定中', 'extracting mesh': '形状を取り出しています', 'baking texture': '表面の色を作成中', inference: '立体を生成中', mesh: '形状を作成中', texture: 'テクスチャを作成中', export: 'GLB を出力中', inspection: '形式を検査中', rendering: '確認画像を作成中', appearance: '入力画像と立体を比較中', needs_review: '確認待ち', failed: '処理に失敗', cancelled: '中止', interrupted: '中断' };
function message(text) { state.textContent = text; }
Object.assign(stages, {
  'sf3d connecting': 'Stability AI 公式Spaceに接続しています', 'sf3d uploaded': '切り抜きを送信しました',
  'sf3d preprocess_waiting': '公式Spaceで前処理を待っています', 'sf3d preprocessed': '前処理が完了しました',
  'sf3d generation_waiting': '公式Spaceで立体生成を待っています', 'sf3d downloading': '材質付きの立体を取得しています',
  'sf3d complete': '公式の立体を取得しました', 'sf3d exporting': '表示用の向きと編集用データを整えています',
  'segmenting source': '背景から被写体を切り出しています',
  'sf3d observed native': 'SF3Dで材質付きの立体を生成しています',
  'sf3d observed observation': 'ChatGPTで写真の色、模様、部品を確認しています',
  'sf3d observed registration': '写真の輪郭と立体の向きを合わせています',
  'sf3d observed native_inspection': '形状と材質を正面・背面から確認しています',
  'sf3d observed guides': '形状に合わせて背面の塗装用画像を準備しています',
  'sf3d observed paint': 'OpenAIで写真に写っていない面の塗装を推定しています',
  'sf3d observed orientation': '推定した塗装と視点の向きを照合しています',
  'sf3d observed projection': '元写真に写っていない領域に塗装を合わせています',
  'sf3d observed export': '材質付きGLBと編集用データを書き出しています',
  'generating shape': '立体の形を推定しています',
  'resuming shape': '保存した計算結果から形状の生成を再開しています',
  'painting model': '入力画像をもとに立体を着色しています',
  'analyzing surface markings': '元写真の色、繰り返し模様、正面だけにある部品を読み取っています',
  'observing photo with ChatGPT': 'ChatGPTで元写真の色・模様・部品を読み取っています',
  'preparing shape guides': '立体から背面の確認画像を作っています',
  'requesting inferred paint': 'OpenAIで背面の塗装を推定しています',
  'checking hidden surfaces': '背面に正面の部品や模様が重複していないか確認しています',
  'applying inferred paint': '推定した塗装を立体に合わせています',
  'exporting guided model': '塗装済みのモデルを書き出しています'
});
function idKey(hash, engine) {
  return 'image3d:' + (engine === tripoSR ? 'finish-v2' : engine) + ':input:' + hash;
}
function updateGenerate() {
  generate.disabled = !pipeline || !file || busy || jobActive || heldInput === pipeline + ':' + imageHash;
}
async function readEngine() {
  const info = document.querySelector('#engine-info');
  try {
    const engine = await api('/api/engine', { cache: 'no-store' });
    const local = engine.local === true && [tripoSR, tripoSG].includes(engine.pipeline);
    const guided = [guidedSG, 'triposg-host-guided-v1', 'triposg-codex-observed-v1'].includes(engine.pipeline) && engine.local === false && engine.recipient === 'OpenAI ChatGPT';
    const sf3d = engine.pipeline === 'sf3d-official-preview' && engine.local === false && engine.recipient === 'Stability AI / Hugging Face';
    const observedSF3D = engine.pipeline === 'sf3d-codex-observed-v1' && engine.local === false
      && Array.isArray(engine.recipients) && engine.recipients.length === 2
      && engine.recipients[0] === 'Stability AI / Hugging Face' && engine.recipients[1] === 'OpenAI ChatGPT';
    if (!local && !guided && !sf3d && !observedSF3D) throw Error('生成方式と送信先を確認できません。');
    pipeline = engine.pipeline;
    if (observedSF3D) {
      info.textContent = '評価用のStable Fast 3Dを使用します。切り抜いた写真をHugging Face上のStability AI公式Spaceに送り、匿名枠で立体を1回生成します。OpenAIにはメタデータを除いた元写真を送り、本人のChatGPT利用枠で外観を1回解析します。塗装が必要な場合は写真と立体の確認画像を送り、画像を1回生成します。';
    } else if (sf3d) {
      info.textContent = '評価用の Stable Fast 3D を使用します。切り抜いた写真を Hugging Face 上の Stability AI 公式Spaceに送り、匿名の利用枠で材質付きの立体を生成します。';
    } else if (guided) {
      info.textContent = '形状はこの端末で生成します。背面の塗装を推定するため、元写真と立体の確認画像をOpenAIに送信し、本人のChatGPT利用枠を使用します。';
      if (pipeline === 'triposg-codex-observed-v1') info.textContent = '形状はこの端末で生成します。本人のChatGPT利用枠で、元写真の外観解析を1回、写真と形状ガイドから背面の塗装を推定する画像生成を1回、OpenAIに依頼します。';
      if (pipeline === 'triposg-host-guided-v1') info.textContent += '画像生成に対応するCodexに、このジョブを引き継いでください。';
    } else info.textContent = pipeline === tripoSG
      ? 'TripoSG は画像の切り抜き・形状の生成・着色をすべてこの端末で処理します。CPU では数時間かかる場合があります。'
      : 'TripoSR がこの端末で形状を生成し、画像に合わせて仕上げます。';
  } catch (error) {
    pipeline = null;
    info.textContent = '生成方式を取得できません。Studio の起動とバージョンを確認し、この画面を再読み込みしてください。';
  }
  updateGenerate();
}
function showControls(job) {
  const active = ['running', 'queued', 'awaiting_host_image'].includes(job.status);
  jobActive = active;
  cancel.hidden = !active; resume.hidden = !['failed', 'cancelled', 'interrupted'].includes(job.status);
  updateGenerate();
}
async function choose(candidate) {
  if (!candidate) return;
  const selection = ++choice;
  file = null; imageHash = ''; updateGenerate();
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(candidate.type) || candidate.size > 20 * 1024 * 1024 || !candidate.size) {
    file = null; generate.disabled = true; selected.textContent = 'PNG・JPEG・WebP の画像を20 MB以下で選んでください。'; return;
  }
  const digest = await crypto.subtle.digest('SHA-256', await candidate.arrayBuffer());
  if (selection !== choice) return;
  file = candidate; imageHash = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
  selected.textContent = `選択中: ${candidate.name || '貼り付け画像'} (${(candidate.size / 1024 / 1024).toFixed(2)} MB)`;
  document.querySelector('#pose-preview').hidden = true;
  updateGenerate();
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
  if (!response.ok) {
    const error = Error(data.error || `通信エラー (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return data;
}
async function loadResult(job) {
  if (loadingJob === job.id) return;
  loadingJob = job.id;
  try {
    const alternative = requested.get('job') === job.id && requested.get('view') === 'shape-alternative';
    const candidate = job.shape_hypothesis_files?.['shape-alternative.glb'];
    const available = /^[0-9a-f]{64}$/.test(candidate || '') && job.artifacts.includes('shape-alternative.glb');
    shapeLink.hidden = !available && !alternative;
    shapeLink.textContent = alternative ? '元のモデルに戻る' : '推定した形状の別案を回す';
    shapeLink.href = '?job=' + job.id + (alternative ? '' : '&view=shape-alternative');
    if (alternative) {
      title.textContent = '形状の別案（推定・要確認）';
      document.body.classList.add('shape-alternative', 'hold-no-preview');
      document.querySelector('#appearance-sheet').hidden = true;
      document.querySelector('#pose-preview').hidden = true;
      if (!available) throw Error('このジョブの形状の別案を検品記録から確認できません。');
    }
    if (!job.artifacts.includes('model.glb') || !job.artifacts.includes('inspection.json')) throw Error('結果ファイルが揃っていません。');
    const filename = alternative ? 'shape-alternative.glb' : 'model.glb';
    const [inspection, settings, response] = await Promise.all([
      api(`/api/jobs/${job.id}/artifacts/inspection.json`),
      api(`/api/jobs/${job.id}/artifacts/settings.json`),
      fetch(`/api/jobs/${job.id}/artifacts/${filename}`)
    ]);
    if (!response.ok) throw Error('GLB を読み込めませんでした。');
    const bytes = await response.arrayBuffer();
    if (jobId !== job.id) return;
    const expected = alternative ? candidate : inspection.sha256;
    const result = new File([bytes], filename, { type: 'model/gltf-binary' });
    const actualHash = await window.openViewerFile(result, expected);
    if (jobId !== job.id) return;
    if (!actualHash || actualHash !== expected) throw Error('表示した GLB と検査結果の SHA が一致しません。');
    if (alternative) {
      document.querySelector('#download').textContent = '推定形状のGLBを保存';
      document.querySelector('[data-view="top"]').click();
      document.body.classList.remove('hold-no-preview');
      message('元写真に写っていない形状や寸法を補った別案です。回転して確認してください。');
      return;
    }
    title.textContent = job.local_continuity?.external_image_generations === 0
      ? '元写真から模様を補ったモデル' : available ? '元のモデルを確認' : '3D を確認';
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
    const resultSG = (settings.pipeline || job.config?.pipeline) === tripoSG;
    const resultSF3D = ['sf3d-official-preview', 'sf3d-codex-observed-v1'].includes(settings.pipeline);
    front.textContent = resultSF3D ? '推定入力側 +Z' : resultSG ? '+Z 視点' : '入力側 +Z';
    const selectedPose = !resultSG && settings.source_selection?.detected_count ? `全${settings.source_selection.detected_count}個の候補から上段の中央に近い1個を選びました。` : '';
    const shapeNote = resultSG
      ? 'TripoSG が立体の形と向きを推定しました。入力画像に写った側は、立体を回して確認してください。'
      : settings.pipeline === 'sf3d-codex-observed-v1'
        ? ({ native_material: '元のSF3D材質を保持しました。', guided_region: '背面の塗装を推定しました。',
             local_continuity: job.local_continuity?.external_image_generations === 0
               ? '元写真の模様をもとに、見えない面の色をこの端末で推定しました。塗装用の外部画像生成は0回です。' : '',
             guided_full: '写真に写る面の塗装を保ち、2方向の形状ガイドに合う面に推定色を反映しました。' }[settings.appearance_mode] || '')
          + '写真に写っていない面と薄い部品を確認してください。'
        : resultSF3D ? 'SF3Dが形状と材質を推定しました。背面の模様と薄い部品を確認してください。' : finishes[settings.finish_profile] || '';
    const pose = document.querySelector('#pose-preview');
    pose.hidden = !selectedPose || !job.artifacts.includes('input-processed.png');
    if (!pose.hidden) pose.querySelector('img').src = `/api/jobs/${job.id}/artifacts/input-processed.png`;
    const mirrorNote = inspection.image_comparison?.opposite_paint?.status === 'review_hold'
      ? ' 正面の特徴的な柄が背面で鏡写しになっています。背面の塗装を確認してください。' : '';
    message(`${selectedPose}${shapeNote}形式検査: ${status(inspection.format)}。形状: ${status(inspection.geometry)}。見た目: ${status(inspection.appearance)}${manualText}。ドラッグで全方向を確認してください。${mirrorNote}`);
    if (scrollOnResult) { scrollOnResult = false; title.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  } catch (error) { if (jobId === job.id) message(`結果を表示できません: ${error.message}`); }
  finally { if (loadingJob === job.id) loadingJob = ''; }
}
async function loadHeld(job) {
  const hold = job.hold;
  title.textContent = '推定塗装保留・未合格';
  document.querySelector('#pose-preview').hidden = true;
  const picture = document.querySelector('#held-sheet');
  picture.hidden = !job.artifacts.includes('review-native.png');
  if (!picture.hidden) picture.href = '/api/jobs/' + job.id + '/artifacts/review-native.png';
  heldInput = job.config.pipeline + ':' + job.input_sha256;
  updateGenerate();
  if (!hold || !job.artifacts.includes('review-native.glb')) {
    document.body.classList.add('hold-no-preview');
    message('推定塗装保留・未合格。元の立体を検品記録と照合できませんでした。');
    return;
  }
  message(hold.reason + ' 推定塗装は保留です。確認用に元の立体を表示します。');
  if (heldDisplay === job.id || loadingJob === job.id) return;
  loadingJob = job.id;
  document.body.classList.add('hold-no-preview');
  try {
    const response = await fetch('/api/jobs/' + job.id + '/artifacts/review-native.glb');
    if (!response.ok) throw Error();
    const bytes = await response.arrayBuffer();
    if (jobId !== job.id) return;
    const sha = await window.openViewerFile(new File([bytes], 'review-native.glb', { type: 'model/gltf-binary' }));
    if (jobId !== job.id) return;
    if (sha !== hold.preview_sha256) throw Error();
    heldDisplay = job.id;
    document.querySelector('#download').textContent = '検品用の元モデルを保存';
    document.body.classList.remove('hold-no-preview');
    if (scrollOnResult) { scrollOnResult = false; title.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  } catch {
    if (jobId === job.id) message('推定塗装保留・未合格。元の立体を読み込めませんでした。');
  } finally {
    if (loadingJob === job.id) loadingJob = '';
  }
}
async function refresh(id) {
  const { job } = await api(`/api/jobs/${id}`);
  if (id !== jobId) return;
  showControls(job);
  const paintFiles = document.querySelector('#local-paint-files');
  paintFiles.hidden = !(job.status === 'needs_review' && job.local_continuity?.external_image_generations === 0
    && job.artifacts.includes('continuity-paint.json') && job.artifacts.includes('continuity-native.glb'));
  if (!paintFiles.hidden) {
    const [report, model] = paintFiles.querySelectorAll('a');
    report.href = `/api/jobs/${id}/artifacts/continuity-paint.json`;
    model.href = `/api/jobs/${id}/artifacts/continuity-native.glb`;
  }
  document.body.classList.toggle('review-held', job.status === 'review_hold');
  if (job.status !== 'review_hold') {
    document.querySelector('#held-sheet').hidden = true;
    if (heldInput) {
      heldDisplay = ''; heldInput = '';
      document.body.classList.toggle('hold-no-preview', job.status !== 'needs_review');
      title.textContent = '3D を確認';
      document.querySelector('#download').textContent = '元の GLB を保存';
      updateGenerate();
    }
    if (job.status === 'needs_review') document.body.classList.remove('hold-no-preview');
  }
  if (job.status !== 'needs_review') document.querySelector('#appearance-sheet').hidden = true;
  const savedStep = job.status === 'running' && /^shape step (\d+)\/(\d+)$/.exec(job.stage);
  const memoryWait = /^waiting for memory (\d+\.\d{2})\/(\d+\.\d{2})$/.exec(job.stage);
  let label = savedStep ? stages['generating shape'] : stages[job.stage] || job.stage || job.status;
  if (job.status === 'awaiting_host_image') label = '画像生成に対応するCodexで、このジョブの推定塗装を続けてください';
  if (memoryWait) label = `空きメモリを待っています（待機開始時 ${memoryWait[1]} GiB、再開の目安 ${memoryWait[2]} GiB）`;
  const progress = savedStep ? {step: Number(savedStep[1]), total: Number(savedStep[2])} : job.progress;
  const steps = job.status === 'running' && Number.isSafeInteger(progress?.step) && Number.isSafeInteger(progress?.total)
    && progress.total > 0 && progress.step >= 0 && progress.step <= progress.total
    ? `（${progress.step}/${progress.total} ステップ）` : '';
  message(`${label}${steps}（ジョブ ${job.id}）${job.error ? ' — ' + job.error : ''}`);
  if (job.status === 'needs_review') await loadResult(job);
  if (job.status === 'review_hold') await loadHeld(job);
  if (terminal.has(job.status)) { stream?.close(); stream = null; }
}
function watch(id) {
  if (jobId !== id) {
    shapeLink.hidden = true;
    document.body.classList.remove('shape-alternative');
    title.textContent = '3D を確認';
    document.querySelector('#download').textContent = '元の GLB を保存';
  }
  stream?.close(); jobId = id; localStorage.setItem(lastKey, id);
  stream = new EventSource(`/api/jobs/${id}/events`);
  stream.onmessage = () => refresh(id).catch(error => message('状態を確認できません: ' + error.message));
  stream.onerror = () => { if (jobId === id) message('接続が切れました。自動で再接続します。'); };
  refresh(id).catch(error => message('状態を確認できません: ' + error.message));
}
generate.addEventListener('click', async () => {
  if (!file || !pipeline || jobActive || busy) return;
  busy = true; generate.disabled = true;
  const chosen = file, expected = pipeline, storageKey = idKey(imageHash, expected);
  let key = localStorage.getItem(storageKey);
  if (!key) { key = crypto.randomUUID(); localStorage.setItem(storageKey, key); }
  message('この端末の Studio に画像を渡しています。');
  try {
    const { job } = await api('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/octet-stream', 'X-Studio-Client': '1', 'X-Studio-Pipeline': expected, 'Idempotency-Key': key }, body: chosen });
    scrollOnResult = true; watch(job.id);
  } catch (error) {
    if (error.status === 409) {
      await readEngine();
      message(`依頼を受け付けられませんでした。生成方式を確認してください。${error.message}`);
    } else {
      message(`送信結果を確認できません: ${error.message}。再度「3D を生成」を押すと同じ依頼を照会します。`);
    }
  } finally { busy = false; updateGenerate(); }
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
const previous = new URLSearchParams(location.search).get('job') || localStorage.getItem(lastKey);
readEngine();
if (previous && /^[0-9a-f-]{36}$/.test(previous)) watch(previous);
