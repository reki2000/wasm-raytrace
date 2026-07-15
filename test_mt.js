// Headless correctness + scaling check for the multithreaded row renderer.
// Runs entirely under node worker_threads (same shared-memory + atomics
// mechanism browsers use for Worker + SharedArrayBuffer), so it validates
// the actual threading primitives without needing a browser.
const { Worker } = require('worker_threads');
const fs = require('fs');
const path = require('path');

const W = 480, H = 270;
const T = 0.5, AZ = -0.85, EL = 0.30, DIST = 5.6;
const MT_PAGES = 8388608 / 65536; // must match build.sh's --initial-memory/--max-memory

async function renderST() {
  const bytes = fs.readFileSync('dino.wasm');
  const { instance } = await WebAssembly.instantiate(bytes);
  const e = instance.exports;
  e.render(T, AZ, EL, DIST, W, H);
  return Buffer.from(new Uint8Array(e.memory.buffer, e.fb(), W * H * 4));
}

function spawnWorkers(n, wasmPath, mem) {
  const workers = [];
  for (let id = 0; id < n; id++) {
    workers.push(new Worker(path.join(__dirname, 'test_mt_worker.js'), {
      workerData: { wasmPath, mem, id },
    }));
  }
  return Promise.all(workers.map(w => new Promise(resolve => {
    w.once('message', () => resolve(w));
  })));
}

// One MT session (module + shared memory + worker pool) reused across frames.
async function makeMtSession(numWorkers) {
  const wasmPath = path.join(__dirname, 'dino-mt.wasm');
  const bytes = fs.readFileSync(wasmPath);
  const mem = new WebAssembly.Memory({ initial: MT_PAGES, maximum: MT_PAGES, shared: true });
  const { instance } = await WebAssembly.instantiate(bytes, { env: { memory: mem } });
  const e = instance.exports;
  const workers = await spawnWorkers(numWorkers, wasmPath, mem);

  async function renderFrame(t, az, el, dist, w, h) {
    e.renderPrep(t);
    e.frameBegin(h);
    const pending = workers.map(wk => new Promise(resolve => {
      wk.once('message', (msg) => { if (msg.type === 'rowsDone') resolve(); });
    }));
    for (const wk of workers) wk.postMessage({ type: 'go', az, el, dist, w, h });
    e.renderRowsSteal(az, el, dist, w, h); // main participates too
    await Promise.all(pending);
    return Buffer.from(new Uint8Array(mem.buffer, e.fb(), w * h * 4));
  }

  function close() { for (const wk of workers) wk.postMessage({ type: 'quit' }); }
  return { renderFrame, close };
}

(async () => {
  console.log(`correctness + scaling check: ${W}x${H}, nproc=${require('os').cpus().length}`);

  const stFb = await renderST();

  // correctness: MT with 0, 1, 3 workers (1, 2, 4 total participants) must
  // match the single-threaded framebuffer exactly, since rows are disjoint
  // and deterministic.
  for (const numWorkers of [0, 1, 3]) {
    const sess = await makeMtSession(numWorkers);
    const mtFb = await sess.renderFrame(T, AZ, EL, DIST, W, H);
    const same = stFb.equals(mtFb);
    console.log(`  ${numWorkers + 1} participants: ${same ? 'OK byte-identical' : 'MISMATCH'}`);
    if (!same) {
      let firstDiff = -1;
      for (let i = 0; i < stFb.length; i++) if (stFb[i] !== mtFb[i]) { firstDiff = i; break; }
      console.log(`    first differing byte at offset ${firstDiff}`);
    }
    sess.close();
  }

  // scaling: average ms/frame over N frames for each participant count.
  const N = 60;
  console.log(`\nscaling (${N} frames each, ${W}x${H}):`);
  {
    const bytes = fs.readFileSync('dino.wasm');
    const { instance } = await WebAssembly.instantiate(bytes);
    const e = instance.exports;
    e.render(0, AZ, EL, DIST, W, H);
    const t0 = process.hrtime.bigint();
    for (let i = 0; i < N; i++) e.render(i / 60, AZ - i * 0.002, EL, DIST, W, H);
    const ms = Number(process.hrtime.bigint() - t0) / 1e6 / N;
    console.log(`  ST (baseline):     ${ms.toFixed(2)} ms/frame`);
  }
  for (const numWorkers of [0, 1, 3, 7]) {
    const sess = await makeMtSession(numWorkers);
    await sess.renderFrame(0, AZ, EL, DIST, W, H);
    const t0 = process.hrtime.bigint();
    for (let i = 0; i < N; i++) await sess.renderFrame(i / 60, AZ - i * 0.002, EL, DIST, W, H);
    const ms = Number(process.hrtime.bigint() - t0) / 1e6 / N;
    console.log(`  MT ${numWorkers + 1} participants: ${ms.toFixed(2)} ms/frame`);
    sess.close();
  }
  console.log('\ndone');
  process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
