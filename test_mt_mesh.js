// Headless correctness + scaling check for the multithreaded mesh-path row
// renderer (same mechanism as test_mt.js, but exercising meshPrep +
// renderMeshRowsSteal instead of the SDF path).
const { Worker } = require('worker_threads');
const fs = require('fs');
const path = require('path');
const GLB = require('./glb.js');

const MODELS = ['T-Rex','Velociraptor','Triceratops','Stegosaurus','Parasaurolophus','Apatosaurus'];
const SPACING = 3.4;
const W = 480, H = 270;
const AZ = 0.95, EL = 0.30, DIST = 10.0;
const MT_PAGES = 8388608 / 65536; // must match build.sh's --initial-memory/--max-memory

function packAll(e, mem){
  const built = MODELS.map(n => GLB.buildModel(GLB.parseGLB(fs.readFileSync(`models/${n}.glb`).buffer)));
  let vb=0, tb=0, jb=0;
  const pos=[], nrm=[], jnt=[], wgt=[], idx=[], col=[], tmdl=[];
  built.forEach((m,i)=>{
    m.vbase=vb; m.jbase=jb; m.slotX=(i-(MODELS.length-1)/2)*SPACING;
    m.fit[12]+=m.slotX;
    for (let v=0; v<m.nverts; v++){
      pos.push(m.pos[v*3],m.pos[v*3+1],m.pos[v*3+2]);
      nrm.push(m.nrm[v*3],m.nrm[v*3+1],m.nrm[v*3+2]);
      jnt.push(m.jnt[v*4]+jb,m.jnt[v*4+1]+jb,m.jnt[v*4+2]+jb,m.jnt[v*4+3]+jb);
      wgt.push(m.wgt[v*4],m.wgt[v*4+1],m.wgt[v*4+2],m.wgt[v*4+3]);
    }
    for (let t=0;t<m.ntris;t++){
      idx.push(m.idx[t*3]+vb,m.idx[t*3+1]+vb,m.idx[t*3+2]+vb);
      col.push(m.col[t*3],m.col[t*3+1],m.col[t*3+2]);
      tmdl.push(i);
    }
    vb+=m.nverts; tb+=m.ntris; jb+=m.nj;
  });
  new Float32Array(mem,e.meshPos(),pos.length).set(pos);
  new Float32Array(mem,e.meshNormal(),nrm.length).set(nrm);
  new Int32Array(mem,e.meshJoint(),jnt.length).set(jnt);
  new Float32Array(mem,e.meshWeight(),wgt.length).set(wgt);
  new Int32Array(mem,e.meshIndex(),idx.length).set(idx);
  new Float32Array(mem,e.meshColor(),col.length).set(col);
  new Int32Array(mem,e.meshTriModel(),tmdl.length).set(tmdl);
  e.meshSetCounts(vb, tb, jb);
  e.meshMat(0,2,0.85,0.65,1.49,0.0,0.9,1); // T-Rex metallic, welded normals
  e.meshMat(3,3,0.35,0.85,1.49,0.0,0.6,0); // Stego acrylic, flat per-face normals
  e.meshSetFocus(0,0);
  return built;
}
function pushBones(e, mem, built, t){
  const view = new Float32Array(mem, e.meshBone(), 12*built.reduce((a,m)=>a+m.nj,0));
  built.forEach(m=>{
    let ai=m.animNames.findIndex(n=>/Idle/i.test(n)); if(ai<0)ai=0;
    view.set(GLB.sampleBones(m,ai,t), m.jbase*12);
  });
}

async function renderMeshST() {
  const bytes = fs.readFileSync('dino.wasm');
  const { instance } = await WebAssembly.instantiate(bytes);
  const e = instance.exports;
  const built = packAll(e, e.memory.buffer);
  pushBones(e, e.memory.buffer, built, 0.6);
  e.renderMesh(AZ, EL, DIST, W, H);
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

async function makeMtSession(numWorkers) {
  const wasmPath = path.join(__dirname, 'dino-mt.wasm');
  const bytes = fs.readFileSync(wasmPath);
  const mem = new WebAssembly.Memory({ initial: MT_PAGES, maximum: MT_PAGES, shared: true });
  const { instance } = await WebAssembly.instantiate(bytes, { env: { memory: mem } });
  const e = instance.exports;
  const built = packAll(e, mem.buffer);
  const workers = await spawnWorkers(numWorkers, wasmPath, mem);

  async function renderFrame(t, az, el, dist, w, h) {
    pushBones(e, mem.buffer, built, t);
    e.meshPrep();
    e.frameBegin(h);
    const pending = workers.map(wk => new Promise(resolve => {
      wk.once('message', (msg) => { if (msg.type === 'rowsDone') resolve(); });
    }));
    const args = [az, el, dist, w, h];
    for (const wk of workers) wk.postMessage({ type: 'go', fn: 'renderMeshRowsSteal', args });
    e.renderMeshRowsSteal(...args);
    await Promise.all(pending);
    return Buffer.from(new Uint8Array(mem.buffer, e.fb(), w * h * 4));
  }

  function close() { for (const wk of workers) wk.postMessage({ type: 'quit' }); }
  return { renderFrame, close };
}

(async () => {
  console.log(`mesh path: correctness + scaling check, ${W}x${H}, nproc=${require('os').cpus().length}`);

  const stFb = await renderMeshST();

  for (const numWorkers of [0, 1, 3]) {
    const sess = await makeMtSession(numWorkers);
    const mtFb = await sess.renderFrame(0.6, AZ, EL, DIST, W, H);
    const same = stFb.equals(mtFb);
    console.log(`  ${numWorkers + 1} participants: ${same ? 'OK byte-identical' : 'MISMATCH'}`);
    if (!same) {
      let firstDiff = -1;
      for (let i = 0; i < stFb.length; i++) if (stFb[i] !== mtFb[i]) { firstDiff = i; break; }
      console.log(`    first differing byte at offset ${firstDiff}`);
    }
    sess.close();
  }

  const N = 30;
  console.log(`\nscaling (${N} frames each, ${W}x${H}):`);
  {
    const bytes = fs.readFileSync('dino.wasm');
    const { instance } = await WebAssembly.instantiate(bytes);
    const e = instance.exports;
    const built = packAll(e, e.memory.buffer);
    pushBones(e, e.memory.buffer, built, 0);
    e.renderMesh(AZ, EL, DIST, W, H);
    const t0 = process.hrtime.bigint();
    for (let i = 0; i < N; i++){ pushBones(e, e.memory.buffer, built, i/30); e.renderMesh(AZ, EL, DIST, W, H); }
    const ms = Number(process.hrtime.bigint() - t0) / 1e6 / N;
    console.log(`  ST (baseline):     ${ms.toFixed(2)} ms/frame`);
  }
  for (const numWorkers of [0, 1, 3, 7]) {
    const sess = await makeMtSession(numWorkers);
    await sess.renderFrame(0, AZ, EL, DIST, W, H);
    const t0 = process.hrtime.bigint();
    for (let i = 0; i < N; i++) await sess.renderFrame(i / 30, AZ, EL, DIST, W, H);
    const ms = Number(process.hrtime.bigint() - t0) / 1e6 / N;
    console.log(`  MT ${numWorkers + 1} participants: ${ms.toFixed(2)} ms/frame`);
    sess.close();
  }
  console.log('\ndone');
  process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
