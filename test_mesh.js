// test_mesh.js — offline verification of the mesh path.
// Loads a Quaternius GLB, uploads it to the wasm mesh renderer, samples an
// animation, renders a frame per model/action, and dumps PPMs + timing.
const fs = require('fs');
const GLB = require('./glb.js');

function upload(e, model) {
  new Float32Array(e.memory.buffer, e.meshPos(),   model.nverts*3).set(model.pos);
  new Int32Array  (e.memory.buffer, e.meshJoint(), model.nverts*4).set(model.jnt);
  new Float32Array(e.memory.buffer, e.meshWeight(),model.nverts*4).set(model.wgt);
  new Int32Array  (e.memory.buffer, e.meshIndex(), model.ntris*3).set(model.idx);
  new Float32Array(e.memory.buffer, e.meshColor(), model.ntris*3).set(model.col);
  e.meshSetCounts(model.nverts, model.ntris, model.nj);
}
function pushBones(e, model, ai, t) {
  const b = GLB.sampleBones(model, ai, t);
  new Float32Array(e.memory.buffer, e.meshBone(), b.length).set(b);
}
function dump(e, name, W, H) {
  const m = new Uint8Array(e.memory.buffer, e.fb(), W*H*4);
  const rgb = Buffer.alloc(W*H*3);
  for (let i=0,j=0;i<W*H;i++){ rgb[j++]=m[i*4]; rgb[j++]=m[i*4+1]; rgb[j++]=m[i*4+2]; }
  fs.writeFileSync(name+'.ppm', Buffer.concat([Buffer.from(`P6\n${W} ${H}\n255\n`), rgb]));
}

(async () => {
  const { instance } = await WebAssembly.instantiate(fs.readFileSync('dino.wasm'));
  const e = instance.exports;
  const W = 480, H = 270;

  const which = process.argv[2] || 'T-Rex';
  const glb = GLB.parseGLB(fs.readFileSync(`models/${which}.glb`).buffer);
  const model = GLB.buildModel(glb);
  console.log(`${which}: verts=${model.nverts} tris=${model.ntris} joints=${model.nj}`);
  console.log('  anims:', model.animNames.map((n,i)=>`${i}:${n.split('|').pop()}`).join(' '));

  upload(e, model);
  // pick a few animations to render (idle-ish / walk / attack by index guess)
  const shots = [
    ['m_idle', 0, 1.0, 0.95, 0.27, 5.6],
    ['m_walk', 4, 0.3, 0.95, 0.27, 5.6],
    ['m_attack', model.animNames.findIndex(n=>/Attack/.test(n)), 0.6, 0.95, 0.27, 5.6],
  ];
  for (const [nm, ai, t, az, el, d] of shots) {
    if (ai<0) continue;
    pushBones(e, model, ai, t);
    e.renderMesh(az, el, d, W, H);
    dump(e, nm, W, H);
    console.log('  dumped', nm, 'anim', ai, model.animNames[ai]);
  }

  // timing (walk animation, changing time = re-skin + rebuild BVH each frame)
  const ai = 4;
  const t0 = process.hrtime.bigint(); const N = 60;
  for (let i=0;i<N;i++){ pushBones(e, model, ai, i/30); e.renderMesh(0.95+i*0.005, 0.27, 5.6, W, H); }
  const ms = Number(process.hrtime.bigint()-t0)/1e6/N;
  console.log(`  ${W}x${H}: ${ms.toFixed(2)} ms/frame (${(1000/ms).toFixed(0)} fps)`);
})().catch(e => { console.error(e); process.exit(1); });
