// test_scene.js — offline verification of the combined scene renderer
// (scene.c): SDF herd + mesh line-up rendered together in one frame, sharing
// lighting/shadows/floor-mirror/materials. Loads all 6 Quaternius GLBs same
// as test_mesh.js, then calls renderScene() (which also drives dino_model's
// animate() internally via scenePrep) and dumps a few frames — including one
// with an SDF dino set to acrylic and a mesh dino set to metal, so the PPMs
// can be eyeballed for: both dinosaur kinds visible at once, mutual shadows,
// mutual floor reflections, and material response applied uniformly.
const fs = require('fs');
const GLB = require('./glb.js');

const MODELS = ['T-Rex','Velociraptor','Triceratops','Stegosaurus','Parasaurolophus','Apatosaurus'];
const SPACING = 3.4;

function packAll(e){
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
  new Float32Array(e.memory.buffer,e.meshPos(),pos.length).set(pos);
  new Float32Array(e.memory.buffer,e.meshNormal(),nrm.length).set(nrm);
  new Int32Array(e.memory.buffer,e.meshJoint(),jnt.length).set(jnt);
  new Float32Array(e.memory.buffer,e.meshWeight(),wgt.length).set(wgt);
  new Int32Array(e.memory.buffer,e.meshIndex(),idx.length).set(idx);
  new Float32Array(e.memory.buffer,e.meshColor(),col.length).set(col);
  new Int32Array(e.memory.buffer,e.meshTriModel(),tmdl.length).set(tmdl);
  e.meshSetCounts(vb, tb, jb);
  return built;
}
function pushBones(e, built, t){
  const view=new Float32Array(e.memory.buffer, e.meshBone(), 12*built.reduce((a,m)=>a+m.nj,0));
  built.forEach(m=>{
    let ai=m.animNames.findIndex(n=>/Idle/i.test(n)); if(ai<0)ai=0;
    view.set(GLB.sampleBones(m,ai,t), m.jbase*12);
  });
}
function dump(e,name,W,H){
  const m=new Uint8Array(e.memory.buffer,e.fb(),W*H*4), rgb=Buffer.alloc(W*H*3);
  for(let i=0,j=0;i<W*H;i++){rgb[j++]=m[i*4];rgb[j++]=m[i*4+1];rgb[j++]=m[i*4+2];}
  fs.writeFileSync(name+'.ppm', Buffer.concat([Buffer.from(`P6\n${W} ${H}\n255\n`),rgb]));
}

(async()=>{
  const {instance}=await WebAssembly.instantiate(fs.readFileSync('dino.wasm'));
  const e=instance.exports; const W=640,H=360;
  const built = packAll(e);
  console.log('packed', MODELS.length, 'mesh models; total tris',
              built.reduce((a,m)=>a+m.ntris,0), 'verts', built.reduce((a,m)=>a+m.nverts,0));
  pushBones(e,built,0.6);
  e.meshSetFocus(0,0);

  // wide shot, default materials: SDF herd (near origin) + mesh line-up
  // (spread along X, also centered near origin) should both be visible.
  e.renderScene(0.5, 0.95, 0.33, 14.0, W, H); dump(e,'scene_wide',W,H);

  // closer shot at the default (0,0) focus, a few different times so the
  // SDF herd's gait/ground-scroll and the mesh line-up's idle pose both
  // show motion-plausible frames.
  e.renderScene(5.0, -0.85, 0.30, 8.0, W, H); dump(e,'scene_f1',W,H);
  pushBones(e,built,2.0);
  e.renderScene(9.5, 0.55, 0.30, 8.0, W, H); dump(e,'scene_f2',W,H);

  // materials: one SDF dino acrylic (should show mesh dinos through it via
  // the combined retrace), one mesh dino metal (should mirror the sky/
  // ground with its own albedo tint, same formula as the SDF metal look).
  e.mat(0, 3, 0.35, 0.85, 1.49, 0.0, 0.6);   // SDF theropod: acrylic
  e.meshMat(2, 2, 0.85, 0.65, 1.49, 0.0, 0.9, 1); // mesh Triceratops: metallic
  e.renderScene(0.5, 0.3, 0.28, 9.0, W, H); dump(e,'scene_materials',W,H);

  // low, close shot toward the ground so the floor-mirror block is well
  // exercised (both dinosaur kinds should appear in the reflection).
  e.renderScene(0.5, 0.4, 0.14, 6.0, W, H); dump(e,'scene_floor',W,H);

  const t0=process.hrtime.bigint(); const N=30;
  for(let i=0;i<N;i++){ pushBones(e,built,i/30); e.renderScene(i/30, 0.95, 0.30, 10.0, 480, 270); }
  const ms=Number(process.hrtime.bigint()-t0)/1e6/N;
  console.log(`480x270 combined scene: ${ms.toFixed(2)} ms/frame (${(1000/ms).toFixed(0)} fps)`);
})().catch(e=>{console.error(e);process.exit(1);});
