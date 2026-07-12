// test_mesh.js — offline verification of the mesh line-up path.
// Loads all 6 Quaternius GLBs, packs them into the wasm mesh store side by side,
// samples an animation per model, sets a couple of materials, renders framing
// on each focus target, and reports timing.
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
    m.fit[12]+=m.slotX;                          // shift the model into its slot
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
  return { built, totalJ:jb };
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
  const {built}=packAll(e);
  console.log('packed', MODELS.length, 'models; total tris',
              built.reduce((a,m)=>a+m.ntris,0), 'verts', built.reduce((a,m)=>a+m.nverts,0));
  pushBones(e,built,0.6);
  // materials: give T-Rex metal, Stego acrylic, others plain/texture
  e.meshMat(0,2,0.85,0.65,1.49,0.0,0.9);   // T-Rex metallic
  e.meshMat(3,3,0.35,0.85,1.49,0.0,0.6);   // Stego acrylic
  // wide shot from far to see the row
  e.meshSetFocus(0,0);
  e.renderMesh(0.95,0.33,14.0,W,H); dump(e,'row_all',W,H);
  // focus on Triceratops (index 2)
  e.meshSetFocus(built[2].slotX,0);
  e.renderMesh(0.95,0.30,5.6,W,H); dump(e,'row_focus2',W,H);
  // focus on Stego (acrylic, index 3)
  e.meshSetFocus(built[3].slotX,0);
  e.renderMesh(0.95,0.30,5.6,W,H); dump(e,'row_focus3',W,H);

  const t0=process.hrtime.bigint(); const N=30;
  for(let i=0;i<N;i++){ pushBones(e,built,i/30); e.renderMesh(0.95,0.30,10.0,480,270); }
  const ms=Number(process.hrtime.bigint()-t0)/1e6/N;
  console.log(`480x270 full row: ${ms.toFixed(2)} ms/frame (${(1000/ms).toFixed(0)} fps)`);
})().catch(e=>{console.error(e);process.exit(1);});
