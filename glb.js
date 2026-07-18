// glb.js — minimal GLB (binary glTF) loader + skeletal-animation sampler.
// Shared by the browser demo and the Node test harness. Parses a single .glb
// (JSON + BIN chunk), flattens the skinned mesh into upload buffers for the
// wasm mesh path, and samples per-joint skin matrices (folded with a scene-fit
// transform) for any animation at time t. Linear-blend skinning; flat colors.
(function (root) {
  'use strict';

  const COMP = { 5120:{a:Int8Array,s:1}, 5121:{a:Uint8Array,s:1},
                 5122:{a:Int16Array,s:2}, 5123:{a:Uint16Array,s:2},
                 5125:{a:Uint32Array,s:4}, 5126:{a:Float32Array,s:4} };
  const NCOMP = { SCALAR:1, VEC2:2, VEC3:3, VEC4:4, MAT2:4, MAT3:9, MAT4:16 };

  function parseGLB(buf) {
    const dv = new DataView(buf);
    if (dv.getUint32(0, true) !== 0x46546C67) throw new Error('not a glb');
    const len = dv.getUint32(8, true);
    let off = 12, json = null, bin = null;
    while (off < len) {
      const clen = dv.getUint32(off, true), ctype = dv.getUint32(off + 4, true);
      const body = new Uint8Array(buf, off + 8, clen);
      if (ctype === 0x4E4F534A) json = JSON.parse(new TextDecoder().decode(body));
      else if (ctype === 0x004E4942) bin = body;
      off += 8 + clen;
    }
    return { json, bin };
  }

  // read an accessor into a plain Float64/typed view, honoring byteStride
  function readAccessor(g, bin, idx) {
    const acc = g.accessors[idx];
    const bv = g.bufferViews[acc.bufferView];
    const comp = COMP[acc.componentType];
    const nc = NCOMP[acc.type];
    const base = (bv.byteOffset || 0) + (acc.byteOffset || 0);
    const stride = bv.byteStride || comp.s * nc;
    const out = new Float64Array(acc.count * nc);
    const buf = bin.buffer, bo = bin.byteOffset;
    for (let i = 0; i < acc.count; i++) {
      const el = new comp.a(buf, bo + base + i * stride, nc);
      for (let c = 0; c < nc; c++) out[i * nc + c] = el[c];
    }
    return { data: out, nc, count: acc.count };
  }

  // ---- 4x4 column-major helpers ----
  function fromTRS(t, q, s) {
    const x=q[0],y=q[1],z=q[2],w=q[3];
    const x2=x+x,y2=y+y,z2=z+z;
    const xx=x*x2,xy=x*y2,xz=x*z2,yy=y*y2,yz=y*z2,zz=z*z2,wx=w*x2,wy=w*y2,wz=w*z2;
    const sx=s[0],sy=s[1],sz=s[2];
    const m=new Float64Array(16);
    m[0]=(1-(yy+zz))*sx; m[1]=(xy+wz)*sx; m[2]=(xz-wy)*sx;
    m[4]=(xy-wz)*sy; m[5]=(1-(xx+zz))*sy; m[6]=(yz+wx)*sy;
    m[8]=(xz+wy)*sz; m[9]=(yz-wx)*sz; m[10]=(1-(xx+yy))*sz;
    m[12]=t[0]; m[13]=t[1]; m[14]=t[2]; m[15]=1;
    return m;
  }
  function mul(a, b) {
    const o=new Float64Array(16);
    for (let c=0;c<4;c++) for (let r=0;r<4;r++){
      o[c*4+r]=a[r]*b[c*4]+a[4+r]*b[c*4+1]+a[8+r]*b[c*4+2]+a[12+r]*b[c*4+3];
    }
    return o;
  }

  // Build a reusable model from a parsed glb. Flattens all primitives into one
  // vertex/index buffer; per-triangle color = primitive material baseColorFactor.
  function buildModel(glb) {
    const g = glb.json, bin = glb.bin;
    const mesh = g.meshes[0];
    const skin = g.skins[0];
    const joints = skin.joints;                       // node ids, length = NJ
    const ibmA = readAccessor(g, bin, skin.inverseBindMatrices).data; // NJ*16
    const nj = joints.length;

    const pos = [], nrm = [], jnt = [], wgt = [], idx = [], col = [];
    let vbase = 0;
    for (const prim of mesh.primitives) {
      const P = readAccessor(g, bin, prim.attributes.POSITION);
      const J = readAccessor(g, bin, prim.attributes.JOINTS_0);
      const W = readAccessor(g, bin, prim.attributes.WEIGHTS_0);
      const N = prim.attributes.NORMAL!=null ? readAccessor(g, bin, prim.attributes.NORMAL) : null;
      const n = P.count;
      for (let i=0;i<n;i++){
        pos.push(P.data[i*3], P.data[i*3+1], P.data[i*3+2]);
        // each triangle corner is its own vertex record in this format (no
        // shared indices across faces), so the per-corner NORMAL is what
        // actually carries the model's authored smoothing groups
        nrm.push(N ? N.data[i*3] : 0, N ? N.data[i*3+1] : 1, N ? N.data[i*3+2] : 0);
        jnt.push(J.data[i*4], J.data[i*4+1], J.data[i*4+2], J.data[i*4+3]);
        // renormalize weights (guards against slight de-normalization)
        let w0=W.data[i*4],w1=W.data[i*4+1],w2=W.data[i*4+2],w3=W.data[i*4+3];
        const ws=w0+w1+w2+w3||1; wgt.push(w0/ws,w1/ws,w2/ws,w3/ws);
      }
      // color
      let c=[0.7,0.7,0.7];
      if (prim.material!=null){
        const m=g.materials[prim.material];
        const bcf=m.pbrMetallicRoughness&&m.pbrMetallicRoughness.baseColorFactor;
        // this pack's baseColorFactor values run very dark (baked from a dim
        // texture atlas); lift them toward the asset's natural on-screen brightness
        if (bcf) c=[bcf[0],bcf[1],bcf[2]].map(v=>Math.pow(Math.max(v,0),0.6));
      }
      const I = readAccessor(g, bin, prim.indices).data;
      for (let i=0;i<I.length;i+=3){
        idx.push(vbase+I[i], vbase+I[i+1], vbase+I[i+2]);
        col.push(c[0],c[1],c[2]);
      }
      vbase += n;
    }

    // Weld by position: this format never shares an index across triangles
    // (every face corner is its own vertex record), so average the authored
    // normal over every corner at the same position -- including the ones the
    // artist meant as a hard crease -- for a fully rounded, seam-free Gouraud
    // look instead of preserving the model's original faceting.
    {
      const groups = new Map();
      const nv = pos.length/3;
      for (let i=0;i<nv;i++){
        const key = pos[i*3]+','+pos[i*3+1]+','+pos[i*3+2];
        let g = groups.get(key); if (!g){ g=[]; groups.set(key,g); }
        g.push(i);
      }
      for (const idxs of groups.values()){
        if (idxs.length<2) continue;
        let sx=0,sy=0,sz=0;
        for (const i of idxs){ sx+=nrm[i*3]; sy+=nrm[i*3+1]; sz+=nrm[i*3+2]; }
        const l = Math.hypot(sx,sy,sz);
        if (l<1e-8) continue;
        sx/=l; sy/=l; sz/=l;
        for (const i of idxs){ nrm[i*3]=sx; nrm[i*3+1]=sy; nrm[i*3+2]=sz; }
      }
    }

    // node rest TRS + hierarchy
    const nodes=g.nodes.map(nd=>({
      t: nd.translation||[0,0,0],
      r: nd.rotation||[0,0,0,1],
      s: nd.scale||[1,1,1],
      children: nd.children||[]
    }));
    const parent=new Int32Array(nodes.length).fill(-1);
    nodes.forEach((nd,i)=>nd.children.forEach(ci=>parent[ci]=i));
    const order=[]; // topological (parents before children)
    const roots=[]; for (let i=0;i<nodes.length;i++) if (parent[i]<0) roots.push(i);
    (function walk(list){ for (const i of list){ order.push(i); walk(nodes[i].children); } })(roots);

    // animations -> channels grouped, with sampler keyframes
    const anims=(g.animations||[]).map(a=>{
      const samplers=a.samplers.map(s=>({
        input: readAccessor(g,bin,s.input).data,
        output: readAccessor(g,bin,s.output),
        interp: s.interpolation||'LINEAR'
      }));
      let dur=0;
      for (const s of samplers) dur=Math.max(dur, s.input[s.input.length-1]||0);
      const channels=a.channels.map(c=>({
        node:c.target.node, path:c.target.path, sampler:c.sampler
      }));
      return { name:a.name, samplers, channels, duration:dur||1 };
    });

    const model = {
      pos, nrm, jnt, wgt, idx, col, nj,
      nverts: pos.length/3, ntris: idx.length/3,
      ibm: ibmA, joints, nodes, parent, order,
      fit: new Float64Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
      anims, animNames: anims.map(a=>a.name)
    };
    computeFit(model);   // scene-fit derived from the actual skinned reference pose
    return model;
  }

  // Fit the model into the scene: skin a reference pose (Idle if present) with
  // an identity fit, measure its world AABB, then scale to ~1.9 tall, drop feet
  // to y=0 and center in x/z. Joint matrices carry the skeleton's own large
  // scale, so the fit must be measured after skinning — not from mesh positions.
  function computeFit(model) {
    let ai = model.anims.findIndex(a=>/Idle/i.test(a.name));
    if (ai<0) ai=0;
    const bones = sampleBones(model, ai, 0);   // model.fit is identity here
    let mn=[1e30,1e30,1e30], mx=[-1e30,-1e30,-1e30];
    for (let v=0; v<model.nverts; v++){
      const p0=model.pos[v*3],p1=model.pos[v*3+1],p2=model.pos[v*3+2];
      let o0=0,o1=0,o2=0;
      for (let k=0;k<4;k++){
        const w=model.wgt[v*4+k]; if(!w) continue;
        const B=bones.subarray(model.jnt[v*4+k]*12, model.jnt[v*4+k]*12+12);
        o0+=w*(B[0]*p0+B[1]*p1+B[2]*p2+B[3]);
        o1+=w*(B[4]*p0+B[5]*p1+B[6]*p2+B[7]);
        o2+=w*(B[8]*p0+B[9]*p1+B[10]*p2+B[11]);
      }
      if(o0<mn[0])mn[0]=o0; if(o0>mx[0])mx[0]=o0;
      if(o1<mn[1])mn[1]=o1; if(o1>mx[1])mx[1]=o1;
      if(o2<mn[2])mn[2]=o2; if(o2>mx[2])mx[2]=o2;
    }
    const sc=1.9/Math.max(mx[1]-mn[1],1e-4);
    const cx=(mn[0]+mx[0])*0.5, cz=(mn[2]+mx[2])*0.5;
    const fit=new Float64Array(16);
    fit[0]=sc; fit[5]=sc; fit[10]=sc; fit[15]=1;
    fit[12]=-sc*cx; fit[13]=-sc*mn[1]; fit[14]=-sc*cz;
    model.fit=fit;
  }

  function lerp(a,b,t){ return a+(b-a)*t; }
  function sampleVec(s, t, n) {
    const inp=s.input, out=s.output.data;
    const N=inp.length;
    if (t<=inp[0]) { const o=[]; for (let c=0;c<n;c++) o.push(out[c]); return o; }
    if (t>=inp[N-1]) { const o=[]; for (let c=0;c<n;c++) o.push(out[(N-1)*n+c]); return o; }
    let i=1; while (i<N && inp[i]<t) i++;
    const t0=inp[i-1], t1=inp[i];
    const f=s.interp==='STEP'?0:(t-t0)/(t1-t0||1);
    const o=[];
    if (n===4 && s.interp!=='STEP') {           // quaternion nlerp (shortest arc)
      let d=0; for (let c=0;c<4;c++) d+=out[(i-1)*4+c]*out[i*4+c];
      const sgn=d<0?-1:1;
      let len=0;
      for (let c=0;c<4;c++){ const v=lerp(out[(i-1)*4+c], sgn*out[i*4+c], f); o.push(v); len+=v*v; }
      len=Math.sqrt(len)||1; for (let c=0;c<4;c++) o[c]/=len;
    } else {
      for (let c=0;c<n;c++) o.push(lerp(out[(i-1)*n+c], out[i*n+c], f));
    }
    return o;
  }

  // Sample animation `ai` at time `t` -> Float32Array(nj*12) skin matrices,
  // each folded with the scene-fit transform, row-major 3x4 for the wasm side.
  function sampleBones(model, ai, t) {
    const anim = model.anims[ai];
    const tt = anim ? (t % anim.duration) : 0;
    const N = model.nodes.length;
    const T = model.nodes.map(n=>n.t.slice());
    const R = model.nodes.map(n=>n.r.slice());
    const S = model.nodes.map(n=>n.s.slice());
    if (anim) for (const ch of anim.channels) {
      const s=anim.samplers[ch.sampler];
      if (ch.path==='translation') T[ch.node]=sampleVec(s,tt,3);
      else if (ch.path==='rotation') R[ch.node]=sampleVec(s,tt,4);
      else if (ch.path==='scale') S[ch.node]=sampleVec(s,tt,3);
    }
    const world=new Array(N);
    for (const i of model.order) {
      const local=fromTRS(T[i],R[i],S[i]);
      world[i]= model.parent[i]<0 ? local : mul(world[model.parent[i]], local);
    }
    const out=new Float32Array(model.nj*12);
    for (let j=0;j<model.nj;j++){
      const node=model.joints[j];
      const ibm=model.ibm.subarray ? model.ibm.subarray(j*16,j*16+16) : model.ibm.slice(j*16,j*16+16);
      const jm=mul(world[node], ibm);          // joint matrix
      const F=mul(model.fit, jm);              // fold scene fit
      // row-major 3x4: b[r*4+c] = F[c*4+r]
      for (let r=0;r<3;r++) for (let c=0;c<4;c++) out[j*12+r*4+c]=F[c*4+r];
    }
    return out;
  }

  // Build a per-clip ground-contact travel curve. Instead of moving the
  // model at a constant average speed, sample low (ground-contact) vertices
  // through a full animation cycle and accumulate their backward sweep along
  // the chosen world axis. Applying this cumulative curve as rigid world
  // travel keeps planted feet aligned to the fixed checker ground.
  function buildGait(model, ai, axis = 2) {
    const anim = model.anims[ai];
    if (!anim) return null;
    const dur = anim.duration || 1, N = 48;
    const frames = [];
    let minY = 1e30;
    for (let f = 0; f < N; f++) {
      const vs = skinAll(model, ai, dur * f / N);
      frames.push(vs);
      for (let v = 0; v < model.nverts; v++) if (vs[v*3+1] < minY) minY = vs[v*3+1];
    }
    const thr = minY + 0.12;                 // "on the ground" band
    const coord = new Float64Array(N + 1);
    for (let f = 0; f <= N; f++) {
      const vs = frames[f % N];
      let sum = 0, cnt = 0;
      for (let v = 0; v < model.nverts; v++) {
        if (vs[v*3+1] < thr) { sum += vs[v*3+axis]; cnt++; }
      }
      coord[f] = cnt ? sum / cnt : (f ? coord[f-1] : 0);
    }
    const cum = new Float64Array(N + 1);
    for (let f = 1; f <= N; f++) {
      const d = coord[f] - coord[f-1];
      cum[f] = cum[f-1] + (d > 0 ? d : 0);
    }
    return { duration: dur, distance: cum[N], samples: cum, count: N };
  }

  function gaitOffset(gait, t) {
    if (!gait || gait.distance <= 0) return 0;
    const dur = gait.duration || 1;
    const cycles = Math.floor(t / dur);
    let u = (t - cycles * dur) / dur * gait.count;
    if (u < 0) u += gait.count;
    const i = Math.min(gait.count - 1, Math.floor(u));
    const f = u - i;
    const a = gait.samples[i], b = gait.samples[i + 1];
    return cycles * gait.distance + lerp(a, b, f);
  }

  function gaitSpeed(model, ai, axis = 2) {
    const gait = buildGait(model, ai, axis);
    return gait ? gait.distance / (gait.duration || 1) : 0;
  }

  // skin every vertex of a model at (ai,t) into a fresh Float32Array(nverts*3)
  function skinAll(model, ai, t) {
    const bones = sampleBones(model, ai, t);
    const out = new Float32Array(model.nverts * 3);
    for (let v = 0; v < model.nverts; v++) {
      const p0 = model.pos[v*3], p1 = model.pos[v*3+1], p2 = model.pos[v*3+2];
      let o0 = 0, o1 = 0, o2 = 0;
      for (let k = 0; k < 4; k++) {
        const w = model.wgt[v*4+k]; if (!w) continue;
        const B = bones.subarray(model.jnt[v*4+k]*12, model.jnt[v*4+k]*12+12);
        o0 += w*(B[0]*p0+B[1]*p1+B[2]*p2+B[3]);
        o1 += w*(B[4]*p0+B[5]*p1+B[6]*p2+B[7]);
        o2 += w*(B[8]*p0+B[9]*p1+B[10]*p2+B[11]);
      }
      out[v*3] = o0; out[v*3+1] = o1; out[v*3+2] = o2;
    }
    return out;
  }

  const API = { parseGLB, buildModel, sampleBones, buildGait, gaitOffset, gaitSpeed };
  if (typeof module!=='undefined' && module.exports) module.exports = API;
  else root.GLB = API;
})(typeof window!=='undefined' ? window : globalThis);
