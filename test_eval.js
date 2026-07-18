// test_eval.js — SIMD strategy evaluation: 4-ray packet (SoA) vs 1-ray (AoS).
// Times renderLite4 / renderLite1 (identical reduced pipeline) plus the full
// renderer for context, across several cameras, and checks the two lite
// kernels produce (near-)identical images.
const fs = require('fs');
(async () => {
  const { instance } = await WebAssembly.instantiate(fs.readFileSync('dino.wasm'));
  const e = instance.exports;
  const W = 480, H = 270;
  const fbp = e.fb();

  // camera set: mid shot, wide shot, close-up (high packet divergence), low angle
  const CAMS = [
    ['mid',   0.5, -0.85,   0.30, 5.6],
    ['wide', 14.0, -2.40,   0.45, 6.2],
    ['close', 9.5,  0.85,   0.26, 3.2],
    ['low',   0.5, -1.5708, 0.12, 5.2],
  ];

  const grab = () => Buffer.from(new Uint8Array(e.memory.buffer, fbp, W*H*4));
  const dump = (name, buf) => {
    const rgb = Buffer.alloc(W*H*3);
    for (let i=0,j=0;i<W*H;i++){ rgb[j++]=buf[i*4]; rgb[j++]=buf[i*4+1]; rgb[j++]=buf[i*4+2]; }
    fs.writeFileSync(name+'.ppm', Buffer.concat([Buffer.from(`P6\n${W} ${H}\n255\n`), rgb]));
  };

  // correctness: same image from both kernels?
  console.log('--- image equivalence (lite4 vs lite1) ---');
  for (const [name, t, az, el, d] of CAMS) {
    e.renderLite4(t, az, el, d, W, H); const a = grab(); dump(`l4_${name}`, a);
    e.renderLite1(t, az, el, d, W, H); const b = grab(); dump(`l1_${name}`, b);
    e.renderLite1p(t, az, el, d, W, H); const c = grab(); dump(`lp_${name}`, c);
    const diff = (a, b) => {
      let sum=0, max=0, diffpx=0;
      for (let i=0;i<W*H*4;i++){
        if ((i&3)===3) continue;
        const df = Math.abs(a[i]-b[i]);
        sum += df; if (df>max) max=df; if (df>2) diffpx++;
      }
      return `meanAbsDiff=${(sum/(W*H*3)).toFixed(4)} max=${max} chans>2: ${(100*diffpx/(W*H*3)).toFixed(3)}%`;
    };
    console.log(`${name}: lite1  vs lite4: ${diff(a,b)}`);
    console.log(`${name}: lite1p vs lite4: ${diff(a,c)}`);
  }

  // timing: interleaved trials, animated camera sweep per camera base
  const time = (fn, t0, az, el, d, frames) => {
    const s = process.hrtime.bigint();
    for (let i=0;i<frames;i++) fn(t0 + i/60, az + i*0.01, el, d, W, H);
    return Number(process.hrtime.bigint() - s) / 1e6 / frames;
  };

  const KERNELS = [
    ['full (4ray)', e.render.bind(e)],
    ['lite4      ', e.renderLite4.bind(e)],
    ['lite1      ', e.renderLite1.bind(e)],
    ['lite1p     ', e.renderLite1p.bind(e)],
  ];
  const FRAMES = 40, TRIALS = 3;

  // warmup (tier-up)
  for (const [,fn] of KERNELS) for (let i=0;i<10;i++) fn(i/60, -0.85, 0.30, 5.6, W, H);

  console.log(`--- timing ${W}x${H}, ${FRAMES} frames x ${TRIALS} trials (best) ---`);
  const results = {};
  for (const [cname, t, az, el, d] of CAMS) {
    results[cname] = {};
    for (const [kname, fn] of KERNELS) {
      let best = Infinity;
      for (let r=0;r<TRIALS;r++) best = Math.min(best, time(fn, t, az, el, d, FRAMES));
      results[cname][kname.trim()] = best;
      console.log(`${cname.padEnd(6)} ${kname}: ${best.toFixed(2)} ms/frame (${(1000/best).toFixed(1)} fps)`);
    }
    const r4 = results[cname]['lite4'], r1 = results[cname]['lite1'], rp = results[cname]['lite1p'];
    console.log(`${cname.padEnd(6)} lite1/lite4: ${(r1/r4).toFixed(2)}x  lite1p/lite4: ${(rp/r4).toFixed(2)}x`);
  }

  // geometric mean over cameras
  const gm = k => Math.exp(CAMS.reduce((s,[c]) => s + Math.log(results[c][k]), 0) / CAMS.length);
  console.log('--- geometric mean over cameras ---');
  for (const k of ['full (4ray)','lite4','lite1','lite1p'])
    console.log(`${k}: ${gm(k).toFixed(2)} ms/frame`);
  console.log(`overall lite1/lite4: ${(gm('lite1')/gm('lite4')).toFixed(2)}x  lite1p/lite4: ${(gm('lite1p')/gm('lite4')).toFixed(2)}x`);
})().catch(e => { console.error(e); process.exit(1); });
