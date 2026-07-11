const fs = require('fs');
(async () => {
  const { instance } = await WebAssembly.instantiate(fs.readFileSync('dino.wasm'));
  const e = instance.exports;
  const W = 480, H = 270;
  e.render(0, -0.85, 0.30, 5.6, W, H);
  const t0 = process.hrtime.bigint();
  const N = 60;
  for (let i = 0; i < N; i++) e.render(i / 60, -0.85 + i*0.01, 0.30, 5.6, W, H);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6 / N;
  console.log(`480x270: ${ms.toFixed(2)} ms/frame (${(1000/ms).toFixed(0)} fps)`);
  const fbp = e.fb();
  // 群れの離散/追走が見えるよう時刻を散らす
  for (const [name, t, az, el, d] of [
      ['f0', 0.5, -0.85, 0.30, 5.6],
      ['f1', 5.0, -0.85, 0.30, 5.6],
      ['f2', 9.5,  0.85, 0.26, 5.6],
      ['f3', 14.0, -2.40, 0.45, 6.2],
      ['f4', 0.5, -1.5708, 0.12, 5.2]]) {
    e.render(t, az, el, d, W, H);
    const m = new Uint8Array(e.memory.buffer, fbp, W*H*4);
    const rgb = Buffer.alloc(W*H*3);
    for (let i=0,j=0;i<W*H;i++){ rgb[j++]=m[i*4]; rgb[j++]=m[i*4+1]; rgb[j++]=m[i*4+2]; }
    fs.writeFileSync(name+'.ppm', Buffer.concat([Buffer.from(`P6\n${W} ${H}\n255\n`), rgb]));
  }
  // per-dino surface materials: mat(i, mode, refl, tran, ior, tex, gloss)
  e.mat(0, 2, 0.85, 0.65, 1.49, 0.0, 0.9);   // theropod: metallic
  e.mat(1, 3, 0.35, 0.85, 1.49, 0.0, 0.6);   // stego: acrylic
  e.mat(2, 0, 0.50, 0.65, 1.49, 0.0, 0.5);   // trike: plain
  e.render(9.5, 0.85, 0.26, 4.5, W, H);
  {
    const m = new Uint8Array(e.memory.buffer, fbp, W*H*4);
    const rgb = Buffer.alloc(W*H*3);
    for (let i=0,j=0;i<W*H;i++){ rgb[j++]=m[i*4]; rgb[j++]=m[i*4+1]; rgb[j++]=m[i*4+2]; }
    fs.writeFileSync('f5.ppm', Buffer.concat([Buffer.from(`P6\n${W} ${H}\n255\n`), rgb]));
  }
  console.log('frames dumped');
})().catch(e => { console.error(e); process.exit(1); });
