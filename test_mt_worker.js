// worker side of test_mt*.js: instantiates dino-mt.wasm against the shared
// memory handed down by the parent, gives itself a private stack slice, then
// steals row chunks whenever asked. `fn` selects which *RowsSteal export to
// call (renderRowsSteal for the SDF path, renderMeshRowsSteal for the mesh
// path) so this one worker script covers both renderers.
const { workerData, parentPort } = require('worker_threads');
const fs = require('fs');

(async () => {
  const { wasmPath, mem, id } = workerData;
  const bytes = fs.readFileSync(wasmPath);
  const { instance } = await WebAssembly.instantiate(bytes, { env: { memory: mem } });
  const e = instance.exports;
  const STACK_SZ = 65536;
  const base = e.threadStackBase();
  e.__stack_pointer.value = base + (id + 1) * STACK_SZ; // disjoint from main's default stack and other workers
  parentPort.postMessage({ type: 'ready' });
  parentPort.on('message', (msg) => {
    if (msg.type === 'go') {
      e[msg.fn](...msg.args);
      parentPort.postMessage({ type: 'rowsDone', id });
    } else if (msg.type === 'quit') {
      process.exit(0);
    }
  });
})();
