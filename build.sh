#!/usr/bin/env bash
# 必要: clang + wasm-ld (lld), python3
set -euo pipefail
SRC="main.c render.c anim.c dino_model.c mesh.c scene.c eval1ray.c"
MT_MEM_BYTES=8388608   # fixed-size shared memory for the MT build (no growth needed); embed.py needs this to size the JS-side WebAssembly.Memory

# single-threaded build (safe fallback; runs anywhere, no SharedArrayBuffer needed)
clang --target=wasm32 -msimd128 -mrelaxed-simd -mbulk-memory -O3 -ffast-math -fno-math-errno \
  -nostdlib -Wl,--no-entry -Wl,-z,stack-size=65536 \
  -o dino.wasm $SRC
ls -l dino.wasm

# multithreaded build: shared+imported linear memory so worker instances can
# share it with the main thread; atomics for the row work-stealing counter.
# Fixed-size memory (no runtime growth needed — no malloc in this project).
clang --target=wasm32 -msimd128 -mrelaxed-simd -matomics -mbulk-memory -O3 -ffast-math -fno-math-errno \
  -DWASM_THREADS \
  -nostdlib -Wl,--no-entry -Wl,-z,stack-size=65536 \
  -Wl,--shared-memory -Wl,--import-memory \
  -Wl,--initial-memory=$MT_MEM_BYTES -Wl,--max-memory=$MT_MEM_BYTES \
  -Wl,--export=__stack_pointer \
  -o dino-mt.wasm $SRC
ls -l dino-mt.wasm

python3 embed.py $MT_MEM_BYTES
