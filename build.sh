#!/usr/bin/env bash
# 必要: clang + wasm-ld (lld), python3
set -euo pipefail
clang --target=wasm32 -msimd128 -mbulk-memory -O3 -ffast-math -fno-math-errno \
  -nostdlib -Wl,--no-entry -Wl,-z,stack-size=65536 \
  -o dino.wasm main.c render.c anim.c dino_model.c
ls -l dino.wasm
python3 embed.py
