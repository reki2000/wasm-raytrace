# DINO // RT HERD — wasm SIMD128 ray-marched dinosaur herd

A dependency-free, real-time SDF ray-marched dinosaur demo powered by
WebAssembly SIMD128. No GPU is used — the herd is rendered entirely on the
CPU via wasm. Also supports rendering the Quaternius triangle-mesh dinosaur
models (6 species).

- No emscripten (clang + wasm-ld directly), no libc
- Output is a single `dino-herd.html` file (wasm embedded as base64)

## Demo

🦖 <https://wasm-raytrace.pages.dev/>

## Usage

- Drag: orbit the camera / Wheel or pinch: zoom
- **MODEL** buttons: **SDF THEROPOD / STEGOSAURUS / TRICERATOPS** pick which
  SDF dinosaur the panel below edits (all three still render together as a
  herd); the six Quaternius names switch to the mesh dinosaurs and add an
  ACTION picker (Idle/Walk/Run/Attack/Jump/Death)
- **MATERIAL** panel: adjust the selected dinosaur's material (normal /
  textured / metallic reflective / acrylic translucent) and reflectivity,
  transmittance, refractive index, etc. in real time

## Requirements

- clang + wasm-ld (lld) with wasm32 target support. On Ubuntu: `apt install clang lld` (tested with clang 18)
- python3 — for base64 embedding

## Build

```sh
./build.sh
```

This produces `dino-herd.html`. It can be opened directly in a browser, but
the Quaternius model feature fetches `glb.js` and `models/*.glb` from the
same origin at runtime, so serve it locally with something like
`python3 -m http.server` (`file://` will fail to fetch).

## Architecture

See **[ARCHITECTURE.md](ARCHITECTURE.md)** (Japanese) for how the ray
tracing and animation are implemented.

## Deployment

`.github/workflows/cloudflare.yml` automatically builds and deploys to
Cloudflare Pages on every push to `main`.

## Credits / License

- **Code**: MIT License (see `LICENSE`)
- **Dinosaur models**: [Quaternius](https://quaternius.com/) — *Animated Dinosaur Pack*, CC0 1.0.
