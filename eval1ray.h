// eval1ray.h — performance-evaluation kernels comparing two SIMD strategies
// on an identical reduced pipeline (primary march, normals, AO, soft shadows,
// flat dino albedo, checker ground, spec, fog, sky):
//   renderLite4  — the current strategy: 4-ray packets, SoA lanes across rays
//   renderLite1  — one ray at a time, SIMD across the xyz components (AoS)
//   renderLite1p — one ray at a time, SIMD across 4 primitives (SoA over prims)
#ifndef EVAL1RAY_H
#define EVAL1RAY_H

void renderLite4(float az, float el, float dist, int w, int h, unsigned char *fb);
void renderLite1(float az, float el, float dist, int w, int h, unsigned char *fb);
void renderLite1p(float az, float el, float dist, int w, int h, unsigned char *fb);

#endif
