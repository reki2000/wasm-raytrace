// mesh.h — triangle-mesh rendering path (Quaternius glTF dinosaurs).
// A second renderer that coexists with the SDF herd. JS uploads up to NMESH
// skinned models concatenated into one vertex/triangle store (with a per-vertex
// joint base and per-triangle model id) plus per-frame bone matrices sampled
// from the glTF animation; this module skins the vertices, rebuilds one BVH and
// scalar-raytraces the whole line-up onto the same checker floor / sky / fog as
// render.c. Per-model materials mirror the SDF path (plain/texture/metal/acrylic).
#ifndef MESH_H
#define MESH_H
#include "vec.h"

#define MAXV 20000   // max total vertices  (pack sum ~19114)
#define MAXT 10000   // max total triangles (pack sum ~9532)
#define MAXJ 200     // max total joints    (6 x 29 = 174)
#define NMESH 6      // number of models in the line-up
#ifndef CAM_TARGET_Y
// Orbit-camera target height. Aim near dinosaur heads so tall frills/horns stay in frame.
#define CAM_TARGET_Y 1.15f
#endif

// upload buffers (JS writes into wasm memory through these pointers)
float* meshPos(void);      // rest positions        [MAXV*3]
float* meshNormal(void);   // rest normals (GLB NORMAL attribute) [MAXV*3]
int*   meshJoint(void);    // 4 joint indices/vert  [MAXV*4]  (global, joint-base added)
float* meshWeight(void);   // 4 skin weights/vert   [MAXV*4]
int*   meshIndex(void);    // triangle vertex ids   [MAXT*3]  (global, vert-base added)
float* meshColor(void);    // per-triangle rgb      [MAXT*3]  (linear baseColor)
int*   meshTriModel(void); // per-triangle model id [MAXT]    (0..NMESH-1)
float* meshBone(void);     // per-frame skin mats   [MAXJ*12] (3x4 row-major, fit-folded)

void meshSetCounts(int nv, int nt, int nj);
void meshSetFocus(float x, float z);   // camera orbit target (selected dino)
extern float FOCX, FOCZ;               // current camera orbit target (shared with scene.c)

// per-model surface material (same semantics as render.c setMaterial, plus
// a normal-welding toggle that applies to any mode)
// mode: 0 plain | 1 textured | 2 metallic | 3 acrylic
// smooth: 0 flat per-face normals | nonzero welded (Gouraud) vertex normals
void meshMat(int i, int mode, float refl, float tran, float ior, float tex, float gloss, int smooth);

// skin -> build BVH -> raytrace the line-up into fb (RGBA8, w*h)
void renderMesh(float az, float el, float dist, int w, int h, unsigned char* fb);

// split prep (skin + BVH + triangle pack, single-threaded) / row-render
// (read-only over that state; rows are independent, safe to call concurrently
// with different [y0,y1) ranges from any thread sharing this module's memory)
void meshPrep(void);
void renderMeshRows(float az, float el, float dist, int w, int h, unsigned char* fb, int y0, int y1);

// ---------- combined-scene API (used by scene.c to unify with the SDF path) ----------
// 4-ray-packet BVH trace against the mesh line-up, bounded by tmax per lane
// (e.g. the SDF path's own nearest-hit distance, so the combined renderer
// only accepts a mesh hit when it's strictly nearer). ro is per-lane (SoA)
// so this covers both a shared camera origin (splat the same point into all
// 4 lanes, as the primary ray packet does) and genuinely per-pixel origins
// (floor-mirror / acrylic-retrace rays, which start at each pixel's own hit
// point). hitId4 gets 4 triangle ids (-1 = no hit closer than tmax) written
// via wasm_v128_store semantics (4 contiguous ints).
void meshTraceP(V3 ro, V3 rd, v4 tmax, v4 *tHitOut, int *hitId4);
// scalar single-ray trace / shadow occlusion test, for secondary rays
// (floor mirror, acrylic retrace, sun visibility) driven per-lane.
int  meshTraceScalar(const float *ro, const float *rd, float tmax, float *tHit);
int  meshOccluded(const float *ro, const float *rd, float tmax);
// Surface lookup for a mesh hit at world point P along incoming ray rd:
// outward normal (flipped to face against rd), vertex-color albedo, that
// triangle's model material, and whether rd is entering the body (for
// acrylic Snell refraction — same convention as render.c's SDF acrylic).
void meshSurface(int tri, const float *P, const float *rd,
                  float *N, float *albedo, int *entering,
                  int *mode, float *refl, float *tran, float *ior, float *gloss);

#endif
