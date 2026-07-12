// mesh.h — triangle-mesh rendering path (Quaternius glTF dinosaurs).
// A second renderer that coexists with the SDF herd. JS uploads up to NMESH
// skinned models concatenated into one vertex/triangle store (with a per-vertex
// joint base and per-triangle model id) plus per-frame bone matrices sampled
// from the glTF animation; this module skins the vertices, rebuilds one BVH and
// scalar-raytraces the whole line-up onto the same checker floor / sky / fog as
// render.c. Per-model materials mirror the SDF path (plain/texture/metal/acrylic).
#ifndef MESH_H
#define MESH_H

#define MAXV 20000   // max total vertices  (pack sum ~19114)
#define MAXT 10000   // max total triangles (pack sum ~9532)
#define MAXJ 200     // max total joints    (6 x 29 = 174)
#define NMESH 6      // number of models in the line-up

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

// per-model surface material (same semantics as render.c setMaterial, plus
// a normal-welding toggle that applies to any mode)
// mode: 0 plain | 1 textured | 2 metallic | 3 acrylic
// smooth: 0 flat per-face normals | nonzero welded (Gouraud) vertex normals
void meshMat(int i, int mode, float refl, float tran, float ior, float tex, float gloss, int smooth);

// skin -> build BVH -> raytrace the line-up into fb (RGBA8, w*h)
void renderMesh(float az, float el, float dist, int w, int h, unsigned char* fb);

#endif
