// mesh.h — triangle-mesh rendering path (Quaternius glTF dinosaurs).
// A second renderer that coexists with the SDF herd: JS uploads a skinned
// mesh (rest positions + joint indices/weights + per-triangle flat color)
// and per-frame bone matrices sampled from the glTF animation; this module
// skins the vertices, rebuilds a BVH, and scalar-raytraces the mesh onto the
// same checker floor / sky / fog as render.c so the two paths look coherent.
#ifndef MESH_H
#define MESH_H

#define MAXV 5200    // max vertices  (largest pack model: Stegosaurus 4590)
#define MAXT 2600    // max triangles (largest pack model: Stegosaurus 2282)
#define MAXJ 40      // max joints    (pack models: 29)

// upload buffers (JS writes into wasm memory through these pointers)
float* meshPos(void);     // rest positions        [MAXV*3]  (x,y,z)
int*   meshJoint(void);   // 4 joint indices/vert  [MAXV*4]
float* meshWeight(void);  // 4 skin weights/vert   [MAXV*4]
int*   meshIndex(void);   // triangle vertex ids   [MAXT*3]
float* meshColor(void);   // per-triangle rgb      [MAXT*3]  (linear)
float* meshBone(void);    // per-frame skin mats   [MAXJ*12] (3x4 row-major,
                          //   already folded with the model fit transform)

// set active counts before rendering (nj = joint count)
void meshSetCounts(int nv, int nt, int nj);

// skin -> build BVH -> raytrace the mesh into fb (RGBA8, w*h)
void renderMesh(float az, float el, float dist, int w, int h, unsigned char* fb);

#endif
