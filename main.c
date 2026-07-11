// main.c — integration + wasm exports. Ties the modules together:
//   anim.c       animation engine (clock, gait, herd kinematics)
//   dino_model.c this demo's dinosaurs (geometry + animation parameters)
//   render.c     rendering engine (SDF store, pipeline, materials)
// Exports: fb() framebuffer pointer / render(t,az,el,dist,w,h) / mat(...)
#include "vec.h"
#include "render.h"
#include "anim.h"
#include "dino_model.h"
#include "mesh.h"

#define WMAX 1280
#define HMAX 720
static unsigned char FB[WMAX * HMAX * 4];

__attribute__((export_name("fb")))
unsigned char* fbptr(void){ return FB; }

// ---- mesh path (Quaternius glTF dinosaurs) ----
__attribute__((export_name("meshPos")))    float* e_meshPos(void){ return meshPos(); }
__attribute__((export_name("meshJoint")))  int*   e_meshJoint(void){ return meshJoint(); }
__attribute__((export_name("meshWeight"))) float* e_meshWeight(void){ return meshWeight(); }
__attribute__((export_name("meshIndex")))  int*   e_meshIndex(void){ return meshIndex(); }
__attribute__((export_name("meshColor")))  float* e_meshColor(void){ return meshColor(); }
__attribute__((export_name("meshTriModel"))) int* e_meshTriModel(void){ return meshTriModel(); }
__attribute__((export_name("meshBone")))   float* e_meshBone(void){ return meshBone(); }
__attribute__((export_name("meshSetCounts")))
void e_meshSetCounts(int nv, int nt, int nj){ meshSetCounts(nv, nt, nj); }
__attribute__((export_name("meshSetFocus")))
void e_meshSetFocus(float x, float z){ meshSetFocus(x, z); }
__attribute__((export_name("meshMat")))
void e_meshMat(int i, int mode, float refl, float tran, float ior, float tex, float gloss){
    meshMat(i, mode, refl, tran, ior, tex, gloss);
}

__attribute__((export_name("renderMesh")))
void e_renderMesh(float az, float el, float dist, int w, int h){
    renderMesh(az, el, dist, w, h, FB);
}

__attribute__((export_name("mat")))
void mat(int i, int mode, float refl, float tran, float ior, float tex, float gloss){
    setMaterial(i, mode, refl, tran, ior, tex, gloss);
}

__attribute__((export_name("render")))
void render(float t, float az, float el, float dist, int w, int h){
    animTick(t);      // advance the animation clock (GT, ground scroll)
    animate(t);       // rebuild the herd's primitives for this instant
    renderFrame(az, el, dist, w, h, FB);
}
