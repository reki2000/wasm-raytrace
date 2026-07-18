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
#include "scene.h"

#define WMAX 1280
#define HMAX 720
static unsigned char FB[WMAX * HMAX * 4];

__attribute__((export_name("fb")))
unsigned char* fbptr(void){ return FB; }

// ---- mesh path (Quaternius glTF dinosaurs) ----
__attribute__((export_name("meshPos")))    float* e_meshPos(void){ return meshPos(); }
__attribute__((export_name("meshNormal"))) float* e_meshNormal(void){ return meshNormal(); }
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
void e_meshMat(int i, int mode, float refl, float tran, float ior, float tex, float gloss, int smooth){
    meshMat(i, mode, refl, tran, ior, tex, gloss, smooth);
}

__attribute__((export_name("renderMesh")))
void e_renderMesh(float az, float el, float dist, int w, int h){
    renderMesh(az, el, dist, w, h, FB);
}

__attribute__((export_name("meshPrep")))
void e_meshPrep(void){ meshPrep(); }

__attribute__((export_name("renderMeshRows")))
void e_renderMeshRows(float az, float el, float dist, int w, int h, int y0, int y1){
    renderMeshRows(az, el, dist, w, h, FB, y0, y1);
}

__attribute__((export_name("mat")))
void mat(int i, int mode, float refl, float tran, float ior, float tex, float gloss){
    setMaterial(i, mode, refl, tran, ior, tex, gloss);
}

__attribute__((export_name("dinoSetActive")))
void e_dinoSetActive(int active){
    dinoSetActive(active);
}

__attribute__((export_name("groundScrollSetActive")))
void e_groundScrollSetActive(int active){
    groundScrollSetActive(active);
}

__attribute__((export_name("render")))
void render(float t, float az, float el, float dist, int w, int h){
    animTick(t);      // advance the animation clock (GT, ground scroll)
    animate(t);       // rebuild the herd's primitives for this instant
    renderFrame(az, el, dist, w, h, FB);
}

// ---- split prep / row-render (multithreaded pipeline) ----
// Prep rebuilds all per-frame scene state (single-threaded, on the main
// instance only). Once done, renderRows may be called for any row range
// from any thread instance sharing this module's memory: rows are read-only
// over the scene and write disjoint fb spans, so no synchronization is
// needed beyond "prep happened-before rows".
__attribute__((export_name("renderPrep")))
void e_renderPrep(float t){
    animTick(t);
    animate(t);
}

__attribute__((export_name("renderRows")))
void e_renderRows(float az, float el, float dist, int w, int h, int y0, int y1){
    renderRows(az, el, dist, w, h, FB, y0, y1);
}

// ---- combined scene (SDF herd + mesh line-up together, sharing lighting/
// shadows/floor-mirror/materials — see scene.h) ----
__attribute__((export_name("renderScene")))
void e_renderScene(float t, float az, float el, float dist, int w, int h){
    renderScene(t, az, el, dist, w, h, FB);
}

__attribute__((export_name("scenePrep")))
void e_scenePrep(float t){
    scenePrep(t);
}

__attribute__((export_name("sceneRows")))
void e_sceneRows(float az, float el, float dist, int w, int h, int y0, int y1){
    sceneRows(az, el, dist, w, h, FB, y0, y1);
}

#ifdef WASM_THREADS
#include <stdatomic.h>

// Private stack for each worker instance (the main instance keeps the
// linker-assigned default stack). JS sets __stack_pointer per instantiation
// to a disjoint slice of this array before calling any other export.
#define MAXTHREADS 8
#define TSTACK_SZ 65536
static unsigned char TSTACK[MAXTHREADS][TSTACK_SZ] __attribute__((aligned(16)));

__attribute__((export_name("threadStackBase")))
unsigned char* e_threadStackBase(void){ return &TSTACK[0][0]; }

static _Atomic int g_nextRow;
static int g_frameH;
#define ROW_CHUNK 4   // rows per work-stealing grab; dinos cluster mid-frame so static bands would be unbalanced

__attribute__((export_name("frameBegin")))
void e_frameBegin(int h){
    g_frameH = h;
    atomic_store(&g_nextRow, 0);
}

// Called by every participating thread (workers + main) after frameBegin.
// Grabs row chunks until the frame is exhausted, then returns.
__attribute__((export_name("renderRowsSteal")))
void e_renderRowsSteal(float az, float el, float dist, int w, int h){
    for (;;){
        int y0 = atomic_fetch_add(&g_nextRow, ROW_CHUNK);
        if (y0 >= g_frameH) break;
        int y1 = y0 + ROW_CHUNK;
        if (y1 > g_frameH) y1 = g_frameH;
        renderRows(az, el, dist, w, h, FB, y0, y1);
    }
}

// Mesh-path counterpart of renderRowsSteal; shares the same g_nextRow/
// g_frameH counter. Kept for callers still using the standalone mesh-only
// path (test_mt_mesh.js); the combined path below (sceneRowsSteal) is what
// template.html drives now, since it renders both scenes together.
__attribute__((export_name("renderMeshRowsSteal")))
void e_renderMeshRowsSteal(float az, float el, float dist, int w, int h){
    for (;;){
        int y0 = atomic_fetch_add(&g_nextRow, ROW_CHUNK);
        if (y0 >= g_frameH) break;
        int y1 = y0 + ROW_CHUNK;
        if (y1 > g_frameH) y1 = g_frameH;
        renderMeshRows(az, el, dist, w, h, FB, y0, y1);
    }
}

// Combined-scene counterpart; shares the same g_nextRow/g_frameH counter.
__attribute__((export_name("sceneRowsSteal")))
void e_sceneRowsSteal(float az, float el, float dist, int w, int h){
    for (;;){
        int y0 = atomic_fetch_add(&g_nextRow, ROW_CHUNK);
        if (y0 >= g_frameH) break;
        int y1 = y0 + ROW_CHUNK;
        if (y1 > g_frameH) y1 = g_frameH;
        sceneRows(az, el, dist, w, h, FB, y0, y1);
    }
}
#endif
