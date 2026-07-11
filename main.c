// main.c — integration + wasm exports. Ties the modules together:
//   anim.c       animation engine (clock, gait, herd kinematics)
//   dino_model.c this demo's dinosaurs (geometry + animation parameters)
//   render.c     rendering engine (SDF store, pipeline, materials)
// Exports: fb() framebuffer pointer / render(t,az,el,dist,w,h) / mat(...)
#include "vec.h"
#include "render.h"
#include "anim.h"
#include "dino_model.h"

#define WMAX 960
#define HMAX 540
static unsigned char FB[WMAX * HMAX * 4];

__attribute__((export_name("fb")))
unsigned char* fbptr(void){ return FB; }

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
