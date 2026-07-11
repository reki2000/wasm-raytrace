// anim.c — animation engine (see anim.h). Deterministic and stateless:
// herd drift, walking phase and the clock all derive analytically from t.
#include "vec.h"
#include "anim.h"

const float VG = 0.85f;
float GT = 0.f;
float SCROLL = 0.f;

void animTick(float t){
    GT = t;
    SCROLL = VG * t;
}

Kin kin(float t, float x0, float D, float w, float ph, float a){
    float u = w*t + ph;
    float su = fsin(u), cu = fcos(u);
    float drift = D * fsin(u + a*su);
    float dspd  = D * w * fcos(u + a*su) * (1.f + a*cu);
    Kin k;
    k.x = x0 + drift;
    k.dist = VG*t + drift;            // integrated travel (gait phase source)
    k.speed = VG + dspd;
    k.run = fclampf((k.speed - 0.95f) * 3.5f, 0.f, 1.f);
    return k;
}

float gaitFoot(float ph, float ds, float A, float *lift){
    float f = ph * 0.15915494f;               // /2pi
    f = f - __builtin_floorf(f);
    if (f < ds){ *lift = 0.f; return A * (1.f - 2.f*f/ds); }
    float s = (f - ds) / (1.f - ds);
    *lift = fsin(3.1415927f * s);
    float e = s*s*(3.f - 2.f*s);              // smoothstep return swing
    return A * (2.f*e - 1.f);
}
