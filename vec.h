// vec.h — SIMD lane math shared by all modules (wasm32, no libc):
// v4 (f32x4) wrappers, SoA vectors V3/C3, scalar math (no libm), 2D value noise.
#ifndef VEC_H
#define VEC_H
#include <wasm_simd128.h>

typedef v128_t v4;
static inline v4 S(float x){ return wasm_f32x4_splat(x); }
static inline v4 vadd(v4 a, v4 b){ return wasm_f32x4_add(a,b); }
static inline v4 vsub(v4 a, v4 b){ return wasm_f32x4_sub(a,b); }
static inline v4 vmul(v4 a, v4 b){ return wasm_f32x4_mul(a,b); }
static inline v4 vdiv(v4 a, v4 b){ return wasm_f32x4_div(a,b); }
static inline v4 vmin(v4 a, v4 b){ return wasm_f32x4_min(a,b); }
static inline v4 vmax(v4 a, v4 b){ return wasm_f32x4_max(a,b); }
static inline v4 vabs(v4 a){ return wasm_f32x4_abs(a); }
static inline v4 vneg(v4 a){ return wasm_f32x4_neg(a); }
static inline v4 vsqrt(v4 a){ return wasm_f32x4_sqrt(a); }
static inline v4 vlt(v4 a, v4 b){ return wasm_f32x4_lt(a,b); }
static inline v4 vgt(v4 a, v4 b){ return wasm_f32x4_gt(a,b); }
static inline v4 vle(v4 a, v4 b){ return wasm_f32x4_le(a,b); }
static inline v4 vge(v4 a, v4 b){ return wasm_f32x4_ge(a,b); }
static inline v4 sel(v4 a, v4 b, v4 m){ return wasm_v128_bitselect(a,b,m); } // m ? a : b
static inline v4 vand(v4 a, v4 b){ return wasm_v128_and(a,b); }
static inline v4 vor (v4 a, v4 b){ return wasm_v128_or(a,b); }
static inline v4 vandn(v4 a, v4 b){ return wasm_v128_andnot(a,b); } // a & ~b
static inline v4 vnot(v4 a){ return wasm_v128_not(a); }
static inline int any(v4 m){ return wasm_v128_any_true(m); }
// Relaxed-SIMD fused multiply-add: 1 instruction on SSE(FMA)/NEON, higher precision.
static inline v4 vfma(v4 a, v4 b, v4 c){ return wasm_f32x4_relaxed_madd(a,b,c); }   // a*b + c
static inline v4 vfnma(v4 a, v4 b, v4 c){ return wasm_f32x4_relaxed_nmadd(a,b,c); }  // c - a*b
static inline v4 clamp01(v4 x){ return vmin(vmax(x, S(0.f)), S(1.f)); }
static inline v4 vclampf(v4 x, float a, float b){ return vmin(vmax(x, S(a)), S(b)); }
static inline v4 mixv(v4 a, v4 b, v4 t){ return vfma(vsub(b,a), t, a); }
static inline v4 vfloor(v4 x){ return wasm_f32x4_floor(x); }
static inline v4 vfract(v4 x){ return vsub(x, vfloor(x)); }

typedef struct { v4 x, y, z; } V3;
typedef struct { v4 r, g, b; } C3;
static inline V3 v3(v4 x, v4 y, v4 z){ V3 r = {x,y,z}; return r; }
static inline V3 v3add(V3 a, V3 b){ return v3(vadd(a.x,b.x), vadd(a.y,b.y), vadd(a.z,b.z)); }
static inline V3 v3scale(V3 a, v4 s){ return v3(vmul(a.x,s), vmul(a.y,s), vmul(a.z,s)); }
static inline v4 v3dot(V3 a, V3 b){ return vfma(a.z,b.z, vfma(a.y,b.y, vmul(a.x,b.x))); }
static inline V3 v3norm(V3 a){ v4 il = vdiv(S(1.f), vsqrt(v3dot(a,a))); return v3scale(a, il); }

// ---------- scalar math (no libm) ----------
static inline float fsqrt(float x){ return __builtin_sqrtf(x); }
static inline float ffabs(float x){ return __builtin_fabsf(x); }
static inline float fsin(float x){
    x = x - 6.2831853f * __builtin_rintf(x * 0.15915494f);
    float s = 1.2732395f * x - 0.4052847f * x * ffabs(x);
    return 0.225f * (s * ffabs(s) - s) + s;
}
static inline float fcos(float x){ return fsin(x + 1.5707963f); }
static inline float fclampf(float x, float a, float b){ return x<a?a:(x>b?b:x); }

static inline v4 vsin(v4 x){
    v4 k = wasm_f32x4_nearest(vmul(x, S(0.15915494f)));
    x = vsub(x, vmul(k, S(6.2831853f)));
    v4 s = vsub(vmul(x, S(1.2732395f)), vmul(vmul(x, vabs(x)), S(0.4052847f)));
    return vadd(vmul(S(0.225f), vsub(vmul(s, vabs(s)), s)), s);
}

// ---------- 2D value noise ----------
static inline v4 hash2(v4 ix, v4 iy){
    return vfract(vmul(vsin(vadd(vmul(ix,S(127.1f)), vmul(iy,S(311.7f)))), S(43758.547f)));
}
static inline v4 vnoise(v4 x, v4 y){
    v4 ix = vfloor(x), iy = vfloor(y);
    v4 fx = vsub(x,ix), fy = vsub(y,iy);
    v4 ux = vmul(vmul(fx,fx), vsub(S(3.f), vmul(S(2.f),fx)));
    v4 uy = vmul(vmul(fy,fy), vsub(S(3.f), vmul(S(2.f),fy)));
    v4 a = hash2(ix,iy), b = hash2(vadd(ix,S(1.f)),iy);
    v4 c = hash2(ix,vadd(iy,S(1.f))), d = hash2(vadd(ix,S(1.f)),vadd(iy,S(1.f)));
    return mixv(mixv(a,b,ux), mixv(c,d,ux), uy);
}

#endif
