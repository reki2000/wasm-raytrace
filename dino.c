// dino.c — SIMD128 ray-marched dinosaur herd (wasm32, no libc)
// 3 species (theropod / stegosaurus / triceratops), reflective floor,
// fresnel env reflections, procedural textures, deterministic herd behavior.
#include <wasm_simd128.h>

#define WMAX 960
#define HMAX 540
static unsigned char FB[WMAX * HMAX * 4];

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
static inline v4 sel(v4 a, v4 b, v4 m){ return wasm_v128_bitselect(a,b,m); } // m ? a : b
static inline v4 vand(v4 a, v4 b){ return wasm_v128_and(a,b); }
static inline v4 vor (v4 a, v4 b){ return wasm_v128_or(a,b); }
static inline v4 vandn(v4 a, v4 b){ return wasm_v128_andnot(a,b); } // a & ~b
static inline v4 vnot(v4 a){ return wasm_v128_not(a); }
static inline int any(v4 m){ return wasm_v128_any_true(m); }
static inline v4 clamp01(v4 x){ return vmin(vmax(x, S(0.f)), S(1.f)); }
static inline v4 vclampf(v4 x, float a, float b){ return vmin(vmax(x, S(a)), S(b)); }
static inline v4 mixv(v4 a, v4 b, v4 t){ return vadd(a, vmul(vsub(b,a), t)); }
static inline v4 vfloor(v4 x){ return wasm_f32x4_floor(x); }
static inline v4 vfract(v4 x){ return vsub(x, vfloor(x)); }

typedef struct { v4 x, y, z; } V3;
typedef struct { v4 r, g, b; } C3;
static inline V3 v3(v4 x, v4 y, v4 z){ V3 r = {x,y,z}; return r; }
static inline V3 v3add(V3 a, V3 b){ return v3(vadd(a.x,b.x), vadd(a.y,b.y), vadd(a.z,b.z)); }
static inline V3 v3scale(V3 a, v4 s){ return v3(vmul(a.x,s), vmul(a.y,s), vmul(a.z,s)); }
static inline v4 v3dot(V3 a, V3 b){ return vadd(vadd(vmul(a.x,b.x), vmul(a.y,b.y)), vmul(a.z,b.z)); }
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

// piecewise gait: stance (phase<ds) moves the foot linearly in traveled distance
// -> exact rolling contact when cadence K = pi*ds/A (A: half-stride, body space).
// Returns foot x-offset in [-A,A]; sets lift 0..1 (0 during stance).
static float gaitFoot(float ph, float ds, float A, float *lift){
    float f = ph * 0.15915494f;               // /2pi
    f = f - __builtin_floorf(f);
    if (f < ds){ *lift = 0.f; return A * (1.f - 2.f*f/ds); }
    float s = (f - ds) / (1.f - ds);
    *lift = fsin(3.1415927f * s);
    float e = s*s*(3.f - 2.f*s);              // smoothstep return swing
    return A * (2.f*e - 1.f);
}

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

// ---------- primitives: anisotropic tapered capsules ----------
// q: 0..2 a | 3..5 b-a | 6 1/l2 | 7 r1 | 8 r2-r1 | 9..11 cull center | 12 SK+boundR | 13..15 Ax,Ay,Az
// Axis-aligned anisotropy A<=1 widens that axis by 1/sqrt(A); Lipschitz stays <=1 (safe to march).
// Constraint: the segment axis must lie within A=1 axes.
#define MAXP 96
#define SK 0.062f
static float PR[MAXP][16];
static int NP = 0;
static float OX = 0.f, OZ = 0.f;   // current dino origin offset

static void coneA(float ax,float ay,float az, float bx,float by,float bz,
                  float r1,float r2, float Ax,float Ay,float Az){
    float *q = PR[NP++];
    ax+=OX; bx+=OX; az+=OZ; bz+=OZ;
    q[0]=ax; q[1]=ay; q[2]=az;
    float bax=bx-ax, bay=by-ay, baz=bz-az;
    q[3]=bax; q[4]=bay; q[5]=baz;
    float l2 = bax*bax + bay*bay + baz*baz;
    q[6]=1.f/l2;
    q[7]=r1; q[8]=r2-r1;
    q[9]=ax+bax*0.5f; q[10]=ay+bay*0.5f; q[11]=az+baz*0.5f;
    float Am = Ax<Ay ? (Ax<Az?Ax:Az) : (Ay<Az?Ay:Az);
    float smax = 1.f/fsqrt(Am);
    float rmax = (r1 > r2 ? r1 : r2) * smax;
    q[12]=SK + fsqrt(l2)*0.5f + rmax;
    q[13]=Ax; q[14]=Ay; q[15]=Az;
}
static void cone(float ax,float ay,float az, float bx,float by,float bz, float r1,float r2){
    coneA(ax,ay,az,bx,by,bz,r1,r2,1.f,1.f,1.f);
}

// eyes: 2 per species; hard-min spheres (not smin-blended) so they stay crisp
#define NEY 6
static float EYP[NEY][3];
static float EYRAD[NEY];
static float EB[3][4];             // per-species eye-pair bound cx,cy,cz,r

static inline v4 smin(v4 a, v4 b){
    v4 h = clamp01(vadd(S(0.5f), vmul(vsub(b,a), S(0.5f/SK))));
    return vsub(mixv(b, a, h), vmul(S(SK), vmul(h, vsub(S(1.f), h))));
}

static inline v4 eyePair(V3 p, int k){
    v4 d = S(100.f);
    for (int i=k*2;i<k*2+2;i++){
        v4 ex=vsub(p.x,S(EYP[i][0])), ey=vsub(p.y,S(EYP[i][1])), ez=vsub(p.z,S(EYP[i][2]));
        d = vmin(d, vsub(vsqrt(vadd(vadd(vmul(ex,ex),vmul(ey,ey)),vmul(ez,ez))), S(EYRAD[i])));
    }
    return d;
}
static inline v4 eyeDistAll(V3 p){
    v4 d = eyePair(p,0);
    d = vmin(d, eyePair(p,1));
    return vmin(d, eyePair(p,2));
}

// prims whose bound comes within margin of ray segment [tA,tB] (any lane), range [from,to)
static inline int buildList(V3 ro, V3 rd, v4 tA, v4 tB, float margin,
                            int from, int to, unsigned char *L, int n){
    for (int i=from;i<to;i++){
        const float *q = PR[i];
        v4 ocx=vsub(ro.x,S(q[9])), ocy=vsub(ro.y,S(q[10])), ocz=vsub(ro.z,S(q[11]));
        v4 tc = vneg(vadd(vadd(vmul(ocx,rd.x),vmul(ocy,rd.y)),vmul(ocz,rd.z)));
        tc = vmin(vmax(tc, tA), tB);
        v4 px = vadd(ocx, vmul(rd.x,tc));
        v4 py = vadd(ocy, vmul(rd.y,tc));
        v4 pz = vadd(ocz, vmul(rd.z,tc));
        v4 d2 = vadd(vadd(vmul(px,px),vmul(py,py)),vmul(pz,pz));
        v4 th = S(q[12] + margin);
        if (any(vlt(d2, vmul(th,th)))) L[n++] = (unsigned char)i;
    }
    return n;
}

static inline v4 mapL(V3 p, const unsigned char *L, int n){
    v4 d = S(100.f);
    for (int k=0;k<n;k++){
        const float *q = PR[L[k]];
        v4 cx = vsub(p.x,S(q[9])), cy = vsub(p.y,S(q[10])), cz = vsub(p.z,S(q[11]));
        v4 d2 = vadd(vadd(vmul(cx,cx),vmul(cy,cy)),vmul(cz,cz));
        v4 rh = vadd(d, S(q[12]));
        if (!any(vlt(d2, vmul(rh,rh)))) continue;
        v4 pax=vsub(p.x,S(q[0])), pay=vsub(p.y,S(q[1])), paz=vsub(p.z,S(q[2]));
        v4 h = clamp01(vmul(vadd(vadd(vmul(pax,S(q[3])),vmul(pay,S(q[4]))),vmul(paz,S(q[5]))), S(q[6])));
        v4 qx=vsub(pax,vmul(S(q[3]),h)), qy=vsub(pay,vmul(S(q[4]),h)), qz=vsub(paz,vmul(S(q[5]),h));
        v4 t = vadd(vadd(vmul(vmul(qx,qx),S(q[13])), vmul(vmul(qy,qy),S(q[14]))), vmul(vmul(qz,qz),S(q[15])));
        v4 dd = vsub(vsqrt(t), vadd(S(q[7]), vmul(S(q[8]),h)));
        d = smin(d, dd);
    }
    return d;
}

static inline v4 mapLE(V3 p, const unsigned char *L, int n){
    v4 d = mapL(p, L, n);
    for (int k=0;k<3;k++){
        v4 ex=vsub(p.x,S(EB[k][0])), ey=vsub(p.y,S(EB[k][1])), ez=vsub(p.z,S(EB[k][2]));
        v4 e2 = vadd(vadd(vmul(ex,ex),vmul(ey,ey)),vmul(ez,ez));
        v4 re = vadd(d, S(EB[k][3]));
        if (any(vlt(e2, vmul(re,re)))) d = vmin(d, eyePair(p,k));
    }
    return d;
}

// ---------- herd / dinos ----------
#define ND 3
static float DB[ND][4];     // bound sphere cx,cy,cz,r
static int   DPR[ND][2];    // prim range [start,end)
static float DXW[ND], DZW[ND]; // world offsets (texture space / id)
static float SCROLL = 0.f;
static float SUNX, SUNY, SUNZ;
static float GT = 0.f;

static const float VG = 0.85f;  // ground / herd reference speed

// herd member kinematics: x(t) = x0 + D*sin(u + a*sin u), u = w t + ph (skewed sine)
// speed(t) = VG + D*w*cos(u + a*sin u)*(1 + a*cos u)
typedef struct { float x, dist, speed, run; } Kin;
static Kin kin(float t, float x0, float D, float w, float ph, float a){
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

// ---- theropod (biped, IK legs) ----
static void tleg(float side, float ph, float hy, float run){
    float hx = -0.22f, hz = 0.17f * side;
    float lift;
    float ax = hx + 0.05f + gaitFoot(ph, 0.56f, 0.315f, &lift);
    float ay = 0.105f + (0.17f + 0.10f*run) * lift;
    float L1 = 0.42f, L2 = 0.40f;
    float dx = ax - hx, dy = ay - hy;
    float d = fsqrt(dx*dx + dy*dy);
    float dmax = L1 + L2 - 0.02f;
    if (d > dmax){ float s = dmax / d; dx*=s; dy*=s; ax=hx+dx; ay=hy+dy; d=dmax; }
    float a = (d*d + L1*L1 - L2*L2) / (2.f*d);
    float h2 = L1*L1 - a*a; float h = h2 > 0.f ? fsqrt(h2) : 0.f;
    float ux = dx/d, uy = dy/d;
    float px = uy, py = -ux;
    if (px > 0.f){ px = -px; py = -py; }
    float kx = hx + ux*a + px*h, ky = hy + uy*a + py*h;
    cone(hx, hy, hz, kx, ky, hz + 0.02f*side, 0.165f, 0.095f);
    cone(kx, ky, hz + 0.02f*side, ax, ay, hz + 0.03f*side, 0.085f, 0.05f);
    float ty = ay - 0.062f; if (ty < 0.045f) ty = 0.045f;
    float fz = hz + 0.03f*side;
    cone(ax + 0.01f, ay, fz, ax + 0.20f, ty, fz,          0.040f, 0.011f);   // middle toe
    cone(ax + 0.01f, ay, fz, ax + 0.155f, ty + 0.004f, fz + 0.056f, 0.034f, 0.009f);
    cone(ax + 0.01f, ay, fz, ax + 0.155f, ty + 0.004f, fz - 0.056f, 0.034f, 0.009f);
}

static void theropod(const Kin *K){
    float ph = K->dist * 5.585f;   // = pi*ds/A -> no foot slip
    float run = K->run;
    float bob = (0.035f + 0.030f*run) * fsin(ph * 2.f);
    float hy = 0.88f + bob;
    float cy = 1.04f + bob * 0.7f;
    float hb = 0.025f * fsin(ph * 2.f + 1.3f);
    float ln = 0.10f * run;   // forward lean when running

    cone(-0.28f, hy, 0.f, 0.30f, cy, 0.f, 0.305f, 0.27f);
    cone(0.34f+ln, cy + 0.06f, 0.f, 0.60f+ln*2.f, 1.44f + hb - ln, 0.f, 0.16f, 0.10f);
    cone(0.60f+ln*2.f, 1.47f + hb - ln, 0.f, 0.92f+ln*2.f, 1.40f + hb - ln, 0.f, 0.125f, 0.055f);
    float gape = fsin(GT*0.9f + 1.7f);
    gape = gape > 0.55f ? (gape - 0.55f)*(1.f/0.45f) : 0.f; gape *= gape;
    cone(0.62f+ln*2.f, 1.372f + hb - ln, 0.f,
         0.89f+ln*2.f, 1.316f + hb - ln - 0.085f*gape, 0.f, 0.050f, 0.020f);

    float z1 = 0.05f*fsin(ph-0.9f), z2 = 0.10f*fsin(ph-1.8f), z3 = 0.16f*fsin(ph-2.7f);
    cone(-0.30f, hy+0.03f, 0.f, -0.78f, 0.86f+bob*0.6f+ln*0.4f, z1, 0.21f, 0.125f);
    cone(-0.78f, 0.86f+bob*0.6f+ln*0.4f, z1, -1.18f, 0.78f+ln*0.5f, z2, 0.125f, 0.065f);
    cone(-1.18f, 0.78f+ln*0.5f, z2, -1.52f, 0.72f+ln*0.6f, z3, 0.065f, 0.02f);

    float aw = 0.04f * fsin(ph * 2.f);
    cone(0.30f, cy, 0.19f, 0.43f+aw, 0.85f, 0.235f, 0.055f, 0.032f);
    cone(0.30f, cy, -0.19f, 0.43f-aw, 0.85f, -0.235f, 0.055f, 0.032f);
    // two claw fingers per hand
    cone(0.43f+aw, 0.85f, 0.235f, 0.505f+aw, 0.780f, 0.253f, 0.014f, 0.004f);
    cone(0.43f+aw, 0.85f, 0.235f, 0.495f+aw, 0.776f, 0.215f, 0.014f, 0.004f);
    cone(0.43f-aw, 0.85f, -0.235f, 0.505f-aw, 0.780f, -0.253f, 0.014f, 0.004f);
    cone(0.43f-aw, 0.85f, -0.235f, 0.495f-aw, 0.776f, -0.215f, 0.014f, 0.004f);

    tleg( 1.f, ph, hy - 0.10f, run);
    tleg(-1.f, ph + 3.1415927f, hy - 0.10f, run);

    EYP[0][0]=0.665f+ln*2.f+OX; EYP[0][1]=1.49f+hb-ln; EYP[0][2]= 0.10f+OZ;
    EYP[1][0]=0.665f+ln*2.f+OX; EYP[1][1]=1.49f+hb-ln; EYP[1][2]=-0.10f+OZ;
    EYRAD[0]=EYRAD[1]=0.031f;
    EB[0][0]=EYP[0][0]; EB[0][1]=EYP[0][1]; EB[0][2]=OZ; EB[0][3]=0.16f;
}

// ---- quadruped leg (simple swing, fixed knee bend) ----
static void qleg(float sx, float sy, float side, float zoff, float ph, float run,
                 float amp, float ru, float rl, float ksign){
    float lift;
    float fx = sx + gaitFoot(ph, 0.62f, amp, &lift);
    float fy = 0.05f + (0.09f + 0.07f*run) * lift;
    float kx = (sx+fx)*0.5f + ksign*0.10f;
    float ky = (sy+fy)*0.5f + 0.03f;
    float z = zoff * side;
    cone(sx, sy, z, kx, ky, z + 0.01f*side, ru, ru*0.70f);
    cone(kx, ky, z + 0.01f*side, fx, fy, z + 0.02f*side, ru*0.66f, rl);
    // two hoof toes
    float tz = z + 0.02f*side;
    float tty = fy - 0.014f; if (tty < 0.034f) tty = 0.034f;
    cone(fx + 0.01f, fy, tz + 0.028f, fx + 0.085f, tty, tz + 0.050f, rl*0.52f, rl*0.24f);
    cone(fx + 0.01f, fy, tz - 0.028f, fx + 0.085f, tty, tz - 0.050f, rl*0.52f, rl*0.24f);
}

// stego spine: mirrors the body/neck/tail capsule chain -> surface top y at x
static float spineTop(float x, float bob){
    float ax, ay, bx, by, ra, rb;
    if      (x >  0.50f){ ax=0.50f;  ay=0.80f+bob; bx=0.92f;  by=0.56f;      ra=0.145f; rb=0.085f; }
    else if (x > -0.10f){ ax=-0.10f; ay=0.98f+bob; bx=0.50f;  by=0.80f+bob;  ra=0.34f;  rb=0.22f; }
    else if (x > -0.90f){ ax=-0.90f; ay=0.86f+bob; bx=-0.10f; by=0.98f+bob;  ra=0.30f;  rb=0.34f; }
    else                { ax=-0.90f; ay=0.86f+bob; bx=-1.42f; by=0.80f;      ra=0.20f;  rb=0.10f; }
    float t = (x - ax) / (bx - ax);
    if (t < 0.f) t = 0.f; if (t > 1.f) t = 1.f;
    return ay + (by - ay)*t + ra + (rb - ra)*t;
}
// plate params for exact material id in albedo: px, baseY, height, zOff, r1 (stego-local)
#define NPLT 9
static float PLT[NPLT][5];

// ---- stegosaurus ----
static void stego(const Kin *K){
    float ph = K->dist * 7.791f;   // = pi*ds/A -> no foot slip
    float run = K->run;
    float bob = (0.022f + 0.020f*run) * fsin(ph * 2.f);

    // arched body
    cone(-0.90f, 0.86f+bob, 0.f, -0.10f, 0.98f+bob, 0.f, 0.30f, 0.34f);
    cone(-0.10f, 0.98f+bob, 0.f,  0.50f, 0.80f+bob, 0.f, 0.34f, 0.22f);
    // low neck + small head
    cone(0.50f, 0.80f+bob, 0.f, 0.92f, 0.56f, 0.f, 0.145f, 0.085f);
    cone(0.92f, 0.56f, 0.f, 1.14f, 0.505f, 0.f, 0.082f, 0.040f);
    float chew = 0.5f + 0.5f*fsin(GT*1.1f + 0.8f);
    cone(0.96f, 0.515f, 0.f, 1.15f, 0.452f - 0.018f*chew, 0.f, 0.034f, 0.014f);
    EYP[2][0]=0.985f+OX; EYP[2][1]=0.585f; EYP[2][2]= 0.055f+OZ;
    EYP[3][0]=0.985f+OX; EYP[3][1]=0.585f; EYP[3][2]=-0.055f+OZ;
    EYRAD[2]=EYRAD[3]=0.017f;
    EB[1][0]=EYP[2][0]; EB[1][1]=EYP[2][1]; EB[1][2]=OZ; EB[1][3]=0.12f;
    // tail
    float z2 = 0.08f*fsin(ph-1.6f), z3 = 0.14f*fsin(ph-2.5f);
    cone(-0.90f, 0.86f+bob, 0.f, -1.42f, 0.80f, z2, 0.20f, 0.10f);
    cone(-1.42f, 0.80f, z2, -1.92f, 0.74f, z3, 0.10f, 0.030f);
    // back plates: rooted on the actual spine, nape (behind the skull) to mid-tail.
    // thin in z, widened along x (s=4.6); embedded 3cm for a clean smin blend
    static const float PX[NPLT] = { 0.86f, 0.62f, 0.38f, 0.12f,-0.14f,-0.40f,-0.66f,-0.92f,-1.18f };
    static const float PH_[NPLT]= { 0.10f, 0.17f, 0.27f, 0.39f, 0.45f, 0.40f, 0.30f, 0.20f, 0.13f };
    for (int i=0;i<NPLT;i++){
        float by = spineTop(PX[i], bob) - 0.030f;
        float zo = (i&1) ? 0.042f : -0.042f;
        if (PX[i] < -0.90f) zo += z2 * (PX[i] + 0.90f) / (-0.52f);   // ride tail sway
        float r1 = 0.018f + 0.038f * (PH_[i] / 0.45f);
        coneA(PX[i], by, zo, PX[i], by + PH_[i], zo,
              r1, 0.011f, 1.f/(4.6f*4.6f), 1.f, 1.f);
        PLT[i][0]=PX[i]; PLT[i][1]=by; PLT[i][2]=PH_[i]; PLT[i][3]=zo; PLT[i][4]=r1;
    }
    // thagomizer: 4 tail spikes
    cone(-1.80f, 0.78f, z3+0.06f, -2.02f, 0.98f, z3+0.16f, 0.030f, 0.005f);
    cone(-1.80f, 0.78f, z3-0.06f, -2.02f, 0.98f, z3-0.16f, 0.030f, 0.005f);
    cone(-1.66f, 0.78f, z3+0.07f, -1.84f, 1.00f, z3+0.17f, 0.030f, 0.005f);
    cone(-1.66f, 0.78f, z3-0.07f, -1.84f, 1.00f, z3-0.17f, 0.030f, 0.005f);
    // legs (trot: diagonal pairs)
    float P = 3.1415927f;
    qleg( 0.34f, 0.74f+bob,  1.f, 0.22f, ph,     run, 0.25f, 0.115f, 0.062f, -0.6f);
    qleg( 0.34f, 0.74f+bob, -1.f, 0.22f, ph+P,   run, 0.25f, 0.115f, 0.062f, -0.6f);
    qleg(-0.62f, 0.82f+bob,  1.f, 0.24f, ph+P,   run, 0.25f, 0.150f, 0.080f,  0.6f);
    qleg(-0.62f, 0.82f+bob, -1.f, 0.24f, ph,     run, 0.25f, 0.150f, 0.080f,  0.6f);
}

// ---- triceratops ----
static void trice(const Kin *K){
    float ph = K->dist * 8.469f;   // = pi*ds/A -> no foot slip
    float run = K->run;
    float bob = (0.024f + 0.022f*run) * fsin(ph * 2.f);
    float hd = -0.04f*run;   // head dips when charging

    // bulky body
    cone(-0.55f, 0.85f+bob, 0.f, 0.30f, 0.95f+bob, 0.f, 0.34f, 0.38f);
    cone( 0.30f, 0.95f+bob, 0.f, 0.62f, 0.88f+bob, 0.f, 0.38f, 0.27f);
    // big head + beak
    cone(0.62f, 0.95f+bob+hd, 0.f, 1.02f, 0.90f+bob+hd, 0.f, 0.205f, 0.125f);
    cone(1.02f, 0.86f+bob+hd, 0.f, 1.24f, 0.78f+bob+hd, 0.f, 0.075f, 0.012f);
    float chew = 0.5f + 0.5f*fsin(GT*1.2f + 2.2f);
    cone(0.97f, 0.795f+bob+hd, 0.f, 1.19f, 0.718f+bob+hd - 0.026f*chew, 0.f, 0.058f, 0.010f);
    EYP[4][0]=0.965f+OX; EYP[4][1]=0.975f+bob+hd; EYP[4][2]= 0.108f+OZ;
    EYP[5][0]=0.965f+OX; EYP[5][1]=0.975f+bob+hd; EYP[5][2]=-0.108f+OZ;
    EYRAD[4]=EYRAD[5]=0.022f;
    EB[2][0]=EYP[4][0]; EB[2][1]=EYP[4][1]; EB[2][2]=OZ; EB[2][3]=0.16f;
    // frill: thin disc normal to +X (axis pure X), widened y/z
    {
        float sy = 7.0f, sz = 5.6f;
        coneA(0.70f, 1.20f+bob+hd, 0.f, 0.62f, 1.20f+bob+hd, 0.f,
              0.050f, 0.050f, 1.f, 1.f/(sy*sy), 1.f/(sz*sz));
    }
    // brow horns + nose horn
    cone(0.94f, 1.04f+bob+hd,  0.105f, 1.34f, 1.38f+bob+hd,  0.18f, 0.042f, 0.006f);
    cone(0.94f, 1.04f+bob+hd, -0.105f, 1.34f, 1.38f+bob+hd, -0.18f, 0.042f, 0.006f);
    cone(1.06f, 0.95f+bob+hd, 0.f, 1.15f, 1.13f+bob+hd, 0.f, 0.038f, 0.007f);
    // tail
    float z2 = 0.07f*fsin(ph-1.4f);
    cone(-0.55f, 0.85f+bob, 0.f, -1.05f, 0.76f, z2*0.5f, 0.19f, 0.085f);
    cone(-1.05f, 0.76f, z2*0.5f, -1.38f, 0.70f, z2, 0.085f, 0.020f);
    // legs
    float P = 3.1415927f;
    qleg( 0.42f, 0.80f+bob,  1.f, 0.25f, ph,   run, 0.23f, 0.135f, 0.075f, -0.6f);
    qleg( 0.42f, 0.80f+bob, -1.f, 0.25f, ph+P, run, 0.23f, 0.135f, 0.075f, -0.6f);
    qleg(-0.36f, 0.80f+bob,  1.f, 0.26f, ph+P, run, 0.23f, 0.150f, 0.082f,  0.6f);
    qleg(-0.36f, 0.80f+bob, -1.f, 0.26f, ph,   run, 0.23f, 0.150f, 0.082f,  0.6f);
}

static void animate(float t){
    NP = 0;
    GT = t;
    SCROLL = VG * t;
    // species: lane z, base offset x0, drift D, freq w, phase, skew a
    Kin k0 = kin(t,  0.55f, 0.50f, 0.42f, 0.0f, 0.62f);  // theropod (front lane)
    Kin k1 = kin(t, -0.15f, 0.55f, 0.31f, 2.4f, 0.70f);  // stego (middle)
    Kin k2 = kin(t, -0.45f, 0.50f, 0.37f, 4.4f, 0.66f);  // trike (back lane)

    OX = k0.x; OZ = -1.40f;
    DPR[0][0]=NP; theropod(&k0); DPR[0][1]=NP;
    DB[0][0]=OX-0.18f; DB[0][1]=0.82f; DB[0][2]=OZ; DB[0][3]=1.66f;
    DXW[0]=OX; DZW[0]=OZ;

    OX = k1.x; OZ = 0.05f;
    DPR[1][0]=NP; stego(&k1); DPR[1][1]=NP;
    DB[1][0]=OX-0.40f; DB[1][1]=0.85f; DB[1][2]=OZ; DB[1][3]=1.82f;
    DXW[1]=OX; DZW[1]=OZ;

    OX = k2.x; OZ = 1.45f;
    DPR[2][0]=NP; trice(&k2); DPR[2][1]=NP;
    DB[2][0]=OX-0.05f; DB[2][1]=0.85f; DB[2][2]=OZ; DB[2][3]=1.62f;
    DXW[2]=OX; DZW[2]=OZ;

    OX = 0.f; OZ = 0.f;

    float lx=0.52f, ly=0.64f, lz=0.46f;
    float il = 1.f / fsqrt(lx*lx+ly*ly+lz*lz);
    SUNX=lx*il; SUNY=ly*il; SUNZ=lz*il;
}

// ---------- environment ----------
static inline C3 skyCol(V3 rd, int clouds){
    v4 sy = clamp01(vmul(rd.y, S(1.7f)));
    C3 c;
    c.r = mixv(S(0.82f), S(0.30f), sy);
    c.g = mixv(S(0.89f), S(0.52f), sy);
    c.b = mixv(S(1.00f), S(0.88f), sy);
    V3 Lv = v3(S(SUNX), S(SUNY), S(SUNZ));
    v4 sd = vmax(v3dot(rd, Lv), S(0.f));
    v4 g2 = vmul(sd,sd); g2=vmul(g2,g2); g2=vmul(g2,g2); g2=vmul(g2,g2); g2=vmul(g2,g2);
    c.r = vadd(c.r, vmul(g2, S(0.9f)));
    c.g = vadd(c.g, vmul(g2, S(0.75f)));
    c.b = vadd(c.b, vmul(g2, S(0.45f)));
    if (clouds){
        v4 iy = vdiv(S(1.f), vadd(vabs(rd.y), S(0.10f)));
        v4 u = vadd(vmul(vmul(rd.x, iy), S(0.9f)), S(GT*0.012f));
        v4 v = vmul(vmul(rd.z, iy), S(0.9f));
        v4 n = vadd(vmul(vnoise(u, v), S(0.62f)),
                    vmul(vnoise(vadd(vmul(u,S(2.6f)),S(11.3f)), vmul(v,S(2.6f))), S(0.38f)));
        v4 cov = vmul(clamp01(vmul(vsub(n, S(0.50f)), S(3.0f))),
                      clamp01(vmul(vsub(rd.y, S(0.03f)), S(7.f))));
        cov = vmul(cov, S(0.85f));
        c.r = mixv(c.r, S(0.99f), cov);
        c.g = mixv(c.g, S(0.99f), cov);
        c.b = mixv(c.b, S(1.00f), cov);
    }
    return c;
}

static inline C3 groundAlbedo(V3 P){
    v4 gu = vfloor(vmul(vadd(P.x, S(SCROLL)), S(1.1f)));
    v4 gv = vfloor(vmul(P.z, S(1.1f)));
    v4 gs = vadd(gu, gv);
    v4 ck = vsub(gs, vmul(S(2.f), vfloor(vmul(gs, S(0.5f)))));
    v4 grime = vmul(vnoise(vmul(vadd(P.x,S(SCROLL)),S(0.7f)), vmul(P.z,S(0.7f))), S(0.10f));
    C3 c;
    c.r = vadd(mixv(S(0.31f), S(0.41f), ck), grime);
    c.g = vadd(mixv(S(0.35f), S(0.45f), ck), grime);
    c.b = vadd(mixv(S(0.26f), S(0.35f), ck), grime);
    return c;
}

// per-lane dino id from hit position
static inline void dinoMasks(V3 P, v4 hit, v4 *m0, v4 *m1, v4 *m2){
    v4 d2[ND];
    for (int i=0;i<ND;i++){
        v4 dx=vsub(P.x,S(DB[i][0])), dy=vsub(P.y,S(DB[i][1])), dz=vsub(P.z,S(DB[i][2]));
        d2[i] = vadd(vadd(vmul(dx,dx),vmul(dy,dy)),vmul(dz,dz));
    }
    v4 b01 = vlt(d2[0], d2[1]);
    v4 dm  = sel(d2[0], d2[1], b01);
    v4 b2  = vlt(d2[2], dm);
    *m2 = vand(hit, b2);
    *m0 = vand(hit, vandn(b01, b2));
    *m1 = vand(hit, vandn(vnot(b01), b2));
}

// species albedo (textured)
static inline C3 dinoAlbedo(V3 P, V3 N, v4 m0, v4 m1, v4 m2){
    // local coords
    v4 selX = sel(S(DXW[0]), sel(S(DXW[1]), S(DXW[2]), m1), m0);
    v4 selZ = sel(S(DZW[0]), sel(S(DZW[1]), S(DZW[2]), m1), m0);
    v4 lx = vsub(P.x, selX), lz = vsub(P.z, selZ);
    // shared noise
    v4 n1 = vnoise(vmul(lx,S(2.6f)), vadd(vmul(P.y,S(2.6f)), vmul(lz,S(1.3f))));
    v4 n2 = vnoise(vadd(vmul(lx,S(8.f)),S(4.7f)), vmul(vadd(vmul(P.y,S(0.6f)),lz),S(8.f)));
    v4 topm = clamp01(vmul(N.y, S(1.5f)));

    // theropod: green w/ stripes
    v4 stripe = vmul(S(0.5f), vadd(vsin(vadd(vmul(lx,S(9.f)), vmul(lz,S(2.f)))), S(1.f)));
    v4 dk0 = vsub(S(1.f), vmul(S(0.25f), vmul(stripe, topm)));
    v4 r0 = vmul(S(0.24f),dk0), g0 = vmul(S(0.47f),dk0), b0 = vmul(S(0.21f),dk0);

    // stego: warm brown; plates red-orange gradient; spikes bone
    v4 r1c = S(0.50f), g1c = S(0.31f), b1c = S(0.15f);
    v4 pd = S(100.f), ty = S(0.f);
    if (any(m1)) for (int i=0;i<NPLT;i++){
        v4 hy = clamp01(vmul(vsub(P.y, S(PLT[i][1])), S(1.f/PLT[i][2])));
        v4 qx = vsub(lx, S(PLT[i][0]));
        v4 qy = vsub(vsub(P.y, S(PLT[i][1])), vmul(S(PLT[i][2]), hy));
        v4 qz = vsub(lz, S(PLT[i][3]));
        v4 dd = vsub(vsqrt(vadd(vadd(vmul(vmul(qx,qx), S(1.f/21.16f)), vmul(qy,qy)), vmul(qz,qz))),
                     mixv(S(PLT[i][4]), S(0.011f), hy));
        v4 closer = vlt(dd, pd);
        ty = sel(hy, ty, closer);
        pd = vmin(pd, dd);
    }
    v4 pm = vand(m1, vlt(pd, S(0.02f)));
    v4 pr = mixv(S(0.55f), S(0.88f), ty), pg = mixv(S(0.16f), S(0.42f), ty), pb = mixv(S(0.10f), S(0.16f), ty);
    r1c = sel(pr, r1c, pm); g1c = sel(pg, g1c, pm); b1c = sel(pb, b1c, pm);
    v4 sm_ = vand(m1, vand(vlt(lx, S(-1.55f)), vgt(P.y, S(0.62f))));
    r1c = sel(S(0.82f), r1c, sm_); g1c = sel(S(0.78f), g1c, sm_); b1c = sel(S(0.66f), b1c, sm_);

    // trike: grey-green; frill rings + warm rim; horns/beak bone
    v4 r2c = S(0.37f), g2c = S(0.39f), b2c = S(0.32f);
    v4 fm = vand(m2, vand(vgt(lx, S(0.40f)), vand(vlt(lx, S(0.92f)), vgt(P.y, S(0.92f)))));
    v4 fr = vsqrt(vadd(vmul(vsub(P.y,S(1.20f)),vsub(P.y,S(1.20f))), vmul(vmul(lz,lz),S(0.85f))));
    v4 ring = vmul(S(0.5f), vadd(vsin(vsub(vmul(fr,S(26.f)),S(1.6f))), S(1.f)));
    v4 fr_r = mixv(S(0.44f), S(0.27f), ring), fr_g = mixv(S(0.42f), S(0.24f), ring), fr_b = mixv(S(0.34f), S(0.20f), ring);
    v4 rim = clamp01(vmul(vsub(fr, S(0.36f)), S(7.f)));
    fr_r = mixv(fr_r, S(0.78f), rim); fr_g = mixv(fr_g, S(0.36f), rim); fr_b = mixv(fr_b, S(0.18f), rim);
    r2c = sel(fr_r, r2c, fm); g2c = sel(fr_g, g2c, fm); b2c = sel(fr_b, b2c, fm);
    v4 hm_ = vand(m2, vor(vand(vgt(P.y, S(1.06f)), vgt(lx, S(0.88f))),
                          vand(vgt(lx, S(1.05f)), vlt(P.y, S(0.90f)))));
    r2c = sel(S(0.80f), r2c, hm_); g2c = sel(S(0.74f), g2c, hm_); b2c = sel(S(0.60f), b2c, hm_);

    C3 c;
    c.r = sel(r0, sel(r1c, r2c, m1), m0);
    c.g = sel(g0, sel(g1c, g2c, m1), m0);
    c.b = sel(b0, sel(b1c, b2c, m1), m0);
    // noise modulation + belly lightening
    v4 nm = vmul(vadd(S(0.84f), vmul(S(0.26f), n1)), vadd(S(0.92f), vmul(S(0.16f), n2)));
    c.r = vmul(c.r, nm); c.g = vmul(c.g, nm); c.b = vmul(c.b, nm);
    v4 belly = clamp01(vmul(vsub(vneg(N.y), S(0.05f)), S(1.6f)));
    belly = vmul(vmul(belly,belly), vsub(S(3.f), vmul(S(2.f), belly)));
    c.r = mixv(c.r, S(0.80f), belly); c.g = mixv(c.g, S(0.74f), belly); c.b = mixv(c.b, S(0.55f), belly);
    // theropod eyes
    v4 em = vlt(eyeDistAll(P), S(0.006f));
    c.r = sel(S(0.05f), c.r, em); c.g = sel(S(0.05f), c.g, em); c.b = sel(S(0.06f), c.b, em);
    return c;
}

// ---------- soft shadow toward sun across all dino bounds ----------
static v4 softshadow(V3 p, v4 m){
    v4 sEnt = S(1e9f), sExit = S(-1e9f), vm = wasm_i32x4_splat(0);
    int dh[ND]; int nh=0;
    for (int i=0;i<ND;i++){
        v4 ocx=vsub(p.x,S(DB[i][0])), ocy=vsub(p.y,S(DB[i][1])), ocz=vsub(p.z,S(DB[i][2]));
        v4 b = vadd(vadd(vmul(ocx,S(SUNX)), vmul(ocy,S(SUNY))), vmul(ocz,S(SUNZ)));
        v4 c = vsub(vadd(vadd(vmul(ocx,ocx),vmul(ocy,ocy)),vmul(ocz,ocz)), S(DB[i][3]*DB[i][3]));
        v4 disc = vsub(vmul(b,b), c);
        v4 sq = vsqrt(vmax(disc, S(0.f)));
        v4 e1 = vsub(sq, b);
        v4 ok = vand(m, vand(vgt(disc, S(0.f)), vgt(e1, S(0.f))));
        dh[i] = any(ok);
        if (dh[i]){ nh=1;
            v4 e0 = vmax(vsub(vneg(b), sq), S(0.03f));
            sEnt  = sel(vmin(sEnt, e0), sEnt, ok);
            sExit = sel(vmax(sExit, e1), sExit, ok);
            vm = vor(vm, ok);
        }
    }
    if (!nh) return S(1.f);
    unsigned char LS[MAXP]; int ns=0;
    V3 sd = v3(S(SUNX), S(SUNY), S(SUNZ));
    for (int i=0;i<ND;i++) if (dh[i]) ns = buildList(p, sd, sEnt, sExit, 0.12f, DPR[i][0], DPR[i][1], LS, ns);
    if (!ns) return S(1.f);
    v4 s = sEnt;
    v4 res = S(1.f);
    v4 live = vm;
    for (int i=0;i<12;i++){
        V3 q = v3(vadd(p.x, vmul(S(SUNX), s)), vadd(p.y, vmul(S(SUNY), s)), vadd(p.z, vmul(S(SUNZ), s)));
        v4 d = mapL(q, LS, ns);
        res = sel(vmin(res, vmul(S(9.f), vdiv(d, s))), res, live);
        s = sel(vadd(s, vclampf(d, 0.04f, 0.30f)), s, live);
        live = vand(live, vand(vgt(res, S(0.015f)), vlt(s, sExit)));
        if (!any(live)) break;
    }
    return clamp01(sel(res, S(1.f), vm));
}

// ---------- main render ----------
__attribute__((export_name("fb")))
unsigned char* fbptr(void){ return FB; }

__attribute__((export_name("render")))
void render(float t, float az, float el, float dist, int w, int h){
    animate(t);

    const float tx=0.f, ty=0.85f, tz=0.f;
    float ce=fcos(el), se=fsin(el);
    float cxx = tx + dist*ce*fsin(az);
    float cyy = ty + dist*se;
    float czz = tz + dist*ce*fcos(az);
    float fx=tx-cxx, fy=ty-cyy, fz=tz-czz;
    float fl = 1.f/fsqrt(fx*fx+fy*fy+fz*fz); fx*=fl; fy*=fl; fz*=fl;
    float rx=-fz, rz=fx;
    float rl = 1.f/fsqrt(rx*rx+rz*rz); rx*=rl; rz*=rl;
    float ux = -rz*fy, uy = rz*fx - rx*fz, uz = rx*fy;
    const float FL = 1.8f;

    v4 X4 = wasm_f32x4_make(0.f,1.f,2.f,3.f);
    float ih = 2.0f/(float)h;
    V3 Lv = v3(S(SUNX), S(SUNY), S(SUNZ));

    for (int y=0; y<h; y++){
        float pyf = ((float)h*0.5f - ((float)y+0.5f)) * ih;
        unsigned char *row = FB + (unsigned)(y*w)*4u;
        for (int x=0; x<w; x+=4){
            v4 px = vmul(vsub(vadd(S((float)x+0.5f), X4), S((float)w*0.5f)), S(ih));
            v4 py = S(pyf);
            V3 rd = v3norm(v3(
                vadd(vadd(S(fx*FL), vmul(S(rx), px)), vmul(S(ux), py)),
                vadd(S(fy*FL), vmul(S(uy), py)),
                vadd(vadd(S(fz*FL), vmul(S(rz), px)), vmul(S(uz), py))));

            v4 down = vlt(rd.y, S(-1e-4f));
            v4 tg = sel(vdiv(S(-cyy), rd.y), S(1e9f), down);

            // dino bound spheres
            v4 t0 = S(1e9f), t1 = S(-1e9f), vm = wasm_i32x4_splat(0);
            int dh[ND];
            for (int i=0;i<ND;i++){
                float ocx=cxx-DB[i][0], ocy=cyy-DB[i][1], ocz=czz-DB[i][2];
                v4 b = vadd(vadd(vmul(S(ocx),rd.x), vmul(S(ocy),rd.y)), vmul(S(ocz),rd.z));
                float cc = ocx*ocx+ocy*ocy+ocz*ocz - DB[i][3]*DB[i][3];
                v4 disc = vsub(vmul(b,b), S(cc));
                v4 sq = vsqrt(vmax(disc, S(0.f)));
                v4 e1 = vsub(sq, b);
                v4 ok = vand(vgt(disc, S(0.f)), vgt(e1, S(0.f)));
                dh[i] = any(ok);
                if (dh[i]){
                    v4 e0 = vmax(vsub(vneg(b), sq), S(0.01f));
                    t0 = sel(vmin(t0, e0), t0, ok);
                    t1 = sel(vmax(t1, e1), t1, ok);
                    vm = vor(vm, ok);
                }
            }
            v4 tEnd = vmin(t1, tg);
            v4 doM = vand(vm, vlt(t0, tEnd));

            v4 tD = t0;
            v4 hit = wasm_i32x4_splat(0);
            unsigned char LP[MAXP];
            int np_ = 0;
            if (any(doM)){
                V3 ro = v3(S(cxx), S(cyy), S(czz));
                for (int i=0;i<ND;i++) if (dh[i])
                    np_ = buildList(ro, rd, t0, tEnd, 0.03f, DPR[i][0], DPR[i][1], LP, np_);
                v4 act = np_ ? doM : wasm_i32x4_splat(0);
                for (int i=0;i<64;i++){
                    V3 p = v3(vadd(S(cxx), vmul(rd.x,tD)),
                              vadd(S(cyy), vmul(rd.y,tD)),
                              vadd(S(czz), vmul(rd.z,tD)));
                    v4 d = mapLE(p, LP, np_);
                    v4 nh = vand(act, vlt(d, vadd(S(0.0012f), vmul(tD, S(0.0006f)))));
                    hit = vor(hit, nh);
                    act = vandn(act, nh);
                    tD = sel(vadd(tD, vmul(d, S(0.92f))), tD, act);
                    act = vand(act, vlt(tD, tEnd));
                    if (!any(act)) break;
                }
            }

            v4 gm = vandn(vand(down, vlt(tg, S(1e8f))), hit);
            v4 hm = vor(hit, gm);
            v4 tH = sel(tD, tg, hit);
            V3 P = v3(vadd(S(cxx), vmul(rd.x,tH)),
                      vadd(S(cyy), vmul(rd.y,tH)),
                      vadd(S(czz), vmul(rd.z,tH)));

            V3 N = v3(S(0.f), S(1.f), S(0.f));
            v4 ao = S(1.f);
            if (any(hit)){
                const float e = 0.0024f;
                v4 d1 = mapLE(v3(vadd(P.x,S(e)), vsub(P.y,S(e)), vsub(P.z,S(e))), LP, np_);
                v4 d2 = mapLE(v3(vsub(P.x,S(e)), vsub(P.y,S(e)), vadd(P.z,S(e))), LP, np_);
                v4 d3 = mapLE(v3(vsub(P.x,S(e)), vadd(P.y,S(e)), vsub(P.z,S(e))), LP, np_);
                v4 d4 = mapLE(v3(vadd(P.x,S(e)), vadd(P.y,S(e)), vadd(P.z,S(e))), LP, np_);
                V3 nn = v3norm(v3(
                    vadd(vsub(vsub(d1,d2),d3), d4),
                    vadd(vsub(vsub(d4,d1),d2), d3),
                    vadd(vsub(vsub(d2,d1),d3), d4)));
                N = v3(sel(nn.x,N.x,hit), sel(nn.y,N.y,hit), sel(nn.z,N.z,hit));
                v4 da = mapLE(v3add(P, v3scale(N, S(0.12f))), LP, np_);
                ao = sel(vadd(S(0.45f), vmul(S(0.55f), clamp01(vmul(da, S(1.f/0.12f))))), S(1.f), hit);
            }

            // albedo
            v4 m0=wasm_i32x4_splat(0), m1=m0, m2=m0;
            C3 alb;
            {
                C3 ga = groundAlbedo(P);
                alb = ga;
                if (any(hit)){
                    dinoMasks(P, hit, &m0, &m1, &m2);
                    C3 da_ = dinoAlbedo(P, N, m0, m1, m2);
                    alb.r = sel(da_.r, ga.r, hit);
                    alb.g = sel(da_.g, ga.g, hit);
                    alb.b = sel(da_.b, ga.b, hit);
                }
            }

            // lighting
            v4 ndl = vmax(v3dot(N, Lv), S(0.f));
            v4 sh = S(1.f);
            v4 lit = vand(hm, vgt(ndl, S(0.02f)));
            if (any(lit)){
                V3 sp_ = v3(vadd(P.x, vmul(N.x, S(0.012f))),
                            vadd(P.y, vmul(N.y, S(0.012f))),
                            vadd(P.z, vmul(N.z, S(0.012f))));
                sh = softshadow(sp_, lit);
            }
            v4 dif = vmul(ndl, sh);
            V3 Hv = v3norm(v3(vsub(S(SUNX), rd.x), vsub(S(SUNY), rd.y), vsub(S(SUNZ), rd.z)));
            v4 sp = vmax(v3dot(N, Hv), S(0.f));
            sp = vmul(sp,sp); sp = vmul(sp,sp); sp = vmul(sp,sp); sp = vmul(sp,sp);
            v4 em = vlt(eyeDistAll(P), S(0.006f));
            v4 spc = sel(S(1.4f), sel(S(0.35f), S(0.05f), hit), em);
            sp = vmul(vmul(sp, spc), sh);

            v4 amb = vmul(vadd(S(0.55f), vmul(S(0.45f), N.y)), ao);
            v4 cr = vadd(vmul(alb.r, vadd(vmul(dif, S(1.30f)), vmul(amb, S(0.42f)))), sp);
            v4 cg = vadd(vmul(alb.g, vadd(vmul(dif, S(1.22f)), vmul(amb, S(0.50f)))), sp);
            v4 cb = vadd(vmul(alb.b, vadd(vmul(dif, S(1.05f)), vmul(amb, S(0.66f)))), sp);

            // fresnel env reflection on dinos (analytic sky + ground plane, no clouds)
            if (any(hit)){
                v4 dnr = v3dot(N, rd);
                V3 rr = v3(vsub(rd.x, vmul(N.x, vmul(S(2.f),dnr))),
                           vsub(rd.y, vmul(N.y, vmul(S(2.f),dnr))),
                           vsub(rd.z, vmul(N.z, vmul(S(2.f),dnr))));
                C3 env = skyCol(rr, 0);
                v4 rdn = vlt(rr.y, S(-1e-3f));
                if (any(rdn)){
                    v4 tgr = vdiv(vneg(P.y), rr.y);
                    V3 GP = v3(vadd(P.x, vmul(rr.x,tgr)), S(0.f), vadd(P.z, vmul(rr.z,tgr)));
                    C3 gc = groundAlbedo(GP);
                    v4 fade = clamp01(vmul(tgr, S(0.18f)));
                    env.r = sel(mixv(vmul(gc.r,S(0.9f)), env.r, fade), env.r, rdn);
                    env.g = sel(mixv(vmul(gc.g,S(0.9f)), env.g, fade), env.g, rdn);
                    env.b = sel(mixv(vmul(gc.b,S(0.9f)), env.b, fade), env.b, rdn);
                }
                v4 ci = vmax(vneg(dnr), S(0.f));
                v4 f5 = vsub(S(1.f), ci); v4 f2=vmul(f5,f5); f5 = vmul(vmul(f2,f2),f5);
                v4 F = vadd(S(0.030f), vmul(S(0.55f), f5));
                // bone/horn parts a bit glossier
                v4 gl = sel(S(1.5f), S(1.f), vgt(alb.r, vadd(alb.g, S(0.25f))));
                F = vmul(vmul(F, gl), vadd(S(0.5f), vmul(S(0.5f), sh)));
                cr = sel(mixv(cr, env.r, F), cr, hit);
                cg = sel(mixv(cg, env.g, F), cg, hit);
                cb = sel(mixv(cb, env.b, F), cb, hit);
            }

            // reflective floor: mirror-march the herd
            if (any(gm)){
                V3 rro = v3(P.x, S(0.012f), P.z);
                V3 rrd = v3(rd.x, vneg(rd.y), rd.z);
                v4 rt0 = S(1e9f), rt1 = S(-1e9f), rvm = wasm_i32x4_splat(0);
                int rdh[ND];
                for (int i=0;i<ND;i++){
                    v4 ocx=vsub(rro.x,S(DB[i][0])), ocy=vsub(rro.y,S(DB[i][1])), ocz=vsub(rro.z,S(DB[i][2]));
                    v4 b = vadd(vadd(vmul(ocx,rrd.x), vmul(ocy,rrd.y)), vmul(ocz,rrd.z));
                    v4 c = vsub(vadd(vadd(vmul(ocx,ocx),vmul(ocy,ocy)),vmul(ocz,ocz)), S(DB[i][3]*DB[i][3]));
                    v4 disc = vsub(vmul(b,b), c);
                    v4 sq = vsqrt(vmax(disc, S(0.f)));
                    v4 e1 = vsub(sq, b);
                    v4 ok = vand(gm, vand(vgt(disc, S(0.f)), vgt(e1, S(0.f))));
                    rdh[i] = any(ok);
                    if (rdh[i]){
                        v4 e0 = vmax(vsub(vneg(b), sq), S(0.02f));
                        rt0 = sel(vmin(rt0, e0), rt0, ok);
                        rt1 = sel(vmax(rt1, e1), rt1, ok);
                        rvm = vor(rvm, ok);
                    }
                }
                C3 rc = skyCol(rrd, 1);   // default: reflected sky
                if (any(rvm)){
                    unsigned char LR[MAXP]; int nr=0;
                    for (int i=0;i<ND;i++) if (rdh[i])
                        nr = buildList(rro, rrd, rt0, rt1, 0.04f, DPR[i][0], DPR[i][1], LR, nr);
                    v4 rt = rt0, rhit = wasm_i32x4_splat(0);
                    if (nr){
                        v4 act = rvm;
                        for (int i=0;i<30;i++){
                            V3 q = v3(vadd(rro.x, vmul(rrd.x,rt)),
                                      vadd(rro.y, vmul(rrd.y,rt)),
                                      vadd(rro.z, vmul(rrd.z,rt)));
                            v4 d = mapLE(q, LR, nr);
                            v4 nh2 = vand(act, vlt(d, vadd(S(0.0025f), vmul(rt, S(0.0012f)))));
                            rhit = vor(rhit, nh2);
                            act = vandn(act, nh2);
                            rt = sel(vadd(rt, vmul(d, S(0.92f))), rt, act);
                            act = vand(act, vlt(rt, rt1));
                            if (!any(act)) break;
                        }
                    }
                    if (any(rhit)){
                        V3 RP = v3(vadd(rro.x, vmul(rrd.x,rt)),
                                   vadd(rro.y, vmul(rrd.y,rt)),
                                   vadd(rro.z, vmul(rrd.z,rt)));
                        const float e = 0.004f;
                        v4 d1 = mapLE(v3(vadd(RP.x,S(e)), vsub(RP.y,S(e)), vsub(RP.z,S(e))), LR, nr);
                        v4 d2 = mapLE(v3(vsub(RP.x,S(e)), vsub(RP.y,S(e)), vadd(RP.z,S(e))), LR, nr);
                        v4 d3 = mapLE(v3(vsub(RP.x,S(e)), vadd(RP.y,S(e)), vsub(RP.z,S(e))), LR, nr);
                        v4 d4 = mapLE(v3(vadd(RP.x,S(e)), vadd(RP.y,S(e)), vadd(RP.z,S(e))), LR, nr);
                        V3 RN = v3norm(v3(
                            vadd(vsub(vsub(d1,d2),d3), d4),
                            vadd(vsub(vsub(d4,d1),d2), d3),
                            vadd(vsub(vsub(d2,d1),d3), d4)));
                        v4 rm0,rm1,rm2;
                        dinoMasks(RP, rhit, &rm0, &rm1, &rm2);
                        C3 ra = dinoAlbedo(RP, RN, rm0, rm1, rm2);
                        v4 rdif = vmax(v3dot(RN, Lv), S(0.f));
                        v4 lr_ = vadd(vmul(ra.r, vadd(vmul(rdif,S(1.1f)), S(0.34f))), S(0.f));
                        v4 lg_ = vadd(vmul(ra.g, vadd(vmul(rdif,S(1.05f)), S(0.38f))), S(0.f));
                        v4 lb_ = vadd(vmul(ra.b, vadd(vmul(rdif,S(0.95f)), S(0.46f))), S(0.f));
                        rc.r = sel(lr_, rc.r, rhit);
                        rc.g = sel(lg_, rc.g, rhit);
                        rc.b = sel(lb_, rc.b, rhit);
                    }
                }
                // fresnel blend into floor
                v4 ci = vmax(vneg(rd.y), S(0.f));
                v4 f5 = vsub(S(1.f), ci); v4 f2=vmul(f5,f5); f5 = vmul(vmul(f2,f2),f5);
                v4 kR = vmin(vadd(S(0.14f), vmul(S(0.60f), f5)), S(0.80f));
                kR = vmul(kR, vadd(S(0.55f), vmul(S(0.45f), sh)));  // dimmer in shadow
                cr = sel(mixv(cr, rc.r, kR), cr, gm);
                cg = sel(mixv(cg, rc.g, kR), cg, gm);
                cb = sel(mixv(cb, rc.b, kR), cb, gm);
            }

            // fog
            v4 fk = vmul(tH, tH);
            v4 fog = vdiv(vmul(fk, S(0.0016f)), vadd(S(1.f), vmul(fk, S(0.0016f))));
            cr = mixv(cr, S(0.78f), fog); cg = mixv(cg, S(0.86f), fog); cb = mixv(cb, S(0.98f), fog);

            // sky
            C3 sk = skyCol(rd, 1);
            cr = sel(cr, sk.r, hm); cg = sel(cg, sk.g, hm); cb = sel(cb, sk.b, hm);

            // pack (gamma 2.0)
            v128_t ri = wasm_i32x4_trunc_sat_f32x4(vmul(vsqrt(clamp01(cr)), S(255.f)));
            v128_t gi = wasm_i32x4_trunc_sat_f32x4(vmul(vsqrt(clamp01(cg)), S(255.f)));
            v128_t bi = wasm_i32x4_trunc_sat_f32x4(vmul(vsqrt(clamp01(cb)), S(255.f)));
            v128_t outp = wasm_v128_or(wasm_v128_or(ri, wasm_i32x4_shl(gi, 8)),
                          wasm_v128_or(wasm_i32x4_shl(bi, 16), wasm_i32x4_splat((int)0xff000000u)));
            wasm_v128_store(row + (unsigned)x*4u, outp);
        }
    }
}
