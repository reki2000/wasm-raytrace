// render.h — rendering engine interface: SDF primitive store (anisotropic
// tapered capsules + hard-min eye spheres), instance registry, per-instance
// surface materials, and the frame pipeline. Hot evaluation functions are
// static inline here so every module compiles them in place.
#ifndef RENDER_H
#define RENDER_H
#include "vec.h"

// ---------- primitive store: anisotropic tapered capsules ----------
// q: 0..2 a | 3..5 b-a | 6 1/l2 | 7 r1 | 8 r2-r1 | 9..11 cull center | 12 SK+boundR | 13..15 Ax,Ay,Az
// Axis-aligned anisotropy A<=1 widens that axis by 1/sqrt(A); Lipschitz stays <=1 (safe to march).
// Constraint: the segment axis must lie within A=1 axes.
#define MAXP 96
#define SK 0.062f
#ifndef CAM_TARGET_Y
// Orbit-camera target height. Aim near the dinosaurs' heads so tall heads,
// frills, and horns stay in frame instead of centering on the torso.
#define CAM_TARGET_Y 1.15f
#endif
extern float PR[MAXP][16];
extern int NP;
extern float OX, OZ;   // current dino origin offset (applied by cone/coneA)

void coneA(float ax,float ay,float az, float bx,float by,float bz,
           float r1,float r2, float Ax,float Ay,float Az);
void cone(float ax,float ay,float az, float bx,float by,float bz, float r1,float r2);
// cone + record the segment (species-local coords) for exact material id in albedo
void boneCone(float *s, float ax,float ay,float az, float bx,float by,float bz,
              float r1,float r2);

// eyes: 2 per instance; hard-min spheres (not smin-blended) so they stay crisp
#define NEY 6
extern float EYP[NEY][3];
extern float EYRAD[NEY];
extern float EB[3][4];             // per-instance eye-pair bound cx,cy,cz,r

// ---------- instance registry (filled by the scene each frame) ----------
#define ND 3
extern float DB[ND][4];     // bound sphere cx,cy,cz,r
extern int   DPR[ND][2];    // prim range [start,end)
extern float DXW[ND], DZW[ND]; // world offsets (texture space / id)
extern float SUNX, SUNY, SUNZ; // sun direction (normalized)

// ---------- per-instance surface materials ----------
// mode: 0 plain(flat) | 1 textured | 2 metallic | 3 acrylic (refractive)
extern int   M_MODE[ND];
extern float M_REFL[ND];    // reflectivity (0.5 = stock look)
extern float M_TRAN[ND];    // transmittance (acrylic)
extern float M_IOR[ND];     // index of refraction (acrylic)
extern float M_TEX[ND];     // procedural texture amount
extern float M_GLOSS[ND];   // specular strength (0.5 = stock)
extern float SH_LIFT[ND];   // shadow lightening (translucent)

void setMaterial(int i, int mode, float refl, float tran, float ior, float tex, float gloss);

// lane-select a per-instance scalar via the exclusive instance masks
static inline v4 dsel(const float *v, v4 m0, v4 m1){
    return sel(S(v[0]), sel(S(v[1]), S(v[2]), m1), m0);
}
static inline v4 modeMask(int mode, v4 m0, v4 m1, v4 m2){
    v4 r = wasm_i32x4_splat(0);
    if (M_MODE[0] == mode) r = vor(r, m0);
    if (M_MODE[1] == mode) r = vor(r, m1);
    if (M_MODE[2] == mode) r = vor(r, m2);
    return r;
}

// ---------- SDF evaluation ----------
static inline v4 smin(v4 a, v4 b){
    v4 h = clamp01(vfma(vsub(b,a), S(0.5f/SK), S(0.5f)));
    return vsub(mixv(b, a, h), vmul(S(SK), vmul(h, vsub(S(1.f), h))));
}

static inline v4 eyePair(V3 p, int k){
    v4 d = S(100.f);
    for (int i=k*2;i<k*2+2;i++){
        v4 ex=vsub(p.x,S(EYP[i][0])), ey=vsub(p.y,S(EYP[i][1])), ez=vsub(p.z,S(EYP[i][2]));
        v4 e2 = vfma(ez,ez, vfma(ey,ey, vmul(ex,ex)));
        d = vmin(d, vsub(vsqrt(e2), S(EYRAD[i])));
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
        v4 tc = vneg(vfma(ocz,rd.z, vfma(ocy,rd.y, vmul(ocx,rd.x))));
        tc = vmin(vmax(tc, tA), tB);
        v4 px = vfma(rd.x,tc, ocx);
        v4 py = vfma(rd.y,tc, ocy);
        v4 pz = vfma(rd.z,tc, ocz);
        v4 d2 = vfma(pz,pz, vfma(py,py, vmul(px,px)));
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
        v4 d2 = vfma(cz,cz, vfma(cy,cy, vmul(cx,cx)));
        v4 rh = vadd(d, S(q[12]));
        if (!any(vlt(d2, vmul(rh,rh)))) continue;
        v4 pax=vsub(p.x,S(q[0])), pay=vsub(p.y,S(q[1])), paz=vsub(p.z,S(q[2]));
        v4 h = clamp01(vmul(vfma(paz,S(q[5]), vfma(pay,S(q[4]), vmul(pax,S(q[3])))), S(q[6])));
        v4 qx=vfnma(S(q[3]),h,pax), qy=vfnma(S(q[4]),h,pay), qz=vfnma(S(q[5]),h,paz);
        v4 t = vfma(vmul(qz,qz),S(q[15]), vfma(vmul(qy,qy),S(q[14]), vmul(vmul(qx,qx),S(q[13]))));
        v4 dd = vsub(vsqrt(t), vfma(S(q[8]),h, S(q[7])));
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

// tapered-segment distance in species-local coords (for exact material id)
static inline v4 segDist(v4 px, v4 py, v4 pz, const float *s){
    v4 ax=vsub(px,S(s[0])), ay=vsub(py,S(s[1])), az=vsub(pz,S(s[2]));
    float bx=s[3]-s[0], by=s[4]-s[1], bz=s[5]-s[2];
    float il2 = 1.f/(bx*bx + by*by + bz*bz);
    v4 h = clamp01(vmul(vfma(az,S(bz), vfma(ay,S(by), vmul(ax,S(bx)))), S(il2)));
    v4 qx=vfnma(S(bx),h,ax), qy=vfnma(S(by),h,ay), qz=vfnma(S(bz),h,az);
    return vsub(vsqrt(vfma(qz,qz, vfma(qy,qy, vmul(qx,qx)))),
                vfma(S(s[7]-s[6]), h, S(s[6])));
}

// ---------- environment + species masks ----------
// Shared with scene.c (the combined SDF+mesh renderer): sky/ground color,
// per-instance species masks from a march hit, and the SDF-only soft shadow
// (generic over any world point, regardless of which renderer produced it).
C3 skyCol(V3 rd, int clouds);
C3 groundAlbedo(V3 P);
void dinoMasks(V3 P, v4 hit, const unsigned char *L, int n, v4 *m0, v4 *m1, v4 *m2);
v4 softshadow(V3 p, v4 m);

// ---------- frame pipeline ----------
// Renders rows [y0,y1) of the frame into fb (RGBA8, w*h). Scene must already
// be built (animate) and the clock advanced (animTick) for this t. Rows are
// fully independent (disjoint fb writes, read-only scene state), so this may
// be called concurrently by multiple threads with different [y0,y1) ranges.
void renderRows(float az, float el, float dist, int w, int h, unsigned char *fb, int y0, int y1);
// Renders the full frame (all rows) — single-threaded convenience wrapper.
void renderFrame(float az, float el, float dist, int w, int h, unsigned char *fb);

#endif
