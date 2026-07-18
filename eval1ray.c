// eval1ray.c — apples-to-apples benchmark kernels for two SIMD strategies.
//
// Both kernels render the exact same reduced pipeline over the same scene
// state (built by animate()): instance bound spheres, culled prim lists,
// 64-step sphere trace, tetrahedral normals, AO, flat per-instance albedo,
// checker ground, sun soft shadows, specular, fog, cloudy sky, gamma pack.
//
//   renderLite4 — current strategy: SoA, one v128 lane per ray (4 rays/packet)
//   renderLite1 — one ray at a time: AoS, one v128 holds (x,y,z,0) of a vector
//
// The full renderer's extras (fresnel env, metal/acrylic, floor mirror) are
// built from the same map-eval primitives, so the relative result carries.
#include "vec.h"
#include "render.h"
#include "anim.h"
#include "eval1ray.h"

// flat per-instance albedo, shared by both kernels (keeps dino_model's
// per-lane texture code out of the comparison)
static const float DCOL[ND][3] = {
    {0.46f, 0.38f, 0.27f},
    {0.35f, 0.43f, 0.29f},
    {0.42f, 0.35f, 0.40f},
};

// ==========================================================================
// renderLite4 — reduced copy of renderRows (render.c), same packet strategy
// ==========================================================================
void renderLite4(float az, float el, float dist, int w, int h, unsigned char *fb){
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
        unsigned char *row = fb + (unsigned)(y*w)*4u;
        for (int x=0; x<w; x+=4){
            v4 px = vmul(vsub(vadd(S((float)x+0.5f), X4), S((float)w*0.5f)), S(ih));
            v4 py = S(pyf);
            V3 rd = v3norm(v3(
                vadd(vadd(S(fx*FL), vmul(S(rx), px)), vmul(S(ux), py)),
                vadd(S(fy*FL), vmul(S(uy), py)),
                vadd(vadd(S(fz*FL), vmul(S(rz), px)), vmul(S(uz), py))));

            v4 down = vlt(rd.y, S(-1e-4f));
            v4 tg = sel(vdiv(S(-cyy), rd.y), S(1e9f), down);

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

            // albedo: checker ground, flat color per instance
            C3 alb = groundAlbedo(P);
            if (any(hit)){
                v4 m0,m1,m2;
                dinoMasks(P, hit, LP, np_, &m0, &m1, &m2);
                v4 ar = sel(S(DCOL[0][0]), sel(S(DCOL[1][0]), S(DCOL[2][0]), m1), m0);
                v4 ag = sel(S(DCOL[0][1]), sel(S(DCOL[1][1]), S(DCOL[2][1]), m1), m0);
                v4 ab = sel(S(DCOL[0][2]), sel(S(DCOL[1][2]), S(DCOL[2][2]), m1), m0);
                alb.r = sel(ar, alb.r, hit);
                alb.g = sel(ag, alb.g, hit);
                alb.b = sel(ab, alb.b, hit);
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
            v4 spc = sel(S(0.35f), S(0.05f), hit);
            sp = vmul(vmul(sp, spc), sh);

            v4 amb = vmul(vadd(S(0.55f), vmul(S(0.45f), N.y)), ao);
            v4 cr = vadd(vmul(alb.r, vadd(vmul(dif, S(1.30f)), vmul(amb, S(0.42f)))), sp);
            v4 cg = vadd(vmul(alb.g, vadd(vmul(dif, S(1.22f)), vmul(amb, S(0.50f)))), sp);
            v4 cb = vadd(vmul(alb.b, vadd(vmul(dif, S(1.05f)), vmul(amb, S(0.66f)))), sp);

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

// ==========================================================================
// renderLite1 — one ray at a time; SIMD used across xyz (AoS: v128 = x,y,z,0)
// ==========================================================================

// --- AoS vec3 helpers: lane 3 is kept 0 so 4-lane horizontal sums are dot3 ---
typedef v128_t vf;
static inline vf F3(float x, float y, float z){ return wasm_f32x4_make(x,y,z,0.f); }
static inline float lane0(vf v){ return wasm_f32x4_extract_lane(v,0); }
static inline float lane1(vf v){ return wasm_f32x4_extract_lane(v,1); }
static inline float lane2(vf v){ return wasm_f32x4_extract_lane(v,2); }
static inline float hsum(vf v){          // x+y+z+w (w must be 0)
    vf t = wasm_f32x4_add(v, wasm_i32x4_shuffle(v,v,2,3,0,1));
    t = wasm_f32x4_add(t, wasm_i32x4_shuffle(t,t,1,0,3,2));
    return wasm_f32x4_extract_lane(t,0);
}
static inline float dot3(vf a, vf b){ return hsum(wasm_f32x4_mul(a,b)); }
static inline vf norm3(vf a){ return vmul(a, S(1.f/fsqrt(dot3(a,a)))); }

static inline float fmin1(float a, float b){ return a<b?a:b; }
static inline float fmax1(float a, float b){ return a>b?a:b; }
static inline float clamp01f(float x){ return x<0.f?0.f:(x>1.f?1.f:x); }
static inline float mix1(float a, float b, float t){ return a + (b-a)*t; }
static inline float smin1(float a, float b){
    float hh = clamp01f((b-a)*(0.5f/SK) + 0.5f);
    return (b + (a-b)*hh) - SK*hh*(1.f-hh);
}

// scalar 2D value noise (same formulas as vec.h's vnoise, one lane)
static inline float fract1(float x){ return x - __builtin_floorf(x); }
static inline float hash1(float ix, float iy){
    return fract1(fsin(ix*127.1f + iy*311.7f) * 43758.547f);
}
static inline float noise1(float x, float y){
    float ix = __builtin_floorf(x), iy = __builtin_floorf(y);
    float fx_ = x-ix, fy_ = y-iy;
    float ux = fx_*fx_*(3.f-2.f*fx_), uy = fy_*fy_*(3.f-2.f*fy_);
    float a = hash1(ix,iy),      b = hash1(ix+1.f,iy);
    float c = hash1(ix,iy+1.f),  d = hash1(ix+1.f,iy+1.f);
    return mix1(mix1(a,b,ux), mix1(c,d,ux), uy);
}

// --- per-frame AoS mirror of the prim store (repacked so lane 3 is 0) ---
typedef struct { vf a, ba, c, A; float il2, r1, dr, rb; } Prim1;
static Prim1 P1S[MAXP];
static vf EY1[NEY]; static vf EB1[3]; static float EBR1[3];
static vf DB1[ND];  static float DBR1[ND];

static void prep1(void){
    for (int i=0;i<NP;i++){
        const float *q = PR[i];
        Prim1 *p = &P1S[i];
        p->a  = F3(q[0],q[1],q[2]);
        p->ba = F3(q[3],q[4],q[5]);
        p->c  = F3(q[9],q[10],q[11]);
        p->A  = F3(q[13],q[14],q[15]);
        p->il2 = q[6]; p->r1 = q[7]; p->dr = q[8]; p->rb = q[12];
    }
    for (int i=0;i<NEY;i++) EY1[i] = F3(EYP[i][0],EYP[i][1],EYP[i][2]);
    for (int k=0;k<3;k++){ EB1[k] = F3(EB[k][0],EB[k][1],EB[k][2]); EBR1[k] = EB[k][3]; }
    for (int i=0;i<ND;i++){ DB1[i] = F3(DB[i][0],DB[i][1],DB[i][2]); DBR1[i] = DB[i][3]; }
}

static inline float eyePair1(vf p, int k){
    float d = 100.f;
    for (int i=k*2;i<k*2+2;i++){
        vf e = vsub(p, EY1[i]);
        d = fmin1(d, fsqrt(dot3(e,e)) - EYRAD[i]);
    }
    return d;
}

static inline float mapL1(vf p, const unsigned char *L, int n){
    float d = 100.f;
    for (int k=0;k<n;k++){
        const Prim1 *q = &P1S[L[k]];
        vf c = vsub(p, q->c);
        float c2 = dot3(c,c);
        float rh = d + q->rb;
        if (c2 >= rh*rh) continue;
        vf pa = vsub(p, q->a);
        float hh = clamp01f(dot3(pa, q->ba) * q->il2);
        vf qq = vfnma(q->ba, S(hh), pa);
        float t = dot3(vmul(qq,qq), q->A);
        d = smin1(d, fsqrt(t) - (q->r1 + q->dr*hh));
    }
    return d;
}

static inline float mapLE1(vf p, const unsigned char *L, int n){
    float d = mapL1(p, L, n);
    for (int k=0;k<3;k++){
        vf e = vsub(p, EB1[k]);
        float e2 = dot3(e,e);
        float re = d + EBR1[k];
        if (e2 < re*re) d = fmin1(d, eyePair1(p,k));
    }
    return d;
}

static inline int buildList1(vf ro, vf rd, float tA, float tB, float margin,
                             int from, int to, unsigned char *L, int n){
    for (int i=from;i<to;i++){
        const Prim1 *q = &P1S[i];
        vf oc = vsub(ro, q->c);
        float tc = -dot3(oc, rd);
        tc = fmin1(fmax1(tc, tA), tB);
        vf p = vfma(rd, S(tc), oc);
        float d2 = dot3(p,p);
        float th = q->rb + margin;
        if (d2 < th*th) L[n++] = (unsigned char)i;
    }
    return n;
}

// nearest instance id at a hit point (1-ray counterpart of dinoMasks)
static inline int dinoId1(vf p, const unsigned char *L, int n){
    float dd[ND];
    for (int i=0;i<ND;i++) dd[i] = eyePair1(p, i);
    for (int k=0;k<n;k++){
        int idx = L[k];
        int di = idx >= DPR[2][0] ? 2 : (idx >= DPR[1][0] ? 1 : 0);
        const Prim1 *q = &P1S[idx];
        vf c = vsub(p, q->c);
        float c2 = dot3(c,c);
        float rh = dd[di] + q->rb;
        if (c2 >= rh*rh) continue;
        vf pa = vsub(p, q->a);
        float hh = clamp01f(dot3(pa, q->ba) * q->il2);
        vf qq = vfnma(q->ba, S(hh), pa);
        float t = dot3(vmul(qq,qq), q->A);
        dd[di] = fmin1(dd[di], fsqrt(t) - (q->r1 + q->dr*hh));
    }
    int best = 0;
    if (dd[1] < dd[best]) best = 1;
    if (dd[2] < dd[best]) best = 2;
    return best;
}

// sun soft shadow, 1-ray counterpart of softshadow()
static float softshadow1(vf p){
    float res = 1.f;
    vf sd = F3(SUNX, SUNY, SUNZ);
    for (int i=0;i<ND;i++){
        if (SH_LIFT[i] >= 0.995f) continue;
        vf oc = vsub(p, DB1[i]);
        float b = dot3(oc, sd);
        float c = dot3(oc,oc) - DBR1[i]*DBR1[i];
        float disc = b*b - c;
        if (disc <= 0.f) continue;
        float sq = fsqrt(disc);
        float e1 = sq - b;
        if (e1 <= 0.f) continue;
        float e0 = fmax1(-b - sq, 0.03f);
        unsigned char LS[MAXP]; int ns = 0;
        ns = buildList1(p, sd, e0, e1, 0.12f, DPR[i][0], DPR[i][1], LS, ns);
        if (!ns) continue;
        float s = e0, ri = 1.f;
        for (int k=0;k<10;k++){
            vf q = vfma(sd, S(s), p);
            float d = mapL1(q, LS, ns);
            ri = fmin1(ri, 9.f * d / s);
            s += fclampf(d, 0.04f, 0.30f);
            if (ri <= 0.015f || s >= e1) break;
        }
        ri = clamp01f(ri);
        if (SH_LIFT[i] > 0.f) ri = mix1(ri, 1.f, SH_LIFT[i]);
        res *= ri;
    }
    return res;
}

// scalar sky/ground (same formulas as skyCol/groundAlbedo, one lane)
static void sky1(float rx, float ry, float rz, int clouds, float *r, float *g, float *b){
    float sy = clamp01f(ry * 1.7f);
    float cr = mix1(0.82f, 0.30f, sy);
    float cg = mix1(0.89f, 0.52f, sy);
    float cb = mix1(1.00f, 0.88f, sy);
    float sd = fmax1(rx*SUNX + ry*SUNY + rz*SUNZ, 0.f);
    float g2 = sd*sd; g2*=g2; g2*=g2; g2*=g2; g2*=g2;
    cr += g2*0.9f; cg += g2*0.75f; cb += g2*0.45f;
    if (clouds){
        float iy = 1.f/(ffabs(ry) + 0.10f);
        float u = rx*iy*0.9f + GT*0.012f;
        float v = rz*iy*0.9f;
        float n = noise1(u,v)*0.62f + noise1(u*2.6f+11.3f, v*2.6f)*0.38f;
        float cov = clamp01f((n-0.50f)*3.0f) * clamp01f((ry-0.03f)*7.f) * 0.85f;
        cr = mix1(cr, 0.99f, cov);
        cg = mix1(cg, 0.99f, cov);
        cb = mix1(cb, 1.00f, cov);
    }
    *r=cr; *g=cg; *b=cb;
}

static void ground1(float px, float pz, float *r, float *g, float *b){
    float gu = __builtin_floorf((px+SCROLL)*1.1f);
    float gv = __builtin_floorf(pz*1.1f);
    float gs = gu+gv;
    float ck = gs - 2.f*__builtin_floorf(gs*0.5f);
    float grime = noise1((px+SCROLL)*0.7f, pz*0.7f) * 0.10f;
    *r = mix1(0.31f,0.41f,ck) + grime;
    *g = mix1(0.35f,0.45f,ck) + grime;
    *b = mix1(0.26f,0.35f,ck) + grime;
}

// ==========================================================================
// renderLite1p — one ray at a time; SIMD across 4 primitives (SoA groups).
// The per-pixel prim list is transposed once into 4-wide groups, then every
// map evaluation during the march computes 4 prim distances per iteration.
// ==========================================================================

typedef struct {
    float ax[4],ay[4],az[4], bx[4],by[4],bz[4], cx[4],cy[4],cz[4],
          Ax[4],Ay[4],Az[4], il2[4], r1[4], dr[4], rb[4];
} PrimG;

// transpose listed prims into 4-wide groups; pad with a far-away dummy whose
// distance is huge (smin(a, huge) == a) and whose bound never passes the cull
static int packGroups(const unsigned char *L, int n, PrimG *G){
    int ng = (n+3)/4;
    for (int k=0;k<ng*4;k++){
        int gi = k>>2, li = k&3;
        PrimG *g = &G[gi];
        if (k < n){
            const float *q = PR[L[k]];
            g->ax[li]=q[0]; g->ay[li]=q[1]; g->az[li]=q[2];
            g->bx[li]=q[3]; g->by[li]=q[4]; g->bz[li]=q[5];
            g->cx[li]=q[9]; g->cy[li]=q[10]; g->cz[li]=q[11];
            g->Ax[li]=q[13]; g->Ay[li]=q[14]; g->Az[li]=q[15];
            g->il2[li]=q[6]; g->r1[li]=q[7]; g->dr[li]=q[8]; g->rb[li]=q[12];
        } else {
            g->ax[li]=1e3f; g->ay[li]=1e3f; g->az[li]=1e3f;
            g->bx[li]=1.f;  g->by[li]=0.f;  g->bz[li]=0.f;
            g->cx[li]=1e3f; g->cy[li]=1e3f; g->cz[li]=1e3f;
            g->Ax[li]=1.f;  g->Ay[li]=1.f;  g->Az[li]=1.f;
            g->il2[li]=1.f; g->r1[li]=0.f;  g->dr[li]=0.f;  g->rb[li]=0.f;
        }
    }
    return ng;
}

#define LD(f) wasm_v128_load(f)

static inline float mapG(vf p, const PrimG *G, int ng){
    float d = 100.f;
    v4 px = S(lane0(p)), py = S(lane1(p)), pz = S(lane2(p));
    for (int gi=0;gi<ng;gi++){
        const PrimG *g = &G[gi];
        v4 cx = vsub(px, LD(g->cx)), cy = vsub(py, LD(g->cy)), cz = vsub(pz, LD(g->cz));
        v4 c2 = vfma(cz,cz, vfma(cy,cy, vmul(cx,cx)));
        v4 rh = vadd(S(d), LD(g->rb));
        if (!any(vlt(c2, vmul(rh,rh)))) continue;
        v4 pax = vsub(px, LD(g->ax)), pay = vsub(py, LD(g->ay)), paz = vsub(pz, LD(g->az));
        v4 bax = LD(g->bx), bay = LD(g->by), baz = LD(g->bz);
        v4 hh = clamp01(vmul(vfma(paz,baz, vfma(pay,bay, vmul(pax,bax))), LD(g->il2)));
        v4 qx = vfnma(bax,hh,pax), qy = vfnma(bay,hh,pay), qz = vfnma(baz,hh,paz);
        v4 t = vfma(vmul(qz,qz),LD(g->Az), vfma(vmul(qy,qy),LD(g->Ay), vmul(vmul(qx,qx),LD(g->Ax))));
        v4 dd = vsub(vsqrt(t), vfma(LD(g->dr),hh, LD(g->r1)));
        // sequential per-lane smin keeps list-order semantics identical to mapL1
        d = smin1(d, wasm_f32x4_extract_lane(dd,0));
        d = smin1(d, wasm_f32x4_extract_lane(dd,1));
        d = smin1(d, wasm_f32x4_extract_lane(dd,2));
        d = smin1(d, wasm_f32x4_extract_lane(dd,3));
    }
    return d;
}

static inline float mapGE(vf p, const PrimG *G, int ng){
    float d = mapG(p, G, ng);
    for (int k=0;k<3;k++){
        vf e = vsub(p, EB1[k]);
        float e2 = dot3(e,e);
        float re = d + EBR1[k];
        if (e2 < re*re) d = fmin1(d, eyePair1(p,k));
    }
    return d;
}

// sun soft shadow with prim-parallel map
static float softshadow1p(vf p){
    float res = 1.f;
    vf sd = F3(SUNX, SUNY, SUNZ);
    for (int i=0;i<ND;i++){
        if (SH_LIFT[i] >= 0.995f) continue;
        vf oc = vsub(p, DB1[i]);
        float b = dot3(oc, sd);
        float c = dot3(oc,oc) - DBR1[i]*DBR1[i];
        float disc = b*b - c;
        if (disc <= 0.f) continue;
        float sq = fsqrt(disc);
        float e1 = sq - b;
        if (e1 <= 0.f) continue;
        float e0 = fmax1(-b - sq, 0.03f);
        unsigned char LS[MAXP]; int ns = 0;
        ns = buildList1(p, sd, e0, e1, 0.12f, DPR[i][0], DPR[i][1], LS, ns);
        if (!ns) continue;
        PrimG GS[MAXP/4+1];
        int ngs = packGroups(LS, ns, GS);
        float s = e0, ri = 1.f;
        for (int k=0;k<10;k++){
            vf q = vfma(sd, S(s), p);
            float d = mapG(q, GS, ngs);
            ri = fmin1(ri, 9.f * d / s);
            s += fclampf(d, 0.04f, 0.30f);
            if (ri <= 0.015f || s >= e1) break;
        }
        ri = clamp01f(ri);
        if (SH_LIFT[i] > 0.f) ri = mix1(ri, 1.f, SH_LIFT[i]);
        res *= ri;
    }
    return res;
}

void renderLite1p(float az, float el, float dist, int w, int h, unsigned char *fb){
    prep1();
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

    float ih = 2.0f/(float)h;
    vf ro = F3(cxx, cyy, czz);
    vf Lv = F3(SUNX, SUNY, SUNZ);
    vf Fv = F3(fx*FL, fy*FL, fz*FL);
    vf Rv = F3(rx, 0.f, rz);
    vf Uv = F3(ux, uy, uz);

    for (int y=0; y<h; y++){
        float pyf = ((float)h*0.5f - ((float)y+0.5f)) * ih;
        unsigned char *row = fb + (unsigned)(y*w)*4u;
        for (int x=0; x<w; x++){
            float pxf = (((float)x+0.5f) - (float)w*0.5f) * ih;
            vf rd = norm3(vadd(vadd(Fv, vmul(Rv, S(pxf))), vmul(Uv, S(pyf))));
            float rdy = lane1(rd);

            int down = rdy < -1e-4f;
            float tg = down ? -cyy/rdy : 1e9f;

            float t0 = 1e9f, t1 = -1e9f;
            int dh[ND], anyd = 0;
            for (int i=0;i<ND;i++){
                vf oc = vsub(ro, DB1[i]);
                float b = dot3(oc, rd);
                float cc = dot3(oc,oc) - DBR1[i]*DBR1[i];
                float disc = b*b - cc;
                dh[i] = 0;
                if (disc > 0.f){
                    float sq = fsqrt(disc);
                    float e1 = sq - b;
                    if (e1 > 0.f){
                        dh[i] = 1; anyd = 1;
                        float e0 = fmax1(-b - sq, 0.01f);
                        t0 = fmin1(t0, e0);
                        t1 = fmax1(t1, e1);
                    }
                }
            }
            float tEnd = fmin1(t1, tg);

            float tD = t0;
            int hit = 0;
            unsigned char LP[MAXP];
            PrimG GP[MAXP/4+1];
            int np_ = 0, ngp = 0;
            if (anyd && t0 < tEnd){
                for (int i=0;i<ND;i++) if (dh[i])
                    np_ = buildList1(ro, rd, t0, tEnd, 0.03f, DPR[i][0], DPR[i][1], LP, np_);
                if (np_){
                    ngp = packGroups(LP, np_, GP);
                    for (int i=0;i<64;i++){
                        vf p = vfma(rd, S(tD), ro);
                        float d = mapGE(p, GP, ngp);
                        if (d < 0.0012f + tD*0.0006f){ hit = 1; break; }
                        tD += d*0.92f;
                        if (tD >= tEnd) break;
                    }
                }
            }

            int gm = !hit && down && tg < 1e8f;
            int hm = hit || gm;
            float tH = hit ? tD : tg;
            vf P = vfma(rd, S(tH), ro);

            vf N = F3(0.f, 1.f, 0.f);
            float ao = 1.f;
            if (hit){
                const float e = 0.0024f;
                float d1 = mapGE(vadd(P, F3( e,-e,-e)), GP, ngp);
                float d2 = mapGE(vadd(P, F3(-e,-e, e)), GP, ngp);
                float d3 = mapGE(vadd(P, F3(-e, e,-e)), GP, ngp);
                float d4 = mapGE(vadd(P, F3( e, e, e)), GP, ngp);
                N = norm3(F3(d1-d2-d3+d4, d4-d1-d2+d3, d2-d1-d3+d4));
                float da = mapGE(vfma(N, S(0.12f), P), GP, ngp);
                ao = 0.45f + 0.55f*clamp01f(da*(1.f/0.12f));
            }

            float ar, ag, ab;
            if (hit){
                int id = dinoId1(P, LP, np_);
                ar = DCOL[id][0]; ag = DCOL[id][1]; ab = DCOL[id][2];
            } else {
                ground1(lane0(P), lane2(P), &ar, &ag, &ab);
            }

            float ndl = fmax1(dot3(N, Lv), 0.f);
            float sh = 1.f;
            if (hm && ndl > 0.02f)
                sh = softshadow1p(vfma(N, S(0.012f), P));
            float dif = ndl*sh;
            vf Hv = norm3(vsub(Lv, rd));
            float sp = fmax1(dot3(N, Hv), 0.f);
            sp = sp*sp; sp = sp*sp; sp = sp*sp; sp = sp*sp;
            sp = sp * (hit ? 0.35f : 0.05f) * sh;

            float amb = (0.55f + 0.45f*lane1(N)) * ao;
            float cr = ar*(dif*1.30f + amb*0.42f) + sp;
            float cg = ag*(dif*1.22f + amb*0.50f) + sp;
            float cb = ab*(dif*1.05f + amb*0.66f) + sp;

            float fk = tH*tH;
            float fog = fk*0.0016f / (1.f + fk*0.0016f);
            cr = mix1(cr, 0.78f, fog); cg = mix1(cg, 0.86f, fog); cb = mix1(cb, 0.98f, fog);

            if (!hm) sky1(lane0(rd), rdy, lane2(rd), 1, &cr, &cg, &cb);

            unsigned char *px4 = row + (unsigned)x*4u;
            px4[0] = (unsigned char)(int)(fsqrt(clamp01f(cr))*255.f);
            px4[1] = (unsigned char)(int)(fsqrt(clamp01f(cg))*255.f);
            px4[2] = (unsigned char)(int)(fsqrt(clamp01f(cb))*255.f);
            px4[3] = 255;
        }
    }
}

void renderLite1(float az, float el, float dist, int w, int h, unsigned char *fb){
    prep1();
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

    float ih = 2.0f/(float)h;
    vf ro = F3(cxx, cyy, czz);
    vf Lv = F3(SUNX, SUNY, SUNZ);
    vf Fv = F3(fx*FL, fy*FL, fz*FL);
    vf Rv = F3(rx, 0.f, rz);
    vf Uv = F3(ux, uy, uz);

    for (int y=0; y<h; y++){
        float pyf = ((float)h*0.5f - ((float)y+0.5f)) * ih;
        unsigned char *row = fb + (unsigned)(y*w)*4u;
        for (int x=0; x<w; x++){
            float pxf = (((float)x+0.5f) - (float)w*0.5f) * ih;
            vf rd = norm3(vadd(vadd(Fv, vmul(Rv, S(pxf))), vmul(Uv, S(pyf))));
            float rdy = lane1(rd);

            int down = rdy < -1e-4f;
            float tg = down ? -cyy/rdy : 1e9f;

            // instance bound spheres
            float t0 = 1e9f, t1 = -1e9f;
            int dh[ND], anyd = 0;
            for (int i=0;i<ND;i++){
                vf oc = vsub(ro, DB1[i]);
                float b = dot3(oc, rd);
                float cc = dot3(oc,oc) - DBR1[i]*DBR1[i];
                float disc = b*b - cc;
                dh[i] = 0;
                if (disc > 0.f){
                    float sq = fsqrt(disc);
                    float e1 = sq - b;
                    if (e1 > 0.f){
                        dh[i] = 1; anyd = 1;
                        float e0 = fmax1(-b - sq, 0.01f);
                        t0 = fmin1(t0, e0);
                        t1 = fmax1(t1, e1);
                    }
                }
            }
            float tEnd = fmin1(t1, tg);

            float tD = t0;
            int hit = 0;
            unsigned char LP[MAXP];
            int np_ = 0;
            if (anyd && t0 < tEnd){
                for (int i=0;i<ND;i++) if (dh[i])
                    np_ = buildList1(ro, rd, t0, tEnd, 0.03f, DPR[i][0], DPR[i][1], LP, np_);
                if (np_){
                    for (int i=0;i<64;i++){
                        vf p = vfma(rd, S(tD), ro);
                        float d = mapLE1(p, LP, np_);
                        if (d < 0.0012f + tD*0.0006f){ hit = 1; break; }
                        tD += d*0.92f;
                        if (tD >= tEnd) break;
                    }
                }
            }

            int gm = !hit && down && tg < 1e8f;
            int hm = hit || gm;
            float tH = hit ? tD : tg;
            vf P = vfma(rd, S(tH), ro);

            vf N = F3(0.f, 1.f, 0.f);
            float ao = 1.f;
            if (hit){
                const float e = 0.0024f;
                float d1 = mapLE1(vadd(P, F3( e,-e,-e)), LP, np_);
                float d2 = mapLE1(vadd(P, F3(-e,-e, e)), LP, np_);
                float d3 = mapLE1(vadd(P, F3(-e, e,-e)), LP, np_);
                float d4 = mapLE1(vadd(P, F3( e, e, e)), LP, np_);
                N = norm3(F3(d1-d2-d3+d4, d4-d1-d2+d3, d2-d1-d3+d4));
                float da = mapLE1(vfma(N, S(0.12f), P), LP, np_);
                ao = 0.45f + 0.55f*clamp01f(da*(1.f/0.12f));
            }

            // albedo: checker ground, flat color per instance
            float ar, ag, ab;
            if (hit){
                int id = dinoId1(P, LP, np_);
                ar = DCOL[id][0]; ag = DCOL[id][1]; ab = DCOL[id][2];
            } else {
                ground1(lane0(P), lane2(P), &ar, &ag, &ab);
            }

            // lighting
            float ndl = fmax1(dot3(N, Lv), 0.f);
            float sh = 1.f;
            if (hm && ndl > 0.02f)
                sh = softshadow1(vfma(N, S(0.012f), P));
            float dif = ndl*sh;
            vf Hv = norm3(vsub(Lv, rd));
            float sp = fmax1(dot3(N, Hv), 0.f);
            sp = sp*sp; sp = sp*sp; sp = sp*sp; sp = sp*sp;
            sp = sp * (hit ? 0.35f : 0.05f) * sh;

            float amb = (0.55f + 0.45f*lane1(N)) * ao;
            float cr = ar*(dif*1.30f + amb*0.42f) + sp;
            float cg = ag*(dif*1.22f + amb*0.50f) + sp;
            float cb = ab*(dif*1.05f + amb*0.66f) + sp;

            // fog
            float fk = tH*tH;
            float fog = fk*0.0016f / (1.f + fk*0.0016f);
            cr = mix1(cr, 0.78f, fog); cg = mix1(cg, 0.86f, fog); cb = mix1(cb, 0.98f, fog);

            // sky
            if (!hm) sky1(lane0(rd), rdy, lane2(rd), 1, &cr, &cg, &cb);

            // pack (gamma 2.0)
            unsigned char *px4 = row + (unsigned)x*4u;
            px4[0] = (unsigned char)(int)(fsqrt(clamp01f(cr))*255.f);
            px4[1] = (unsigned char)(int)(fsqrt(clamp01f(cg))*255.f);
            px4[2] = (unsigned char)(int)(fsqrt(clamp01f(cb))*255.f);
            px4[3] = 255;
        }
    }
}
