// render.c — rendering engine: primitive registration, environment (sky /
// ground), soft shadows, and the SIMD 4-ray-packet frame pipeline
// (sphere tracing, lighting, materials, floor reflection, fog, output).
// Scene-specific surface color comes from dinoAlbedo() (dino_model.c).
#include "vec.h"
#include "render.h"
#include "anim.h"
#include "dino_model.h"

float PR[MAXP][16];
int NP = 0;
float OX = 0.f, OZ = 0.f;

void coneA(float ax,float ay,float az, float bx,float by,float bz,
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
void cone(float ax,float ay,float az, float bx,float by,float bz, float r1,float r2){
    coneA(ax,ay,az,bx,by,bz,r1,r2,1.f,1.f,1.f);
}
void boneCone(float *s, float ax,float ay,float az, float bx,float by,float bz,
              float r1,float r2){
    s[0]=ax; s[1]=ay; s[2]=az; s[3]=bx; s[4]=by; s[5]=bz; s[6]=r1; s[7]=r2;
    cone(ax,ay,az,bx,by,bz,r1,r2);
}

float EYP[NEY][3];
float EYRAD[NEY];
float EB[3][4];

float DB[ND][4];
int   DPR[ND][2];
float DXW[ND], DZW[ND];
float SUNX, SUNY, SUNZ;

int   M_MODE[ND] = {1,1,1};
float M_REFL[ND] = {0.5f,0.5f,0.5f};
float M_TRAN[ND] = {0.65f,0.65f,0.65f};
float M_IOR[ND]  = {1.49f,1.49f,1.49f};
float M_TEX[ND]  = {1.f,1.f,1.f};
float M_GLOSS[ND]= {0.5f,0.5f,0.5f};
float SH_LIFT[ND]= {0.f,0.f,0.f};

void setMaterial(int i, int mode, float refl, float tran, float ior, float tex, float gloss){
    if (i < 0 || i >= ND) return;
    M_MODE[i] = mode;
    M_REFL[i] = fclampf(refl, 0.f, 1.f);
    M_TRAN[i] = fclampf(tran, 0.f, 1.f);
    M_IOR[i]  = fclampf(ior, 1.01f, 2.5f);
    M_TEX[i]  = fclampf(tex, 0.f, 1.f);
    M_GLOSS[i]= fclampf(gloss, 0.f, 1.f);
    SH_LIFT[i]= mode == 3 ? M_TRAN[i] * 0.85f : 0.f;
}

// ---------- environment ----------
C3 skyCol(V3 rd, int clouds){
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

C3 groundAlbedo(V3 P){
    v4 gu = vfloor(vmul(P.x, S(1.1f)));
    v4 gv = vfloor(vmul(P.z, S(1.1f)));
    v4 gs = vadd(gu, gv);
    v4 ck = vsub(gs, vmul(S(2.f), vfloor(vmul(gs, S(0.5f)))));
    v4 grime = vmul(vnoise(vmul(P.x,S(0.7f)), vmul(P.z,S(0.7f))), S(0.10f));
    C3 c;
    c.r = vadd(mixv(S(0.31f), S(0.41f), ck), grime);
    c.g = vadd(mixv(S(0.35f), S(0.45f), ck), grime);
    c.b = vadd(mixv(S(0.26f), S(0.35f), ck), grime);
    return c;
}

// per-lane instance id from hit position: nearest per-instance SDF over the
// active prim list (the hit surface's prim is always in the list, at distance ~0)
void dinoMasks(V3 P, v4 hit, const unsigned char *L, int n,
               v4 *m0, v4 *m1, v4 *m2){
    v4 dd[ND];
    for (int i=0;i<ND;i++) dd[i] = eyePair(P, i);
    for (int k=0;k<n;k++){
        int idx = L[k];
        int di = idx >= DPR[2][0] ? 2 : (idx >= DPR[1][0] ? 1 : 0);
        const float *q = PR[idx];
        v4 cx = vsub(P.x,S(q[9])), cy = vsub(P.y,S(q[10])), cz = vsub(P.z,S(q[11]));
        v4 c2 = vfma(cz,cz, vfma(cy,cy, vmul(cx,cx)));
        v4 rh = vadd(dd[di], S(q[12]));
        if (!any(vlt(c2, vmul(rh,rh)))) continue;
        v4 pax=vsub(P.x,S(q[0])), pay=vsub(P.y,S(q[1])), paz=vsub(P.z,S(q[2]));
        v4 h = clamp01(vmul(vfma(paz,S(q[5]), vfma(pay,S(q[4]), vmul(pax,S(q[3])))), S(q[6])));
        v4 qx=vfnma(S(q[3]),h,pax), qy=vfnma(S(q[4]),h,pay), qz=vfnma(S(q[5]),h,paz);
        v4 t = vfma(vmul(qz,qz),S(q[15]), vfma(vmul(qy,qy),S(q[14]), vmul(vmul(qx,qx),S(q[13]))));
        dd[di] = vmin(dd[di], vsub(vsqrt(t), vfma(S(q[8]),h, S(q[7]))));
    }
    v4 b01 = vlt(dd[0], dd[1]);
    v4 dm  = sel(dd[0], dd[1], b01);
    v4 b2  = vlt(dd[2], dm);
    *m2 = vand(hit, b2);
    *m0 = vand(hit, vandn(b01, b2));
    *m1 = vand(hit, vandn(vnot(b01), b2));
}

// ---------- soft shadow toward sun, per instance (translucent ones cast lighter shadows) ----------
v4 softshadow(V3 p, v4 m){
    v4 res = S(1.f);
    V3 sd = v3(S(SUNX), S(SUNY), S(SUNZ));
    for (int i=0;i<ND;i++){
        if (SH_LIFT[i] >= 0.995f) continue;
        v4 ocx=vsub(p.x,S(DB[i][0])), ocy=vsub(p.y,S(DB[i][1])), ocz=vsub(p.z,S(DB[i][2]));
        v4 b = vadd(vadd(vmul(ocx,S(SUNX)), vmul(ocy,S(SUNY))), vmul(ocz,S(SUNZ)));
        v4 c = vsub(vadd(vadd(vmul(ocx,ocx),vmul(ocy,ocy)),vmul(ocz,ocz)), S(DB[i][3]*DB[i][3]));
        v4 disc = vsub(vmul(b,b), c);
        v4 sq = vsqrt(vmax(disc, S(0.f)));
        v4 e1 = vsub(sq, b);
        v4 ok = vand(m, vand(vgt(disc, S(0.f)), vgt(e1, S(0.f))));
        if (!any(ok)) continue;
        v4 e0 = vmax(vsub(vneg(b), sq), S(0.03f));
        unsigned char LS[MAXP]; int ns = 0;
        ns = buildList(p, sd, e0, e1, 0.12f, DPR[i][0], DPR[i][1], LS, ns);
        if (!ns) continue;
        v4 s = e0, ri = S(1.f), live = ok;
        for (int k=0;k<10;k++){
            V3 q = v3(vadd(p.x, vmul(S(SUNX), s)), vadd(p.y, vmul(S(SUNY), s)), vadd(p.z, vmul(S(SUNZ), s)));
            v4 d = mapL(q, LS, ns);
            ri = sel(vmin(ri, vmul(S(9.f), vdiv(d, s))), ri, live);
            s = sel(vadd(s, vclampf(d, 0.04f, 0.30f)), s, live);
            live = vand(live, vand(vgt(ri, S(0.015f)), vlt(s, e1)));
            if (!any(live)) break;
        }
        ri = clamp01(sel(ri, S(1.f), ok));
        if (SH_LIFT[i] > 0.f) ri = mixv(ri, S(1.f), S(SH_LIFT[i]));
        res = vmul(res, ri);
    }
    return res;
}

// ---------- secondary-ray herd retrace ----------
// March a secondary ray (acrylic exit ray, metal mirror ray) against the
// herd, excluding each lane's own instance, and where it hits replace *env
// with the shaded surface — so glass bodies show the dinosaurs behind them
// and mirror bodies reflect the ones around them, not just the analytic
// sky/ground.
static void herdRetrace(V3 ro2, V3 od, v4 mask, const v4 selfM[3], C3 *env){
    V3 Lv = v3(S(SUNX), S(SUNY), S(SUNZ));
    v4 tt0 = S(1e9f), tt1 = S(-1e9f), tvm = wasm_i32x4_splat(0);
    int tdh[ND];
    for (int i=0;i<ND;i++){
        v4 ocx=vsub(ro2.x,S(DB[i][0])), ocy=vsub(ro2.y,S(DB[i][1])), ocz=vsub(ro2.z,S(DB[i][2]));
        v4 bb = vadd(vadd(vmul(ocx,od.x), vmul(ocy,od.y)), vmul(ocz,od.z));
        v4 cc = vsub(vadd(vadd(vmul(ocx,ocx),vmul(ocy,ocy)),vmul(ocz,ocz)), S(DB[i][3]*DB[i][3]));
        v4 disc = vsub(vmul(bb,bb), cc);
        v4 sq = vsqrt(vmax(disc, S(0.f)));
        v4 e1 = vsub(sq, bb);
        v4 ok = vand(vandn(mask, selfM[i]), vand(vgt(disc, S(0.f)), vgt(e1, S(0.f))));
        tdh[i] = any(ok);
        if (tdh[i]){
            v4 e0 = vmax(vsub(vneg(bb), sq), S(0.f));
            tt0 = sel(vmin(tt0, e0), tt0, ok);
            tt1 = sel(vmax(tt1, e1), tt1, ok);
            tvm = vor(tvm, ok);
        }
    }
    if (!any(tvm)) return;
    unsigned char LT[MAXP]; int nt = 0;
    for (int i=0;i<ND;i++) if (tdh[i])
        nt = buildList(ro2, od, tt0, tt1, 0.03f, DPR[i][0], DPR[i][1], LT, nt);
    v4 tt = tt0, thit = wasm_i32x4_splat(0);
    if (nt){
        v4 act = tvm;
        for (int i=0;i<40;i++){
            V3 q = v3(vadd(ro2.x, vmul(od.x,tt)),
                      vadd(ro2.y, vmul(od.y,tt)),
                      vadd(ro2.z, vmul(od.z,tt)));
            v4 d = mapLE(q, LT, nt);
            v4 nh3 = vand(act, vlt(d, vadd(S(0.0025f), vmul(tt, S(0.0012f)))));
            thit = vor(thit, nh3);
            act = vandn(act, nh3);
            tt = sel(vadd(tt, vmul(d, S(0.92f))), tt, act);
            act = vand(act, vlt(tt, tt1));
            if (!any(act)) break;
        }
    }
    if (!any(thit)) return;
    V3 TP = v3(vadd(ro2.x, vmul(od.x,tt)),
               vadd(ro2.y, vmul(od.y,tt)),
               vadd(ro2.z, vmul(od.z,tt)));
    const float e3 = 0.004f;
    v4 h1 = mapLE(v3(vadd(TP.x,S(e3)), vsub(TP.y,S(e3)), vsub(TP.z,S(e3))), LT, nt);
    v4 h2 = mapLE(v3(vsub(TP.x,S(e3)), vsub(TP.y,S(e3)), vadd(TP.z,S(e3))), LT, nt);
    v4 h3 = mapLE(v3(vsub(TP.x,S(e3)), vadd(TP.y,S(e3)), vsub(TP.z,S(e3))), LT, nt);
    v4 h4 = mapLE(v3(vadd(TP.x,S(e3)), vadd(TP.y,S(e3)), vadd(TP.z,S(e3))), LT, nt);
    V3 TN = v3norm(v3(
        vadd(vsub(vsub(h1,h2),h3), h4),
        vadd(vsub(vsub(h4,h1),h2), h3),
        vadd(vsub(vsub(h2,h1),h3), h4)));
    v4 tm0,tm1,tm2;
    dinoMasks(TP, thit, LT, nt, &tm0, &tm1, &tm2);
    C3 ta = dinoAlbedo(TP, TN, tm0, tm1, tm2);
    v4 tdif = vmax(v3dot(TN, Lv), S(0.f));
    v4 tr_ = vmul(ta.r, vadd(vmul(tdif,S(1.1f)), S(0.34f)));
    v4 tg_ = vmul(ta.g, vadd(vmul(tdif,S(1.05f)), S(0.38f)));
    v4 tb_ = vmul(ta.b, vadd(vmul(tdif,S(0.95f)), S(0.46f)));
    v4 tMet = modeMask(2, tm0, tm1, tm2);
    v4 tAcr = modeMask(3, tm0, tm1, tm2);
    if (any(tMet)){   // tinted sky mirror
        v4 trefl = dsel(M_REFL, tm0, tm1);
        v4 dn3 = v3dot(TN, od);
        V3 rr3 = v3(vsub(od.x, vmul(TN.x, vmul(S(2.f),dn3))),
                    vsub(od.y, vmul(TN.y, vmul(S(2.f),dn3))),
                    vsub(od.z, vmul(TN.z, vmul(S(2.f),dn3))));
        C3 e3c = skyCol(rr3, 0);
        v4 mR = vmin(vadd(vmul(ta.r, S(1.9f)), S(0.08f)), S(1.f));
        v4 mG = vmin(vadd(vmul(ta.g, S(1.9f)), S(0.08f)), S(1.f));
        v4 mB = vmin(vadd(vmul(ta.b, S(1.9f)), S(0.08f)), S(1.f));
        tr_ = sel(mixv(tr_, vmul(e3c.r,mR), trefl), tr_, tMet);
        tg_ = sel(mixv(tg_, vmul(e3c.g,mG), trefl), tg_, tMet);
        tb_ = sel(mixv(tb_, vmul(e3c.b,mB), trefl), tb_, tMet);
    }
    if (any(tAcr)){   // let the sky behind show through
        v4 ttran = dsel(M_TRAN, tm0, tm1);
        C3 skyBeyond = skyCol(od, 0);
        tr_ = sel(mixv(tr_, skyBeyond.r, ttran), tr_, tAcr);
        tg_ = sel(mixv(tg_, skyBeyond.g, ttran), tg_, tAcr);
        tb_ = sel(mixv(tb_, skyBeyond.b, ttran), tb_, tAcr);
    }
    env->r = sel(tr_, env->r, thit);
    env->g = sel(tg_, env->g, thit);
    env->b = sel(tb_, env->b, thit);
}

// ---------- frame pipeline ----------
void renderRows(float az, float el, float dist, int w, int h, unsigned char *fb, int y0, int y1){
    const float tx=0.f, ty=CAM_TARGET_Y, tz=0.f;
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

    for (int y=y0; y<y1; y++){
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

            // instance bound spheres
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
                    dinoMasks(P, hit, LP, np_, &m0, &m1, &m2);
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
            v4 gv = vmul(dsel(M_GLOSS, m0, m1), S(0.7f));   // 0.35 at stock gloss
            gv = sel(vmul(gv, S(3.4f)), gv, modeMask(2, m0, m1, m2));
            v4 spc = sel(S(1.4f), sel(gv, S(0.05f), hit), em);
            sp = vmul(vmul(sp, spc), sh);

            v4 amb = vmul(vadd(S(0.55f), vmul(S(0.45f), N.y)), ao);
            v4 cr = vadd(vmul(alb.r, vadd(vmul(dif, S(1.30f)), vmul(amb, S(0.42f)))), sp);
            v4 cg = vadd(vmul(alb.g, vadd(vmul(dif, S(1.22f)), vmul(amb, S(0.50f)))), sp);
            v4 cb = vadd(vmul(alb.b, vadd(vmul(dif, S(1.05f)), vmul(amb, S(0.66f)))), sp);

            // fresnel env reflection on dinos (analytic sky + ground plane, no clouds)
            if (any(hit)){
                v4 reflP = dsel(M_REFL, m0, m1);
                v4 mMet = vand(modeMask(2, m0, m1, m2), hit);
                v4 mAcr = vand(modeMask(3, m0, m1, m2), hit);
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

                // acrylic: refract in, march through the body to the back face,
                // refract out, sample env along the exit ray; thick parts pick up tint
                if (any(mAcr)){
                    v4 iorP = dsel(M_IOR, m0, m1);
                    v4 tranP = dsel(M_TRAN, m0, m1);
                    v4 eta = vdiv(S(1.f), iorP);
                    v4 kk = vsub(S(1.f), vmul(vmul(eta,eta), vsub(S(1.f), vmul(ci,ci))));
                    v4 tcf = vsub(vmul(eta,ci), vsqrt(vmax(kk, S(0.f))));
                    V3 td = v3(vadd(vmul(rd.x,eta), vmul(N.x,tcf)),
                               vadd(vmul(rd.y,eta), vmul(N.y,tcf)),
                               vadd(vmul(rd.z,eta), vmul(N.z,tcf)));
                    v4 s = S(0.03f);
                    v4 live = mAcr;
                    for (int it=0; it<18; it++){       // interior march (SDF < 0)
                        V3 q = v3(vadd(P.x, vmul(td.x,s)),
                                  vadd(P.y, vmul(td.y,s)),
                                  vadd(P.z, vmul(td.z,s)));
                        v4 d = mapLE(q, LP, np_);
                        live = vand(live, vlt(d, S(-0.004f)));
                        if (!any(live)) break;
                        s = sel(vadd(s, vclampf(vneg(d), 0.016f, 0.45f)), s, live);
                    }
                    V3 XP = v3(vadd(P.x, vmul(td.x,s)),
                               vadd(P.y, vmul(td.y,s)),
                               vadd(P.z, vmul(td.z,s)));
                    const float e2 = 0.004f;
                    v4 g1 = mapLE(v3(vadd(XP.x,S(e2)), vsub(XP.y,S(e2)), vsub(XP.z,S(e2))), LP, np_);
                    v4 g2 = mapLE(v3(vsub(XP.x,S(e2)), vsub(XP.y,S(e2)), vadd(XP.z,S(e2))), LP, np_);
                    v4 g3 = mapLE(v3(vsub(XP.x,S(e2)), vadd(XP.y,S(e2)), vsub(XP.z,S(e2))), LP, np_);
                    v4 g4 = mapLE(v3(vadd(XP.x,S(e2)), vadd(XP.y,S(e2)), vadd(XP.z,S(e2))), LP, np_);
                    V3 N2 = v3norm(v3(
                        vadd(vsub(vsub(g1,g2),g3), g4),
                        vadd(vsub(vsub(g4,g1),g2), g3),
                        vadd(vsub(vsub(g2,g1),g3), g4)));
                    v4 c2 = vmax(v3dot(td, N2), S(0.f));
                    v4 k2 = vsub(S(1.f), vmul(vmul(iorP,iorP), vsub(S(1.f), vmul(c2,c2))));
                    v4 tir = vlt(k2, S(0.f));          // total internal reflection: keep td
                    v4 tc2 = vsub(vsqrt(vmax(k2, S(0.f))), vmul(iorP, c2));
                    V3 od = v3(vadd(vmul(td.x,iorP), vmul(N2.x,tc2)),
                               vadd(vmul(td.y,iorP), vmul(N2.y,tc2)),
                               vadd(vmul(td.z,iorP), vmul(N2.z,tc2)));
                    od = v3(sel(td.x, od.x, tir), sel(td.y, od.y, tir), sel(td.z, od.z, tir));
                    C3 tenv = skyCol(od, 0);
                    v4 odn = vlt(od.y, S(-1e-3f));
                    if (any(odn)){
                        v4 tg2 = vdiv(vneg(XP.y), vmin(od.y, S(-1e-3f)));
                        V3 GP2 = v3(vadd(XP.x, vmul(od.x,tg2)), S(0.f), vadd(XP.z, vmul(od.z,tg2)));
                        C3 gc2 = groundAlbedo(GP2);
                        v4 fd2 = clamp01(vmul(tg2, S(0.18f)));
                        tenv.r = sel(mixv(vmul(gc2.r,S(0.9f)), tenv.r, fd2), tenv.r, odn);
                        tenv.g = sel(mixv(vmul(gc2.g,S(0.9f)), tenv.g, fd2), tenv.g, odn);
                        tenv.b = sel(mixv(vmul(gc2.b,S(0.9f)), tenv.b, fd2), tenv.b, odn);
                    }

                    // see through: march the exit ray against the rest of the herd
                    // (excluding the acrylic instance itself) so other dinosaurs show
                    // up behind/through the acrylic body instead of just sky/ground
                    {
                        V3 ro2 = v3(vadd(XP.x, vmul(od.x, S(0.04f))),
                                    vadd(XP.y, vmul(od.y, S(0.04f))),
                                    vadd(XP.z, vmul(od.z, S(0.04f))));
                        v4 selfM[3] = { m0, m1, m2 };
                        herdRetrace(ro2, od, mAcr, selfM, &tenv);
                    }

                    v4 att = vdiv(S(1.f), vadd(S(1.f), vmul(s, S(1.6f))));  // beer-ish falloff
                    v4 kt = vmul(tranP, vsub(S(1.f), vmul(f5, S(0.6f))));   // less at grazing
                    cr = sel(mixv(cr, vmul(tenv.r, mixv(alb.r, S(1.f), att)), kt), cr, mAcr);
                    cg = sel(mixv(cg, vmul(tenv.g, mixv(alb.g, S(1.f), att)), kt), cg, mAcr);
                    cb = sel(mixv(cb, vmul(tenv.b, mixv(alb.b, S(1.f), att)), kt), cb, mAcr);
                }

                // dielectric fresnel blend, reflectivity slider (stock at 0.5); metal opts out
                v4 F = vmul(vadd(S(0.030f), vmul(S(0.55f), f5)), vmul(reflP, S(2.f)));
                // bone/horn parts a bit glossier
                v4 gl = sel(S(1.5f), S(1.f), vgt(alb.r, vadd(alb.g, S(0.25f))));
                F = vmul(vmul(F, gl), vadd(S(0.5f), vmul(S(0.5f), sh)));
                F = sel(S(0.f), F, mMet);
                cr = sel(mixv(cr, env.r, F), cr, hit);
                cg = sel(mixv(cg, env.g, F), cg, hit);
                cb = sel(mixv(cb, env.b, F), cb, hit);

                // metal: albedo-tinted mirror, diffuse crushed by reflectivity
                if (any(mMet)){
                    // mirror the herd: march the reflected ray against the other
                    // instances so mirror bodies reflect the surroundings, not
                    // just the analytic sky/ground
                    {
                        V3 ro2 = v3(vadd(P.x, vmul(rr.x, S(0.04f))),
                                    vadd(P.y, vmul(rr.y, S(0.04f))),
                                    vadd(P.z, vmul(rr.z, S(0.04f))));
                        v4 selfM[3] = { m0, m1, m2 };
                        herdRetrace(ro2, rr, mMet, selfM, &env);
                    }
                    v4 Fm = vmul(reflP, vadd(S(0.62f), vmul(S(0.38f), f5)));
                    Fm = vmul(Fm, vadd(S(0.6f), vmul(S(0.4f), sh)));
                    v4 tR = vmin(vadd(vmul(alb.r, S(1.9f)), S(0.08f)), S(1.f));
                    v4 tG = vmin(vadd(vmul(alb.g, S(1.9f)), S(0.08f)), S(1.f));
                    v4 tB = vmin(vadd(vmul(alb.b, S(1.9f)), S(0.08f)), S(1.f));
                    v4 dk = vsub(S(1.f), vmul(reflP, S(0.72f)));
                    cr = sel(mixv(vmul(cr,dk), vmul(env.r,tR), Fm), cr, mMet);
                    cg = sel(mixv(vmul(cg,dk), vmul(env.g,tG), Fm), cg, mMet);
                    cb = sel(mixv(vmul(cb,dk), vmul(env.b,tB), Fm), cb, mMet);
                }
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
                        dinoMasks(RP, rhit, LR, nr, &rm0, &rm1, &rm2);
                        C3 ra = dinoAlbedo(RP, RN, rm0, rm1, rm2);
                        v4 rdif = vmax(v3dot(RN, Lv), S(0.f));
                        v4 lr_ = vadd(vmul(ra.r, vadd(vmul(rdif,S(1.1f)), S(0.34f))), S(0.f));
                        v4 lg_ = vadd(vmul(ra.g, vadd(vmul(rdif,S(1.05f)), S(0.38f))), S(0.f));
                        v4 lb_ = vadd(vmul(ra.b, vadd(vmul(rdif,S(0.95f)), S(0.46f))), S(0.f));
                        // material hints in the mirror image
                        v4 rMet = modeMask(2, rm0, rm1, rm2);
                        v4 rAcr = modeMask(3, rm0, rm1, rm2);
                        if (any(rMet)){   // tinted sky mirror
                            v4 rrefl = dsel(M_REFL, rm0, rm1);
                            v4 dn2 = v3dot(RN, rrd);
                            V3 rr2 = v3(vsub(rrd.x, vmul(RN.x, vmul(S(2.f),dn2))),
                                        vsub(rrd.y, vmul(RN.y, vmul(S(2.f),dn2))),
                                        vsub(rrd.z, vmul(RN.z, vmul(S(2.f),dn2))));
                            C3 e2c = skyCol(rr2, 0);
                            v4 tR = vmin(vadd(vmul(ra.r, S(1.9f)), S(0.08f)), S(1.f));
                            v4 tG = vmin(vadd(vmul(ra.g, S(1.9f)), S(0.08f)), S(1.f));
                            v4 tB = vmin(vadd(vmul(ra.b, S(1.9f)), S(0.08f)), S(1.f));
                            lr_ = sel(mixv(lr_, vmul(e2c.r,tR), rrefl), lr_, rMet);
                            lg_ = sel(mixv(lg_, vmul(e2c.g,tG), rrefl), lg_, rMet);
                            lb_ = sel(mixv(lb_, vmul(e2c.b,tB), rrefl), lb_, rMet);
                        }
                        if (any(rAcr)){   // let the sky behind show through
                            v4 rtran = dsel(M_TRAN, rm0, rm1);
                            lr_ = sel(mixv(lr_, rc.r, rtran), lr_, rAcr);
                            lg_ = sel(mixv(lg_, rc.g, rtran), lg_, rAcr);
                            lb_ = sel(mixv(lb_, rc.b, rtran), lb_, rAcr);
                        }
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

void renderFrame(float az, float el, float dist, int w, int h, unsigned char *fb){
    renderRows(az, el, dist, w, h, fb, 0, h);
}
