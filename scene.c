// scene.c — combined SDF + mesh renderer. See scene.h for the design intent.
//
// This is structurally render.c's renderRows with three additions woven in:
//  1. the primary visibility test also runs mesh.c's 4-ray-packet BVH trace
//     (meshTraceP), bounded by the SDF march's own nearest hit so a mesh
//     triangle only wins when it's strictly closer;
//  2. wherever a mesh triangle wins, its normal/albedo/material are gathered
//     per-lane (mesh geometry is inherently per-triangle, so this part is
//     scalar loads assembled back into SIMD lanes — the same pattern
//     mesh.c's own renderMeshRows already uses for its scalar shading step)
//     and merged into the same v4 albedo/normal/material used downstream by
//     lighting, shadows, and the fresnel/metal blend;
//  3. sun visibility and the floor-mirror reflection both re-run against the
//     combined scene (SDF march + mesh trace), so the two dinosaur kinds
//     shadow and mirror each other instead of only their own kind.
//
// Acrylic finds its back face per geometry kind — an SDF acrylic hit
// marches the implicit surface's interior, a mesh acrylic hit bends the
// ray by Snell's law at the entry face and BVH-traces to the far side —
// then both retrace the combined scene from the exit point (sceneRetrace),
// so the models behind show through either kind of glass body. Both blend
// into the same cr/cg/cb output through the same transmittance formula.
#include "vec.h"
#include "render.h"
#include "mesh.h"
#include "anim.h"
#include "dino_model.h"
#include "scene.h"

void scenePrep(float t){
    animTick(t);
    animate(t);
    meshPrep();
}

// ---------- secondary-ray combined-scene retrace ----------
// Trace a secondary ray (acrylic exit ray, metal mirror ray) against the
// combined scene — the SDF herd minus each lane's own instance, plus the
// whole mesh line-up — and where it hits replace *env with the shaded
// surface, so glass bodies show the dinosaurs behind them and mirror
// bodies reflect the ones around them, whichever kind they are.
static void sceneRetrace(V3 ro2, V3 od, v4 mask, const v4 selfM[3], C3 *env){
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
    v4 tt = tt0, thit = wasm_i32x4_splat(0);
    unsigned char LT[MAXP]; int nt=0;
    if (any(tvm)){
        for (int i=0;i<ND;i++) if (tdh[i])
            nt = buildList(ro2, od, tt0, tt1, 0.03f, DPR[i][0], DPR[i][1], LT, nt);
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
    }
    // mesh candidates along the same secondary ray, bounded by the SDF
    // retrace hit (or a generous fallback)
    v4 tBoundMesh2 = sel(tt, S(30.f), thit);
    v4 tMesh2; int meshId2[4];
    meshTraceP(ro2, od, tBoundMesh2, &tMesh2, meshId2);
    int mesh2Ok[4]; for(int l=0;l<4;l++) mesh2Ok[l]=meshId2[l]>=0?-1:0;
    v4 useMesh2 = vand(wasm_v128_load(mesh2Ok), mask);
    v4 useSdfRe = vandn(thit, useMesh2);
    v4 anyRetraceHit = vor(useSdfRe, useMesh2);
    v4 ttFinal = sel(tt, tMesh2, useSdfRe);

    if (!any(anyRetraceHit)) return;
    V3 TP = v3(vadd(ro2.x, vmul(od.x,ttFinal)),
               vadd(ro2.y, vmul(od.y,ttFinal)),
               vadd(ro2.z, vmul(od.z,ttFinal)));
    V3 TN = v3(S(0.f),S(1.f),S(0.f));
    C3 ta = {S(0.f),S(0.f),S(0.f)};
    v4 tRefl=S(0.5f), tTran=S(0.65f);
    v4 tMetM=wasm_i32x4_splat(0), tAcrM=wasm_i32x4_splat(0);
    if (any(useSdfRe)){
        const float e3 = 0.004f;
        v4 h1 = mapLE(v3(vadd(TP.x,S(e3)), vsub(TP.y,S(e3)), vsub(TP.z,S(e3))), LT, nt);
        v4 h2 = mapLE(v3(vsub(TP.x,S(e3)), vsub(TP.y,S(e3)), vadd(TP.z,S(e3))), LT, nt);
        v4 h3 = mapLE(v3(vsub(TP.x,S(e3)), vadd(TP.y,S(e3)), vsub(TP.z,S(e3))), LT, nt);
        v4 h4 = mapLE(v3(vadd(TP.x,S(e3)), vadd(TP.y,S(e3)), vadd(TP.z,S(e3))), LT, nt);
        V3 TNs = v3norm(v3(
            vadd(vsub(vsub(h1,h2),h3), h4),
            vadd(vsub(vsub(h4,h1),h2), h3),
            vadd(vsub(vsub(h2,h1),h3), h4)));
        TN = v3(sel(TNs.x,TN.x,useSdfRe), sel(TNs.y,TN.y,useSdfRe), sel(TNs.z,TN.z,useSdfRe));
        v4 tm0,tm1,tm2;
        dinoMasks(TP, useSdfRe, LT, nt, &tm0, &tm1, &tm2);
        C3 ta_ = dinoAlbedo(TP, TN, tm0, tm1, tm2);
        ta.r = sel(ta_.r, ta.r, useSdfRe);
        ta.g = sel(ta_.g, ta.g, useSdfRe);
        ta.b = sel(ta_.b, ta.b, useSdfRe);
        tRefl = sel(dsel(M_REFL,tm0,tm1), tRefl, useSdfRe);
        tTran = sel(dsel(M_TRAN,tm0,tm1), tTran, useSdfRe);
        tMetM = vor(tMetM, vand(modeMask(2,tm0,tm1,tm2), useSdfRe));
        tAcrM = vor(tAcrM, vand(modeMask(3,tm0,tm1,tm2), useSdfRe));
    }
    if (any(useMesh2)){
        float TPxA[4],TPyA[4],TPzA[4], odxA[4],odyA[4],odzA[4];
        wasm_v128_store(TPxA,TP.x); wasm_v128_store(TPyA,TP.y); wasm_v128_store(TPzA,TP.z);
        wasm_v128_store(odxA,od.x); wasm_v128_store(odyA,od.y); wasm_v128_store(odzA,od.z);
        int useA[4]; wasm_v128_store(useA, useMesh2);
        float tnxA[4],tnyA[4],tnzA[4], tarA[4],tagA[4],tabA[4];
        float trA[4],ttrA[4];
        int tMetA[4], tAcrA[4];
        for (int l=0;l<4;l++){
            if (!useA[l]){
                tnxA[l]=tnyA[l]=tnzA[l]=0.f; tarA[l]=tagA[l]=tabA[l]=0.f;
                trA[l]=ttrA[l]=0.f; tMetA[l]=tAcrA[l]=0; continue;
            }
            float Pl[3]={TPxA[l],TPyA[l],TPzA[l]}, rdl[3]={odxA[l],odyA[l],odzA[l]};
            float Nl[3], al[3]; int ent, mode; float refl,tran,ior,gloss;
            meshSurface(meshId2[l], Pl, rdl, Nl, al, &ent, &mode, &refl, &tran, &ior, &gloss);
            tnxA[l]=Nl[0]; tnyA[l]=Nl[1]; tnzA[l]=Nl[2];
            tarA[l]=al[0]; tagA[l]=al[1]; tabA[l]=al[2];
            trA[l]=refl; ttrA[l]=tran;
            tMetA[l]=(mode==2)?-1:0; tAcrA[l]=(mode==3)?-1:0;
        }
        V3 TNmesh = v3(wasm_v128_load(tnxA), wasm_v128_load(tnyA), wasm_v128_load(tnzA));
        C3 taMesh = { wasm_v128_load(tarA), wasm_v128_load(tagA), wasm_v128_load(tabA) };
        TN = v3(sel(TNmesh.x,TN.x,useMesh2), sel(TNmesh.y,TN.y,useMesh2), sel(TNmesh.z,TN.z,useMesh2));
        ta.r = sel(taMesh.r, ta.r, useMesh2);
        ta.g = sel(taMesh.g, ta.g, useMesh2);
        ta.b = sel(taMesh.b, ta.b, useMesh2);
        tRefl = sel(wasm_v128_load(trA),  tRefl, useMesh2);
        tTran = sel(wasm_v128_load(ttrA), tTran, useMesh2);
        tMetM = vor(tMetM, vand(wasm_v128_load(tMetA), useMesh2));
        tAcrM = vor(tAcrM, vand(wasm_v128_load(tAcrA), useMesh2));
    }

    v4 tdif = vmax(v3dot(TN, Lv), S(0.f));
    v4 tr_ = vmul(ta.r, vadd(vmul(tdif,S(1.1f)), S(0.34f)));
    v4 tg_ = vmul(ta.g, vadd(vmul(tdif,S(1.05f)), S(0.38f)));
    v4 tb_ = vmul(ta.b, vadd(vmul(tdif,S(0.95f)), S(0.46f)));
    if (any(tMetM)){
        v4 dn3 = v3dot(TN, od);
        V3 rr3 = v3(vsub(od.x, vmul(TN.x, vmul(S(2.f),dn3))),
                    vsub(od.y, vmul(TN.y, vmul(S(2.f),dn3))),
                    vsub(od.z, vmul(TN.z, vmul(S(2.f),dn3))));
        C3 e3c = skyCol(rr3, 0);
        v4 mR = vmin(vadd(vmul(ta.r, S(1.9f)), S(0.08f)), S(1.f));
        v4 mG = vmin(vadd(vmul(ta.g, S(1.9f)), S(0.08f)), S(1.f));
        v4 mB = vmin(vadd(vmul(ta.b, S(1.9f)), S(0.08f)), S(1.f));
        tr_ = sel(mixv(tr_, vmul(e3c.r,mR), tRefl), tr_, tMetM);
        tg_ = sel(mixv(tg_, vmul(e3c.g,mG), tRefl), tg_, tMetM);
        tb_ = sel(mixv(tb_, vmul(e3c.b,mB), tRefl), tb_, tMetM);
    }
    if (any(tAcrM)){
        C3 skyBeyond = skyCol(od, 0);
        tr_ = sel(mixv(tr_, skyBeyond.r, tTran), tr_, tAcrM);
        tg_ = sel(mixv(tg_, skyBeyond.g, tTran), tg_, tAcrM);
        tb_ = sel(mixv(tb_, skyBeyond.b, tTran), tb_, tAcrM);
    }
    env->r = sel(tr_, env->r, anyRetraceHit);
    env->g = sel(tg_, env->g, anyRetraceHit);
    env->b = sel(tb_, env->b, anyRetraceHit);
}

void sceneRows(float az, float el, float dist, int w, int h, unsigned char *fb, int y0, int y1){
    const float tx=FOCX+CAM_TARGET_X, ty=CAM_TARGET_Y, tz=FOCZ;
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
    V3 camO = v3(S(cxx), S(cyy), S(czz));

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

            // ---- SDF instance bound spheres (unchanged from render.c).
            // Placement is decided upstream (dino_model.c's animate() only
            // registers a species' primitives when it's placed in the
            // scene), so this loop naturally finds nothing for an unplaced
            // species — no visibility flag needed here. ----
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
            v4 sdfHit = wasm_i32x4_splat(0);
            unsigned char LP[MAXP];
            int np_ = 0;
            if (any(doM)){
                for (int i=0;i<ND;i++) if (dh[i])
                    np_ = buildList(camO, rd, t0, tEnd, 0.03f, DPR[i][0], DPR[i][1], LP, np_);
                v4 act = np_ ? doM : wasm_i32x4_splat(0);
                for (int i=0;i<64;i++){
                    V3 p = v3(vadd(S(cxx), vmul(rd.x,tD)),
                              vadd(S(cyy), vmul(rd.y,tD)),
                              vadd(S(czz), vmul(rd.z,tD)));
                    v4 d = mapLE(p, LP, np_);
                    v4 nh = vand(act, vlt(d, vadd(S(0.0012f), vmul(tD, S(0.0006f)))));
                    sdfHit = vor(sdfHit, nh);
                    act = vandn(act, nh);
                    tD = sel(vadd(tD, vmul(d, S(0.92f))), tD, act);
                    act = vand(act, vlt(tD, tEnd));
                    if (!any(act)) break;
                }
            }

            // ---- mesh line-up: 4-ray-packet BVH trace, bounded by the SDF
            // hit (or the ground plane) so it only wins when strictly
            // nearer than what the SDF march already found. When the mesh
            // line-up isn't placed (JS uploaded zero triangles), the BVH is
            // empty and this naturally finds nothing. ----
            v4 tPreMesh = sel(tD, tg, sdfHit);
            v4 tMeshHit; int meshHitId[4];
            meshTraceP(camO, rd, tPreMesh, &tMeshHit, meshHitId);
            int meshOkArr[4];
            for (int l=0;l<4;l++) meshOkArr[l] = meshHitId[l]>=0 ? -1 : 0;
            v4 useMesh = wasm_v128_load(meshOkArr);
            v4 useSDF  = vandn(sdfHit, useMesh);
            v4 hitFinal = vor(useSDF, useMesh);
            v4 tFinal = sel(tD, tMeshHit, useSDF);

            v4 gm = vandn(vand(down, vlt(tg, S(1e8f))), hitFinal);
            v4 hm = vor(hitFinal, gm);
            v4 tH = sel(tFinal, tg, hitFinal);
            V3 P = v3(vadd(S(cxx), vmul(rd.x,tH)),
                      vadd(S(cyy), vmul(rd.y,tH)),
                      vadd(S(czz), vmul(rd.z,tH)));

            // ---- normal: SDF gradient where useSDF, filled in by the mesh
            // gather below where useMesh ----
            V3 N = v3(S(0.f), S(1.f), S(0.f));
            v4 ao = S(1.f);
            if (any(useSDF)){
                const float e = 0.0024f;
                v4 d1 = mapLE(v3(vadd(P.x,S(e)), vsub(P.y,S(e)), vsub(P.z,S(e))), LP, np_);
                v4 d2 = mapLE(v3(vsub(P.x,S(e)), vsub(P.y,S(e)), vadd(P.z,S(e))), LP, np_);
                v4 d3 = mapLE(v3(vsub(P.x,S(e)), vadd(P.y,S(e)), vsub(P.z,S(e))), LP, np_);
                v4 d4 = mapLE(v3(vadd(P.x,S(e)), vadd(P.y,S(e)), vadd(P.z,S(e))), LP, np_);
                V3 nn = v3norm(v3(
                    vadd(vsub(vsub(d1,d2),d3), d4),
                    vadd(vsub(vsub(d4,d1),d2), d3),
                    vadd(vsub(vsub(d2,d1),d3), d4)));
                N = v3(sel(nn.x,N.x,useSDF), sel(nn.y,N.y,useSDF), sel(nn.z,N.z,useSDF));
                v4 da = mapLE(v3add(P, v3scale(N, S(0.12f))), LP, np_);
                ao = sel(vadd(S(0.45f), vmul(S(0.55f), clamp01(vmul(da, S(1.f/0.12f))))), ao, useSDF);
            }

            // ---- albedo + material: unified regardless of geometry source ----
            v4 m0=wasm_i32x4_splat(0), m1=m0, m2=m0;
            C3 alb;
            v4 reflP=S(0.5f), tranP=S(0.65f), iorP=S(1.49f), glossP=S(0.5f);
            v4 mMet=wasm_i32x4_splat(0), mAcrSDF=wasm_i32x4_splat(0), mAcrMesh=wasm_i32x4_splat(0);
            int entA[4] = {0,0,0,0};   // acrylic "entering the body" flag, mesh lanes only
            {
                C3 ga = groundAlbedo(P);
                alb = ga;
                if (any(useSDF)){
                    dinoMasks(P, useSDF, LP, np_, &m0, &m1, &m2);
                    C3 da_ = dinoAlbedo(P, N, m0, m1, m2);
                    alb.r = sel(da_.r, alb.r, useSDF);
                    alb.g = sel(da_.g, alb.g, useSDF);
                    alb.b = sel(da_.b, alb.b, useSDF);
                    reflP  = sel(dsel(M_REFL, m0, m1),  reflP,  useSDF);
                    tranP  = sel(dsel(M_TRAN, m0, m1),  tranP,  useSDF);
                    iorP   = sel(dsel(M_IOR,  m0, m1),  iorP,   useSDF);
                    glossP = sel(dsel(M_GLOSS,m0, m1),  glossP, useSDF);
                    mMet    = vor(mMet,    vand(modeMask(2,m0,m1,m2), useSDF));
                    mAcrSDF = vor(mAcrSDF, vand(modeMask(3,m0,m1,m2), useSDF));
                }
                if (any(useMesh)){
                    float PxA[4],PyA[4],PzA[4], rdxA[4],rdyA[4],rdzA[4];
                    wasm_v128_store(PxA,P.x); wasm_v128_store(PyA,P.y); wasm_v128_store(PzA,P.z);
                    wasm_v128_store(rdxA,rd.x); wasm_v128_store(rdyA,rd.y); wasm_v128_store(rdzA,rd.z);
                    float nxA[4],nyA[4],nzA[4], arA[4],agA[4],abA[4];
                    float reflA[4],tranA[4],iorA[4],glossA[4];
                    int mMetA[4], mAcrA[4];
                    for (int l=0;l<4;l++){
                        if (meshHitId[l] < 0){
                            nxA[l]=nyA[l]=nzA[l]=0.f; arA[l]=agA[l]=abA[l]=0.f;
                            reflA[l]=tranA[l]=iorA[l]=glossA[l]=0.f;
                            mMetA[l]=mAcrA[l]=0;
                            continue;
                        }
                        float Pl[3]={PxA[l],PyA[l],PzA[l]}, rdl[3]={rdxA[l],rdyA[l],rdzA[l]};
                        float Nl[3], al[3]; int ent, mode; float refl,tran,ior,gloss;
                        meshSurface(meshHitId[l], Pl, rdl, Nl, al, &ent, &mode, &refl, &tran, &ior, &gloss);
                        nxA[l]=Nl[0]; nyA[l]=Nl[1]; nzA[l]=Nl[2];
                        arA[l]=al[0]; agA[l]=al[1]; abA[l]=al[2];
                        reflA[l]=refl; tranA[l]=tran; iorA[l]=ior; glossA[l]=gloss;
                        mMetA[l]=(mode==2)?-1:0; mAcrA[l]=(mode==3)?-1:0;
                        entA[l]=ent?-1:0;
                    }
                    V3 Nmesh = v3(wasm_v128_load(nxA), wasm_v128_load(nyA), wasm_v128_load(nzA));
                    C3 albMesh = { wasm_v128_load(arA), wasm_v128_load(agA), wasm_v128_load(abA) };
                    N = v3(sel(Nmesh.x,N.x,useMesh), sel(Nmesh.y,N.y,useMesh), sel(Nmesh.z,N.z,useMesh));
                    alb.r = sel(albMesh.r, alb.r, useMesh);
                    alb.g = sel(albMesh.g, alb.g, useMesh);
                    alb.b = sel(albMesh.b, alb.b, useMesh);
                    reflP  = sel(wasm_v128_load(reflA),  reflP,  useMesh);
                    tranP  = sel(wasm_v128_load(tranA),  tranP,  useMesh);
                    iorP   = sel(wasm_v128_load(iorA),   iorP,   useMesh);
                    glossP = sel(wasm_v128_load(glossA), glossP, useMesh);
                    mMet     = vor(mMet,     vand(wasm_v128_load(mMetA), useMesh));
                    mAcrMesh = vor(mAcrMesh, vand(wasm_v128_load(mAcrA), useMesh));
                }
            }

            // ---- lighting ----
            v4 ndl = vmax(v3dot(N, Lv), S(0.f));
            v4 sh = S(1.f);
            v4 lit = vand(hm, vgt(ndl, S(0.02f)));
            if (any(lit)){
                V3 sp_ = v3(vadd(P.x, vmul(N.x, S(0.012f))),
                            vadd(P.y, vmul(N.y, S(0.012f))),
                            vadd(P.z, vmul(N.z, S(0.012f))));
                sh = softshadow(sp_, lit);
                // mesh occlusion: any mesh triangle between the shading point
                // and the sun dims it too (same 0.25 residual mesh.c's own
                // scalar shadow uses) — this is what lets mesh dinosaurs
                // shadow SDF dinosaurs, the ground, and each other. When the
                // mesh line-up isn't placed, meshOccluded() always reports
                // no occlusion (empty BVH), so this is a no-op.
                float spxA[4],spyA[4],spzA[4]; int litA[4];
                wasm_v128_store(spxA, sp_.x); wasm_v128_store(spyA, sp_.y); wasm_v128_store(spzA, sp_.z);
                wasm_v128_store(litA, lit);
                float meshShA[4] = {1.f,1.f,1.f,1.f};
                const float sund[3] = {SUNX, SUNY, SUNZ};
                for (int l=0;l<4;l++){
                    if (!litA[l]) continue;
                    float so[3] = {spxA[l], spyA[l], spzA[l]};
                    if (meshOccluded(so, sund, 20.f)) meshShA[l] = 0.25f;
                }
                sh = vmul(sh, wasm_v128_load(meshShA));
            }
            v4 dif = vmul(ndl, sh);
            V3 Hv = v3norm(v3(vsub(S(SUNX), rd.x), vsub(S(SUNY), rd.y), vsub(S(SUNZ), rd.z)));
            v4 sp = vmax(v3dot(N, Hv), S(0.f));
            sp = vmul(sp,sp); sp = vmul(sp,sp); sp = vmul(sp,sp); sp = vmul(sp,sp);
            v4 em = vlt(eyeDistAll(P), S(0.006f));
            v4 gv = vmul(glossP, S(0.7f));
            gv = sel(vmul(gv, S(3.4f)), gv, mMet);
            v4 spc = sel(S(1.4f), sel(gv, S(0.05f), hitFinal), em);
            sp = vmul(vmul(sp, spc), sh);

            v4 amb = vmul(vadd(S(0.55f), vmul(S(0.45f), N.y)), ao);
            v4 cr = vadd(vmul(alb.r, vadd(vmul(dif, S(1.30f)), vmul(amb, S(0.42f)))), sp);
            v4 cg = vadd(vmul(alb.g, vadd(vmul(dif, S(1.22f)), vmul(amb, S(0.50f)))), sp);
            v4 cb = vadd(vmul(alb.b, vadd(vmul(dif, S(1.05f)), vmul(amb, S(0.66f)))), sp);

            // ---- surface reflection: fresnel env blend, metal mirror,
            // acrylic transmission (shared math; analytic sky/ground unless
            // the acrylic sub-blocks below retrace the scene) ----
            if (any(hitFinal)){
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

                // acrylic (SDF-sourced): march the SDF interior to the back
                // face, refract out, then retrace the exit ray against the
                // rest of the combined scene (SDF herd minus self + the
                // whole mesh line-up) so glass SDF dinos show other dinos
                // of either kind behind them.
                if (any(mAcrSDF)){
                    v4 eta = vdiv(S(1.f), iorP);
                    v4 kk = vsub(S(1.f), vmul(vmul(eta,eta), vsub(S(1.f), vmul(ci,ci))));
                    v4 tcf = vsub(vmul(eta,ci), vsqrt(vmax(kk, S(0.f))));
                    V3 td = v3(vadd(vmul(rd.x,eta), vmul(N.x,tcf)),
                               vadd(vmul(rd.y,eta), vmul(N.y,tcf)),
                               vadd(vmul(rd.z,eta), vmul(N.z,tcf)));
                    v4 s = S(0.03f);
                    v4 live = mAcrSDF;
                    for (int it=0; it<18; it++){
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
                    v4 tir = vlt(k2, S(0.f));
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

                    {
                        V3 ro2 = v3(vadd(XP.x, vmul(od.x, S(0.04f))),
                                    vadd(XP.y, vmul(od.y, S(0.04f))),
                                    vadd(XP.z, vmul(od.z, S(0.04f))));
                        v4 selfM[3] = { m0, m1, m2 };
                        sceneRetrace(ro2, od, mAcrSDF, selfM, &tenv);
                    }

                    v4 att = vdiv(S(1.f), vadd(S(1.f), vmul(s, S(1.6f))));
                    v4 kt = vmul(tranP, vsub(S(1.f), vmul(f5, S(0.6f))));
                    cr = sel(mixv(cr, vmul(tenv.r, mixv(alb.r, S(1.f), att)), kt), cr, mAcrSDF);
                    cg = sel(mixv(cg, vmul(tenv.g, mixv(alb.g, S(1.f), att)), kt), cg, mAcrSDF);
                    cb = sel(mixv(cb, vmul(tenv.b, mixv(alb.b, S(1.f), att)), kt), cb, mAcrSDF);
                }

                // acrylic (mesh-sourced): Snell-bend at the entry face (skip
                // the bend when exiting, same convention as mesh.c's own
                // acrylic), trace through the body to its back face, then
                // retrace the combined scene from the exit point so the
                // models behind show through instead of just sky/ground.
                if (any(mAcrMesh)){
                    v4 eta = vdiv(S(1.f), iorP);
                    v4 kk = vsub(S(1.f), vmul(vmul(eta,eta), vsub(S(1.f), vmul(ci,ci))));
                    v4 tcf = vsub(vmul(eta,ci), vsqrt(vmax(kk, S(0.f))));
                    V3 bentDir = v3(vadd(vmul(rd.x,eta), vmul(N.x,tcf)),
                                     vadd(vmul(rd.y,eta), vmul(N.y,tcf)),
                                     vadd(vmul(rd.z,eta), vmul(N.z,tcf)));
                    v4 enterMask = wasm_v128_load(entA);
                    V3 od2 = v3norm(v3(sel(bentDir.x, rd.x, enterMask),
                                        sel(bentDir.y, rd.y, enterMask),
                                        sel(bentDir.z, rd.z, enterMask)));
                    // exit point: nearest mesh surface along the interior ray
                    // (entering lanes only — when leaving the body the ray is
                    // already outside, so continue straight from P)
                    V3 ri = v3(vadd(P.x, vmul(od2.x, S(0.02f))),
                               vadd(P.y, vmul(od2.y, S(0.02f))),
                               vadd(P.z, vmul(od2.z, S(0.02f))));
                    v4 tExit; int exitId[4];
                    meshTraceP(ri, od2, S(30.f), &tExit, exitId);
                    int exOk[4]; for (int l=0;l<4;l++) exOk[l]=exitId[l]>=0?-1:0;
                    v4 exHit = vand(vand(wasm_v128_load(exOk), mAcrMesh), enterMask);
                    v4 tX = sel(tExit, S(0.f), exHit);
                    V3 XP2 = v3(vadd(ri.x, vmul(od2.x,tX)),
                                vadd(ri.y, vmul(od2.y,tX)),
                                vadd(ri.z, vmul(od2.z,tX)));
                    C3 tenv2 = skyCol(od2, 0);
                    v4 odn2 = vlt(od2.y, S(-1e-3f));
                    if (any(odn2)){
                        v4 tg3 = vdiv(vneg(XP2.y), vmin(od2.y, S(-1e-3f)));
                        V3 GP3 = v3(vadd(XP2.x, vmul(od2.x,tg3)), S(0.f), vadd(XP2.z, vmul(od2.z,tg3)));
                        C3 gc3 = groundAlbedo(GP3);
                        v4 fd3 = clamp01(vmul(tg3, S(0.18f)));
                        tenv2.r = sel(mixv(vmul(gc3.r,S(0.9f)), tenv2.r, fd3), tenv2.r, odn2);
                        tenv2.g = sel(mixv(vmul(gc3.g,S(0.9f)), tenv2.g, fd3), tenv2.g, odn2);
                        tenv2.b = sel(mixv(vmul(gc3.b,S(0.9f)), tenv2.b, fd3), tenv2.b, odn2);
                    }
                    // pass straight through the back face and retrace the
                    // combined scene behind the body (SDF herd + mesh line-up)
                    {
                        V3 ro3 = v3(vadd(XP2.x, vmul(od2.x, S(0.04f))),
                                    vadd(XP2.y, vmul(od2.y, S(0.04f))),
                                    vadd(XP2.z, vmul(od2.z, S(0.04f))));
                        v4 z = wasm_i32x4_splat(0);
                        v4 selfNone[3] = { z, z, z };
                        sceneRetrace(ro3, od2, mAcrMesh, selfNone, &tenv2);
                    }
                    v4 kt2 = vmul(tranP, vsub(S(1.f), vmul(f5, S(0.5f))));
                    cr = sel(mixv(cr, vmul(tenv2.r, vadd(S(0.5f), vmul(S(0.5f),alb.r))), kt2), cr, mAcrMesh);
                    cg = sel(mixv(cg, vmul(tenv2.g, vadd(S(0.5f), vmul(S(0.5f),alb.g))), kt2), cg, mAcrMesh);
                    cb = sel(mixv(cb, vmul(tenv2.b, vadd(S(0.5f), vmul(S(0.5f),alb.b))), kt2), cb, mAcrMesh);
                }

                v4 F = vmul(vadd(S(0.030f), vmul(S(0.55f), f5)), vmul(reflP, S(2.f)));
                v4 gl = sel(S(1.5f), S(1.f), vgt(alb.r, vadd(alb.g, S(0.25f))));
                F = vmul(vmul(F, gl), vadd(S(0.5f), vmul(S(0.5f), sh)));
                F = sel(S(0.f), F, mMet);
                cr = sel(mixv(cr, env.r, F), cr, hitFinal);
                cg = sel(mixv(cg, env.g, F), cg, hitFinal);
                cb = sel(mixv(cb, env.b, F), cb, hitFinal);

                if (any(mMet)){
                    // mirror the combined scene: trace the reflected ray against
                    // the SDF herd (minus self) and the mesh line-up so mirror
                    // bodies reflect the surroundings, not just sky/ground
                    {
                        V3 ro2 = v3(vadd(P.x, vmul(rr.x, S(0.04f))),
                                    vadd(P.y, vmul(rr.y, S(0.04f))),
                                    vadd(P.z, vmul(rr.z, S(0.04f))));
                        v4 selfM[3] = { m0, m1, m2 };
                        sceneRetrace(ro2, rr, mMet, selfM, &env);
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

            // ---- reflective floor: mirror-trace the combined scene (SDF
            // march + mesh BVH) so the floor mirrors both dinosaur kinds ----
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
                C3 rc = skyCol(rrd, 1);
                unsigned char LR[MAXP]; int nr=0;
                v4 rt = rt0, rSdfHit = wasm_i32x4_splat(0);
                if (any(rvm)){
                    for (int i=0;i<ND;i++) if (rdh[i])
                        nr = buildList(rro, rrd, rt0, rt1, 0.04f, DPR[i][0], DPR[i][1], LR, nr);
                    if (nr){
                        v4 act = rvm;
                        for (int i=0;i<30;i++){
                            V3 q = v3(vadd(rro.x, vmul(rrd.x,rt)),
                                      vadd(rro.y, vmul(rrd.y,rt)),
                                      vadd(rro.z, vmul(rrd.z,rt)));
                            v4 d = mapLE(q, LR, nr);
                            v4 nh2 = vand(act, vlt(d, vadd(S(0.0025f), vmul(rt, S(0.0012f)))));
                            rSdfHit = vor(rSdfHit, nh2);
                            act = vandn(act, nh2);
                            rt = sel(vadd(rt, vmul(d, S(0.92f))), rt, act);
                            act = vand(act, vlt(rt, rt1));
                            if (!any(act)) break;
                        }
                    }
                }
                v4 rMeshBound = sel(rt, S(30.f), rSdfHit);
                v4 rMeshT; int rMeshId[4];
                meshTraceP(rro, rrd, rMeshBound, &rMeshT, rMeshId);
                int rMeshOk[4]; for(int l=0;l<4;l++) rMeshOk[l]=rMeshId[l]>=0?-1:0;
                v4 rUseMesh = wasm_v128_load(rMeshOk);
                v4 rUseSDF  = vandn(rSdfHit, rUseMesh);
                v4 rHit = vor(rUseSDF, rUseMesh);
                v4 rtFinal = sel(rt, rMeshT, rUseSDF);

                if (any(rHit)){
                    V3 RP = v3(vadd(rro.x, vmul(rrd.x,rtFinal)),
                               vadd(rro.y, vmul(rrd.y,rtFinal)),
                               vadd(rro.z, vmul(rrd.z,rtFinal)));
                    V3 RN = v3(S(0.f),S(1.f),S(0.f));
                    C3 ra = {S(0.f),S(0.f),S(0.f)};
                    v4 rRefl=S(0.5f), rTran=S(0.65f);
                    v4 rMetM=wasm_i32x4_splat(0), rAcrM=wasm_i32x4_splat(0);
                    if (any(rUseSDF)){
                        const float e = 0.004f;
                        v4 d1 = mapLE(v3(vadd(RP.x,S(e)), vsub(RP.y,S(e)), vsub(RP.z,S(e))), LR, nr);
                        v4 d2 = mapLE(v3(vsub(RP.x,S(e)), vsub(RP.y,S(e)), vadd(RP.z,S(e))), LR, nr);
                        v4 d3 = mapLE(v3(vsub(RP.x,S(e)), vadd(RP.y,S(e)), vsub(RP.z,S(e))), LR, nr);
                        v4 d4 = mapLE(v3(vadd(RP.x,S(e)), vadd(RP.y,S(e)), vadd(RP.z,S(e))), LR, nr);
                        V3 RNs = v3norm(v3(
                            vadd(vsub(vsub(d1,d2),d3), d4),
                            vadd(vsub(vsub(d4,d1),d2), d3),
                            vadd(vsub(vsub(d2,d1),d3), d4)));
                        RN = v3(sel(RNs.x,RN.x,rUseSDF), sel(RNs.y,RN.y,rUseSDF), sel(RNs.z,RN.z,rUseSDF));
                        v4 rm0,rm1,rm2;
                        dinoMasks(RP, rUseSDF, LR, nr, &rm0, &rm1, &rm2);
                        C3 ra_ = dinoAlbedo(RP, RN, rm0, rm1, rm2);
                        ra.r = sel(ra_.r, ra.r, rUseSDF); ra.g = sel(ra_.g, ra.g, rUseSDF); ra.b = sel(ra_.b, ra.b, rUseSDF);
                        rRefl = sel(dsel(M_REFL,rm0,rm1), rRefl, rUseSDF);
                        rTran = sel(dsel(M_TRAN,rm0,rm1), rTran, rUseSDF);
                        rMetM = vor(rMetM, vand(modeMask(2,rm0,rm1,rm2), rUseSDF));
                        rAcrM = vor(rAcrM, vand(modeMask(3,rm0,rm1,rm2), rUseSDF));
                    }
                    if (any(rUseMesh)){
                        float RPxA[4],RPyA[4],RPzA[4], rrdxA[4],rrdyA[4],rrdzA[4];
                        wasm_v128_store(RPxA,RP.x); wasm_v128_store(RPyA,RP.y); wasm_v128_store(RPzA,RP.z);
                        wasm_v128_store(rrdxA,rrd.x); wasm_v128_store(rrdyA,rrd.y); wasm_v128_store(rrdzA,rrd.z);
                        float rnxA[4],rnyA[4],rnzA[4], rarA[4],ragA[4],rabA[4];
                        float rrA[4],rtrA[4];
                        int rMetA[4], rAcrA[4];
                        for (int l=0;l<4;l++){
                            if (rMeshId[l] < 0){
                                rnxA[l]=rnyA[l]=rnzA[l]=0.f; rarA[l]=ragA[l]=rabA[l]=0.f;
                                rrA[l]=rtrA[l]=0.f; rMetA[l]=rAcrA[l]=0; continue;
                            }
                            float Pl[3]={RPxA[l],RPyA[l],RPzA[l]}, rdl[3]={rrdxA[l],rrdyA[l],rrdzA[l]};
                            float Nl[3], al[3]; int ent, mode; float refl,tran,ior,gloss;
                            meshSurface(rMeshId[l], Pl, rdl, Nl, al, &ent, &mode, &refl, &tran, &ior, &gloss);
                            rnxA[l]=Nl[0]; rnyA[l]=Nl[1]; rnzA[l]=Nl[2];
                            rarA[l]=al[0]; ragA[l]=al[1]; rabA[l]=al[2];
                            rrA[l]=refl; rtrA[l]=tran;
                            rMetA[l]=(mode==2)?-1:0; rAcrA[l]=(mode==3)?-1:0;
                        }
                        V3 RNmesh = v3(wasm_v128_load(rnxA), wasm_v128_load(rnyA), wasm_v128_load(rnzA));
                        C3 raMesh = { wasm_v128_load(rarA), wasm_v128_load(ragA), wasm_v128_load(rabA) };
                        RN = v3(sel(RNmesh.x,RN.x,rUseMesh), sel(RNmesh.y,RN.y,rUseMesh), sel(RNmesh.z,RN.z,rUseMesh));
                        ra.r = sel(raMesh.r, ra.r, rUseMesh); ra.g = sel(raMesh.g, ra.g, rUseMesh); ra.b = sel(raMesh.b, ra.b, rUseMesh);
                        rRefl = sel(wasm_v128_load(rrA), rRefl, rUseMesh);
                        rTran = sel(wasm_v128_load(rtrA), rTran, rUseMesh);
                        rMetM = vor(rMetM, vand(wasm_v128_load(rMetA), rUseMesh));
                        rAcrM = vor(rAcrM, vand(wasm_v128_load(rAcrA), rUseMesh));
                    }

                    v4 rdif = vmax(v3dot(RN, Lv), S(0.f));
                    v4 lr_ = vmul(ra.r, vadd(vmul(rdif,S(1.1f)), S(0.34f)));
                    v4 lg_ = vmul(ra.g, vadd(vmul(rdif,S(1.05f)), S(0.38f)));
                    v4 lb_ = vmul(ra.b, vadd(vmul(rdif,S(0.95f)), S(0.46f)));
                    if (any(rMetM)){
                        v4 dn2 = v3dot(RN, rrd);
                        V3 rr2 = v3(vsub(rrd.x, vmul(RN.x, vmul(S(2.f),dn2))),
                                    vsub(rrd.y, vmul(RN.y, vmul(S(2.f),dn2))),
                                    vsub(rrd.z, vmul(RN.z, vmul(S(2.f),dn2))));
                        C3 e2c = skyCol(rr2, 0);
                        v4 tR = vmin(vadd(vmul(ra.r, S(1.9f)), S(0.08f)), S(1.f));
                        v4 tG = vmin(vadd(vmul(ra.g, S(1.9f)), S(0.08f)), S(1.f));
                        v4 tB = vmin(vadd(vmul(ra.b, S(1.9f)), S(0.08f)), S(1.f));
                        lr_ = sel(mixv(lr_, vmul(e2c.r,tR), rRefl), lr_, rMetM);
                        lg_ = sel(mixv(lg_, vmul(e2c.g,tG), rRefl), lg_, rMetM);
                        lb_ = sel(mixv(lb_, vmul(e2c.b,tB), rRefl), lb_, rMetM);
                    }
                    if (any(rAcrM)){
                        lr_ = sel(mixv(lr_, rc.r, rTran), lr_, rAcrM);
                        lg_ = sel(mixv(lg_, rc.g, rTran), lg_, rAcrM);
                        lb_ = sel(mixv(lb_, rc.b, rTran), lb_, rAcrM);
                    }
                    rc.r = sel(lr_, rc.r, rHit);
                    rc.g = sel(lg_, rc.g, rHit);
                    rc.b = sel(lb_, rc.b, rHit);
                }
                v4 ci = vmax(vneg(rd.y), S(0.f));
                v4 f5 = vsub(S(1.f), ci); v4 f2=vmul(f5,f5); f5 = vmul(vmul(f2,f2),f5);
                v4 kR = vmin(vadd(S(0.14f), vmul(S(0.60f), f5)), S(0.80f));
                kR = vmul(kR, vadd(S(0.55f), vmul(S(0.45f), sh)));
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

void renderScene(float t, float az, float el, float dist, int w, int h, unsigned char *fb){
    scenePrep(t);
    sceneRows(az, el, dist, w, h, fb, 0, h);
}
