// dino_model.c — model + animation parameters for this demo's dinosaurs:
// theropod (biped, IK legs), stegosaurus (plates + thagomizer),
// triceratops (frill + horns), herd layout, and species albedo.
#include "vec.h"
#include "render.h"
#include "anim.h"
#include "dino_model.h"

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
// bone-part segments (ax,ay,az,bx,by,bz,r1,r2 species-local) for exact material id
#define NSPK 4
static float SPK[NSPK][8];   // stego thagomizer spikes
#define NTBN 5
static float TBN[NTBN][8];   // trike brow horns x2 + nose horn + beak + jaw
static float FRLY;           // trike frill center y (incl. bob + head dip)

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
    boneCone(SPK[0], -1.80f, 0.78f, z3+0.06f, -2.02f, 0.98f, z3+0.16f, 0.030f, 0.005f);
    boneCone(SPK[1], -1.80f, 0.78f, z3-0.06f, -2.02f, 0.98f, z3-0.16f, 0.030f, 0.005f);
    boneCone(SPK[2], -1.66f, 0.78f, z3+0.07f, -1.84f, 1.00f, z3+0.17f, 0.030f, 0.005f);
    boneCone(SPK[3], -1.66f, 0.78f, z3-0.07f, -1.84f, 1.00f, z3-0.17f, 0.030f, 0.005f);
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
    boneCone(TBN[3], 1.02f, 0.86f+bob+hd, 0.f, 1.24f, 0.78f+bob+hd, 0.f, 0.075f, 0.012f);
    float chew = 0.5f + 0.5f*fsin(GT*1.2f + 2.2f);
    boneCone(TBN[4], 0.97f, 0.795f+bob+hd, 0.f, 1.19f, 0.718f+bob+hd - 0.026f*chew, 0.f, 0.058f, 0.010f);
    EYP[4][0]=0.965f+OX; EYP[4][1]=0.975f+bob+hd; EYP[4][2]= 0.108f+OZ;
    EYP[5][0]=0.965f+OX; EYP[5][1]=0.975f+bob+hd; EYP[5][2]=-0.108f+OZ;
    EYRAD[4]=EYRAD[5]=0.022f;
    EB[2][0]=EYP[4][0]; EB[2][1]=EYP[4][1]; EB[2][2]=OZ; EB[2][3]=0.16f;
    // frill: thin disc normal to +X (axis pure X), widened y/z
    {
        float sy = 7.0f, sz = 5.6f;
        FRLY = 1.20f+bob+hd;
        coneA(0.70f, FRLY, 0.f, 0.62f, FRLY, 0.f,
              0.050f, 0.050f, 1.f, 1.f/(sy*sy), 1.f/(sz*sz));
    }
    // brow horns + nose horn
    boneCone(TBN[0], 0.94f, 1.04f+bob+hd,  0.105f, 1.34f, 1.38f+bob+hd,  0.18f, 0.042f, 0.006f);
    boneCone(TBN[1], 0.94f, 1.04f+bob+hd, -0.105f, 1.34f, 1.38f+bob+hd, -0.18f, 0.042f, 0.006f);
    boneCone(TBN[2], 1.06f, 0.95f+bob+hd, 0.f, 1.15f, 1.13f+bob+hd, 0.f, 0.038f, 0.007f);
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

// Whether the herd is placed in the scene this frame (dinoSetActive()).
// When not placed, animate() registers zero primitives and empty/zero-radius
// instance bounds instead of skipping its own call — the renderer (render.c
// / scene.c) never checks this flag or otherwise knows the herd exists; an
// unplaced herd is just an empty scene as far as it's concerned.
static int g_placed = 1;
void dinoSetActive(int active){ g_placed = active; }

void animate(float t){
    NP = 0;

    // sun direction is a fixed constant, not tied to any species, so it's
    // set unconditionally — the mesh line-up's lighting depends on it too
    float lx=0.52f, ly=0.64f, lz=0.46f;
    float il = 1.f / fsqrt(lx*lx+ly*ly+lz*lz);
    SUNX=lx*il; SUNY=ly*il; SUNZ=lz*il;

    if (!g_placed){
        for (int i=0;i<ND;i++){
            DPR[i][0]=0; DPR[i][1]=0;
            DB[i][0]=0.f; DB[i][1]=0.f; DB[i][2]=0.f; DB[i][3]=0.f;
            DXW[i]=0.f; DZW[i]=0.f;
        }
        OX = 0.f; OZ = 0.f;
        return;
    }

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
}

// species albedo (textured)
C3 dinoAlbedo(V3 P, V3 N, v4 m0, v4 m1, v4 m2){
    // local coords
    v4 selX = sel(S(DXW[0]), sel(S(DXW[1]), S(DXW[2]), m1), m0);
    v4 selZ = sel(S(DZW[0]), sel(S(DZW[1]), S(DZW[2]), m1), m0);
    v4 lx = vsub(P.x, selX), lz = vsub(P.z, selZ);
    // shared noise
    v4 n1 = vnoise(vmul(lx,S(2.6f)), vadd(vmul(P.y,S(2.6f)), vmul(lz,S(1.3f))));
    v4 n2 = vnoise(vadd(vmul(lx,S(8.f)),S(4.7f)), vmul(vadd(vmul(P.y,S(0.6f)),lz),S(8.f)));
    v4 topm = clamp01(vmul(N.y, S(1.5f)));
    v4 texf = dsel(M_TEX, m0, m1);   // 0 = flat color, 1 = full pattern

    // theropod: green w/ stripes
    v4 stripe = vmul(S(0.5f), vadd(vsin(vadd(vmul(lx,S(9.f)), vmul(lz,S(2.f)))), S(1.f)));
    v4 dk0 = vsub(S(1.f), vmul(S(0.25f), vmul(vmul(stripe, topm), texf)));
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
    // thagomizer spikes bone (exact segment distance, not a coarse box)
    v4 sd_ = S(100.f);
    if (any(m1)) for (int i=0;i<NSPK;i++) sd_ = vmin(sd_, segDist(lx, P.y, lz, SPK[i]));
    v4 sm_ = vand(m1, vlt(sd_, S(0.025f)));
    r1c = sel(S(0.82f), r1c, sm_); g1c = sel(S(0.78f), g1c, sm_); b1c = sel(S(0.66f), b1c, sm_);

    // trike: grey-green; frill rings + warm rim; horns/beak bone
    v4 r2c = S(0.37f), g2c = S(0.39f), b2c = S(0.32f);
    if (any(m2)){
        // frill membership: exact anisotropic-disc distance (axis (0.70,FRLY,0)->(0.62,FRLY,0))
        v4 pax = vsub(lx, S(0.70f)), pay = vsub(P.y, S(FRLY));
        v4 fh = clamp01(vmul(vmul(pax, S(-0.08f)), S(1.f/0.0064f)));
        v4 fqx = vadd(pax, vmul(S(0.08f), fh));
        v4 fd = vsub(vsqrt(vadd(vadd(vmul(fqx,fqx), vmul(vmul(pay,pay), S(1.f/49.f))),
                                vmul(vmul(lz,lz), S(1.f/31.36f)))), S(0.05f));
        v4 fm = vand(m2, vlt(fd, S(0.02f)));
        v4 fr = vsqrt(vadd(vmul(pay,pay), vmul(vmul(lz,lz),S(0.85f))));
        v4 ring = vmul(S(0.5f), vadd(vsin(vsub(vmul(fr,S(26.f)),S(1.6f))), S(1.f)));
        ring = mixv(S(0.5f), ring, texf);
        v4 fr_r = mixv(S(0.44f), S(0.27f), ring), fr_g = mixv(S(0.42f), S(0.24f), ring), fr_b = mixv(S(0.34f), S(0.20f), ring);
        v4 rim = clamp01(vmul(vsub(fr, S(0.36f)), S(7.f)));
        fr_r = mixv(fr_r, S(0.78f), rim); fr_g = mixv(fr_g, S(0.36f), rim); fr_b = mixv(fr_b, S(0.18f), rim);
        r2c = sel(fr_r, r2c, fm); g2c = sel(fr_g, g2c, fm); b2c = sel(fr_b, b2c, fm);
        // horns + beak + jaw bone (exact segment distance, not a coarse box)
        v4 td_ = S(100.f);
        for (int i=0;i<NTBN;i++) td_ = vmin(td_, segDist(lx, P.y, lz, TBN[i]));
        v4 hm_ = vand(m2, vlt(td_, S(0.025f)));
        r2c = sel(S(0.80f), r2c, hm_); g2c = sel(S(0.74f), g2c, hm_); b2c = sel(S(0.60f), b2c, hm_);
    }

    C3 c;
    c.r = sel(r0, sel(r1c, r2c, m1), m0);
    c.g = sel(g0, sel(g1c, g2c, m1), m0);
    c.b = sel(b0, sel(b1c, b2c, m1), m0);
    // noise modulation + belly lightening
    v4 nm = vmul(vadd(S(0.84f), vmul(S(0.26f), n1)), vadd(S(0.92f), vmul(S(0.16f), n2)));
    nm = mixv(S(0.97f), nm, texf);
    c.r = vmul(c.r, nm); c.g = vmul(c.g, nm); c.b = vmul(c.b, nm);
    v4 belly = clamp01(vmul(vsub(vneg(N.y), S(0.05f)), S(1.6f)));
    belly = vmul(vmul(belly,belly), vsub(S(3.f), vmul(S(2.f), belly)));
    c.r = mixv(c.r, S(0.80f), belly); c.g = mixv(c.g, S(0.74f), belly); c.b = mixv(c.b, S(0.55f), belly);
    // eyes
    v4 em = vlt(eyeDistAll(P), S(0.006f));
    c.r = sel(S(0.05f), c.r, em); c.g = sel(S(0.05f), c.g, em); c.b = sel(S(0.06f), c.b, em);
    return c;
}
