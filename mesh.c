// mesh.c — triangle-mesh renderer for the Quaternius glTF dinosaurs.
// The whole 6-model line-up is skinned each frame, packed into one BVH, and
// scalar-raytraced (primary + sun shadow + floor mirror + acrylic see-through).
// Environment (sky, checker ground, fog, gamma 2.0) and the per-model materials
// mirror render.c so switching between the SDF herd and the mesh line-up is
// seamless. Normals are per-face by default (flat shading suits the low-poly
// models); each material can opt into interpolating the GLB's authored
// per-vertex normals instead (Gouraud) — each triangle corner is its own
// vertex record in this format (no shared indices across faces), so the
// source NORMAL attribute, not the index buffer, is what carries the
// model's smoothing groups.
#include "vec.h"
#include "mesh.h"

// ---------- upload buffers ----------
static float RPOS[MAXV*3];   // rest positions
static float RNRM[MAXV*3];   // rest normals (from the GLB's NORMAL attribute)
static int   JIDX[MAXV*4];   // joint indices (global)
static float JW[MAXV*4];     // joint weights
static int   IDX[MAXT*3];    // triangle indices (global)
static float COL[MAXT*3];    // per-triangle color (linear)
static int   TMDL[MAXT];     // per-triangle model id
static float BONE[MAXJ*12];  // per-frame skin matrices (3x4 row-major)
static int   NV = 0, NT = 0, NJ = 0;

float* meshPos(void){ return RPOS; }
float* meshNormal(void){ return RNRM; }
int*   meshJoint(void){ return JIDX; }
float* meshWeight(void){ return JW; }
int*   meshIndex(void){ return IDX; }
float* meshColor(void){ return COL; }
int*   meshTriModel(void){ return TMDL; }
float* meshBone(void){ return BONE; }
void meshSetCounts(int nv, int nt, int nj){
    NV = nv<MAXV?nv:MAXV; NT = nt<MAXT?nt:MAXT; NJ = nj<MAXJ?nj:MAXJ;
}

static float FOCX=0.f, FOCZ=0.f;
void meshSetFocus(float x, float z){ FOCX=x; FOCZ=z; }

// ---------- per-model material (mirrors render.c) ----------
static int   M_MODE[NMESH]  = {0,0,0,0,0,0};   // plain: show the GLB's own colors
static float M_REFL[NMESH]  = {0.5f,0.5f,0.5f,0.5f,0.5f,0.5f};
static float M_TRAN[NMESH]  = {0.65f,0.65f,0.65f,0.65f,0.65f,0.65f};
static float M_IOR[NMESH]   = {1.49f,1.49f,1.49f,1.49f,1.49f,1.49f};
static float M_TEX[NMESH]   = {1.f,1.f,1.f,1.f,1.f,1.f};
static float M_GLOSS[NMESH] = {0.5f,0.5f,0.5f,0.5f,0.5f,0.5f};
static int   M_SMOOTH[NMESH]= {1,1,1,1,1,1};   // welded (Gouraud) vertex normals vs. flat per-face
void meshMat(int i, int mode, float refl, float tran, float ior, float tex, float gloss, int smooth){
    if (i<0||i>=NMESH) return;
    M_MODE[i]=mode;
    M_REFL[i]=fclampf(refl,0.f,1.f); M_TRAN[i]=fclampf(tran,0.f,1.f);
    M_IOR[i]=fclampf(ior,1.01f,2.5f); M_TEX[i]=fclampf(tex,0.f,1.f);
    M_GLOSS[i]=fclampf(gloss,0.f,1.f); M_SMOOTH[i]=smooth!=0;
}

// ---------- skinned vertex store ----------
static float SV[MAXV*3];
static float SN[MAXV*3];   // skinned per-vertex normals (texture-mode Gouraud shading)

static void skin(void){
    for (int v=0; v<NV; v++){
        const float *p = RPOS + v*3;
        const float *rn = RNRM + v*3;
        const int   *j = JIDX + v*4;
        const float *w = JW   + v*4;
        float ox=0.f, oy=0.f, oz=0.f;
        float nx=0.f, ny=0.f, nz=0.f;
        for (int k=0;k<4;k++){
            float wk = w[k];
            if (wk==0.f) continue;
            const float *b = BONE + j[k]*12;
            ox += wk*(b[0]*p[0]+b[1]*p[1]+b[2]*p[2]+b[3]);
            oy += wk*(b[4]*p[0]+b[5]*p[1]+b[6]*p[2]+b[7]);
            oz += wk*(b[8]*p[0]+b[9]*p[1]+b[10]*p[2]+b[11]);
            nx += wk*(b[0]*rn[0]+b[1]*rn[1]+b[2]*rn[2]);
            ny += wk*(b[4]*rn[0]+b[5]*rn[1]+b[6]*rn[2]);
            nz += wk*(b[8]*rn[0]+b[9]*rn[1]+b[10]*rn[2]);
        }
        SV[v*3]=ox; SV[v*3+1]=oy; SV[v*3+2]=oz;
        float nl2 = nx*nx+ny*ny+nz*nz;
        if (nl2>1e-20f){ float il=1.f/fsqrt(nl2); nx*=il; ny*=il; nz*=il; }
        SN[v*3]=nx; SN[v*3+1]=ny; SN[v*3+2]=nz;
    }
}

// ---------- BVH (median split, rebuilt each frame) ----------
#define MAXNODE (MAXT*2 + 8)
#define LEAF 4
static float NMIN[MAXNODE*3], NMAX[MAXNODE*3];
static int   NLEFT[MAXNODE], NSTART[MAXNODE], NCOUNT[MAXNODE];
static int   TRI[MAXT];
static float CEN[MAXT*3];
static int   NNODE = 0;

static inline float fmin2(float a,float b){ return a<b?a:b; }
static inline float fmax2(float a,float b){ return a>b?a:b; }

// SoA triangle store in BVH (TRI) order, for 4-wide SIMD Moller-Trumbore.
// Padded by 4 so a leaf's tail group can over-read safely (padding = det 0).
static float PV0X[MAXT+4],PV0Y[MAXT+4],PV0Z[MAXT+4];
static float PE1X[MAXT+4],PE1Y[MAXT+4],PE1Z[MAXT+4];
static float PE2X[MAXT+4],PE2Y[MAXT+4],PE2Z[MAXT+4];
static int   PTRI[MAXT+4];
static void packTris(void){
    for (int p=0;p<NT;p++){
        int t=TRI[p]; const int *id=IDX+t*3;
        const float *v0=SV+id[0]*3,*v1=SV+id[1]*3,*v2=SV+id[2]*3;
        PV0X[p]=v0[0]; PV0Y[p]=v0[1]; PV0Z[p]=v0[2];
        PE1X[p]=v1[0]-v0[0]; PE1Y[p]=v1[1]-v0[1]; PE1Z[p]=v1[2]-v0[2];
        PE2X[p]=v2[0]-v0[0]; PE2Y[p]=v2[1]-v0[1]; PE2Z[p]=v2[2]-v0[2];
        PTRI[p]=t;
    }
    for (int p=NT;p<NT+4;p++){
        PV0X[p]=PV0Y[p]=PV0Z[p]=0.f;
        PE1X[p]=PE1Y[p]=PE1Z[p]=0.f;
        PE2X[p]=PE2Y[p]=PE2Z[p]=0.f;
        PTRI[p]=-1;
    }
}
#define LD(a,b) wasm_v128_load((a)+(b))

static void nodeBounds(int start,int count,float*mn,float*mx){
    mn[0]=mn[1]=mn[2]=1e30f; mx[0]=mx[1]=mx[2]=-1e30f;
    for (int i=start;i<start+count;i++){
        const int *id=IDX+TRI[i]*3;
        for (int c=0;c<3;c++){ const float *pv=SV+id[c]*3;
            for (int a=0;a<3;a++){ mn[a]=fmin2(mn[a],pv[a]); mx[a]=fmax2(mx[a],pv[a]); } }
    }
}
// Refit: keep the tree topology + TRI order, just recompute node bounds from the
// current skinned vertices. Children are always allocated after their parent
// (l,r = NNODE++ inside the split), so a reverse-index pass finishes children
// before parents. O(nodes) vs the O(n log n) median-split rebuild.
static void refitBVH(void){
    for (int node=NNODE-1; node>=0; node--){
        if (NLEFT[node]<0){
            nodeBounds(NSTART[node],NCOUNT[node],NMIN+node*3,NMAX+node*3);
        } else {
            int l=NLEFT[node], r=l+1;
            for (int a=0;a<3;a++){
                NMIN[node*3+a]=fmin2(NMIN[l*3+a],NMIN[r*3+a]);
                NMAX[node*3+a]=fmax2(NMAX[l*3+a],NMAX[r*3+a]);
            }
        }
    }
}
static void buildBVH(void){
    for (int t=0;t<NT;t++){
        const int *id=IDX+t*3;
        const float *a=SV+id[0]*3,*b=SV+id[1]*3,*c=SV+id[2]*3;
        CEN[t*3]=(a[0]+b[0]+c[0])*(1.f/3.f);
        CEN[t*3+1]=(a[1]+b[1]+c[1])*(1.f/3.f);
        CEN[t*3+2]=(a[2]+b[2]+c[2])*(1.f/3.f);
        TRI[t]=t;
    }
    NNODE=0;
    int sN[160],sS[160],sC[160],sp=0;
    int root=NNODE++; sN[sp]=root; sS[sp]=0; sC[sp]=NT; sp++;
    while (sp>0){
        sp--; int node=sN[sp],start=sS[sp],count=sC[sp];
        nodeBounds(start,count,NMIN+node*3,NMAX+node*3);
        NSTART[node]=start; NCOUNT[node]=count; NLEFT[node]=-1;
        if (count<=LEAF) continue;
        float cmn[3]={1e30f,1e30f,1e30f},cmx[3]={-1e30f,-1e30f,-1e30f};
        for (int i=start;i<start+count;i++){ const float *cc=CEN+TRI[i]*3;
            for (int a=0;a<3;a++){ cmn[a]=fmin2(cmn[a],cc[a]); cmx[a]=fmax2(cmx[a],cc[a]); } }
        int axis=0; float ext=cmx[0]-cmn[0];
        if (cmx[1]-cmn[1]>ext){axis=1;ext=cmx[1]-cmn[1];}
        if (cmx[2]-cmn[2]>ext){axis=2;ext=cmx[2]-cmn[2];}
        if (ext<1e-8f) continue;
        float mid=0.5f*(cmn[axis]+cmx[axis]);
        int i=start,jN=start+count-1;
        while (i<=jN){
            if (CEN[TRI[i]*3+axis]<mid) i++;
            else { int tmp=TRI[i]; TRI[i]=TRI[jN]; TRI[jN]=tmp; jN--; }
        }
        int lc=i-start; if (lc==0||lc==count) lc=count/2;
        int l=NNODE++, r=NNODE++;
        NLEFT[node]=l;
        sN[sp]=l; sS[sp]=start;    sC[sp]=lc;       sp++;
        sN[sp]=r; sS[sp]=start+lc; sC[sp]=count-lc; sp++;
    }
}

static inline int slab(const float*ro,const float*inv,int node,float tmax,float*tn){
    float t0=0.f,t1=tmax; const float*mn=NMIN+node*3,*mx=NMAX+node*3;
    for (int a=0;a<3;a++){
        float lo=(mn[a]-ro[a])*inv[a], hi=(mx[a]-ro[a])*inv[a];
        if (lo>hi){float s=lo;lo=hi;hi=s;}
        if (lo>t0)t0=lo; if (hi<t1)t1=hi; if (t0>t1) return 0;
    }
    *tn=t0; return 1;
}
// nearest triangle; returns tri id or -1, writes t
static int traceMesh(const float*ro,const float*rd,float tmax,float*tHit){
    float inv[3]={1.f/rd[0],1.f/rd[1],1.f/rd[2]};
    float best=tmax; int hit=-1;
    if (NNODE==0) return -1;
    v4 rdx=S(rd[0]),rdy=S(rd[1]),rdz=S(rd[2]);
    v4 rox=S(ro[0]),roy=S(ro[1]),roz=S(ro[2]);
    v4 LANE=wasm_f32x4_make(0.f,1.f,2.f,3.f);
    int st[160],sp=0; st[sp++]=0;
    while (sp>0){
        int node=st[--sp]; float tn;
        if (!slab(ro,inv,node,best,&tn)) continue;
        if (NLEFT[node]<0){
            int s=NSTART[node],c=NCOUNT[node];
            for (int base=s; base<s+c; base+=4){
                int m=s+c-base; if (m>4) m=4;
                v4 e1x=LD(PE1X,base),e1y=LD(PE1Y,base),e1z=LD(PE1Z,base);
                v4 e2x=LD(PE2X,base),e2y=LD(PE2Y,base),e2z=LD(PE2Z,base);
                v4 v0x=LD(PV0X,base),v0y=LD(PV0Y,base),v0z=LD(PV0Z,base);
                v4 px=vfnma(rdz,e2y, vmul(rdy,e2z));   // rd x e2
                v4 py=vfnma(rdx,e2z, vmul(rdz,e2x));
                v4 pz=vfnma(rdy,e2x, vmul(rdx,e2y));
                v4 det=vfma(e1z,pz, vfma(e1y,py, vmul(e1x,px)));
                v4 invd=vdiv(S(1.f),det);
                v4 tvx=vsub(rox,v0x),tvy=vsub(roy,v0y),tvz=vsub(roz,v0z);
                v4 u=vmul(vfma(tvz,pz, vfma(tvy,py, vmul(tvx,px))), invd);
                v4 qx=vfnma(tvz,e1y, vmul(tvy,e1z));   // tv x e1
                v4 qy=vfnma(tvx,e1z, vmul(tvz,e1x));
                v4 qz=vfnma(tvy,e1x, vmul(tvx,e1y));
                v4 vv=vmul(vfma(rdz,qz, vfma(rdy,qy, vmul(rdx,qx))), invd);
                v4 tt=vmul(vfma(e2z,qz, vfma(e2y,qy, vmul(e2x,qx))), invd);
                v4 ok=vgt(vabs(det), S(1e-9f));
                ok=vand(ok, vge(u,S(0.f)));  ok=vand(ok, vle(u,S(1.f)));
                ok=vand(ok, vge(vv,S(0.f))); ok=vand(ok, vle(vadd(u,vv),S(1.f)));
                ok=vand(ok, vgt(tt,S(1e-4f)));
                ok=vand(ok, vlt(LANE, S((float)m)));
                tt=sel(tt, S(1e30f), ok);
                float tv[4]; wasm_v128_store(tv, tt);
                for (int l=0;l<m;l++) if (tv[l]<best){ best=tv[l]; hit=PTRI[base+l]; }
            }
        } else { st[sp++]=NLEFT[node]; st[sp++]=NLEFT[node]+1; }
    }
    *tHit=best; return hit;
}
static int occluded(const float*ro,const float*rd,float tmax){
    float inv[3]={1.f/rd[0],1.f/rd[1],1.f/rd[2]};
    if (NNODE==0) return 0;
    v4 rdx=S(rd[0]),rdy=S(rd[1]),rdz=S(rd[2]);
    v4 rox=S(ro[0]),roy=S(ro[1]),roz=S(ro[2]);
    v4 LANE=wasm_f32x4_make(0.f,1.f,2.f,3.f);
    int st[160],sp=0; st[sp++]=0;
    while (sp>0){
        int node=st[--sp]; float tn;
        if (!slab(ro,inv,node,tmax,&tn)) continue;
        if (NLEFT[node]<0){
            int s=NSTART[node],c=NCOUNT[node];
            for (int base=s; base<s+c; base+=4){
                int m=s+c-base; if (m>4) m=4;
                v4 e1x=LD(PE1X,base),e1y=LD(PE1Y,base),e1z=LD(PE1Z,base);
                v4 e2x=LD(PE2X,base),e2y=LD(PE2Y,base),e2z=LD(PE2Z,base);
                v4 v0x=LD(PV0X,base),v0y=LD(PV0Y,base),v0z=LD(PV0Z,base);
                v4 px=vfnma(rdz,e2y, vmul(rdy,e2z));
                v4 py=vfnma(rdx,e2z, vmul(rdz,e2x));
                v4 pz=vfnma(rdy,e2x, vmul(rdx,e2y));
                v4 det=vfma(e1z,pz, vfma(e1y,py, vmul(e1x,px)));
                v4 invd=vdiv(S(1.f),det);
                v4 tvx=vsub(rox,v0x),tvy=vsub(roy,v0y),tvz=vsub(roz,v0z);
                v4 u=vmul(vfma(tvz,pz, vfma(tvy,py, vmul(tvx,px))), invd);
                v4 qx=vfnma(tvz,e1y, vmul(tvy,e1z));
                v4 qy=vfnma(tvx,e1z, vmul(tvz,e1x));
                v4 qz=vfnma(tvy,e1x, vmul(tvx,e1y));
                v4 vv=vmul(vfma(rdz,qz, vfma(rdy,qy, vmul(rdx,qx))), invd);
                v4 tt=vmul(vfma(e2z,qz, vfma(e2y,qy, vmul(e2x,qx))), invd);
                v4 ok=vgt(vabs(det), S(1e-9f));
                ok=vand(ok, vge(u,S(0.f)));  ok=vand(ok, vle(u,S(1.f)));
                ok=vand(ok, vge(vv,S(0.f))); ok=vand(ok, vle(vadd(u,vv),S(1.f)));
                ok=vand(ok, vgt(tt,S(1e-3f))); ok=vand(ok, vlt(tt,S(tmax)));
                ok=vand(ok, vlt(LANE, S((float)m)));
                if (any(ok)) return 1;
            }
        } else { st[sp++]=NLEFT[node]; st[sp++]=NLEFT[node]+1; }
    }
    return 0;
}

// ---------- environment (scalar mirror of render.c) ----------
static float SUN[3];
static inline float ffloorf(float x){ return __builtin_floorf(x); }
static float hashs(float ix,float iy){ float s=fsin(ix*127.1f+iy*311.7f)*43758.547f; return s-ffloorf(s); }
static float noises(float x,float y){
    float ix=ffloorf(x),iy=ffloorf(y),fx=x-ix,fy=y-iy;
    float ux=fx*fx*(3.f-2.f*fx),uy=fy*fy*(3.f-2.f*fy);
    float a=hashs(ix,iy),b=hashs(ix+1.f,iy),c=hashs(ix,iy+1.f),d=hashs(ix+1.f,iy+1.f);
    float ab=a+(b-a)*ux, cd=c+(d-c)*ux; return ab+(cd-ab)*uy;
}
static void skyCols(const float*rd,int clouds,float*o){
    float sy=fclampf(rd[1]*1.7f,0.f,1.f);
    float r=0.82f+(0.30f-0.82f)*sy, g=0.89f+(0.52f-0.89f)*sy, b=1.00f+(0.88f-1.00f)*sy;
    float sd=rd[0]*SUN[0]+rd[1]*SUN[1]+rd[2]*SUN[2]; if (sd<0.f) sd=0.f;
    float g2=sd*sd; g2*=g2; g2*=g2; g2*=g2; g2*=g2; g2*=g2;
    r+=g2*0.9f; g+=g2*0.75f; b+=g2*0.45f;
    if (clouds){
        float iy=1.f/(ffabs(rd[1])+0.10f);
        float u=rd[0]*iy*0.9f, v=rd[2]*iy*0.9f;
        float n=noises(u,v)*0.62f+noises(u*2.6f+11.3f,v*2.6f)*0.38f;
        float cov=fclampf((n-0.50f)*3.0f,0.f,1.f)*fclampf((rd[1]-0.03f)*7.f,0.f,1.f); cov*=0.85f;
        r+=(0.99f-r)*cov; g+=(0.99f-g)*cov; b+=(1.00f-b)*cov;
    }
    o[0]=r; o[1]=g; o[2]=b;
}
static void groundCols(float x,float z,float*o){
    float gu=ffloorf(x*1.1f),gv=ffloorf(z*1.1f),gs=gu+gv;
    float ck=gs-2.f*ffloorf(gs*0.5f);
    float grime=noises(x*0.7f,z*0.7f)*0.10f;
    o[0]=(0.31f+0.10f*ck)+grime; o[1]=(0.35f+0.10f*ck)+grime; o[2]=(0.26f+0.09f*ck)+grime;
}
// reflected-ray environment: sky, blended with the ground plane where it points down
static void envReflect(const float*P,const float*rd,float*o){
    skyCols(rd,0,o);
    if (rd[1]<-1e-3f){
        float tg=-P[1]/rd[1];
        float gx=P[0]+rd[0]*tg, gz=P[2]+rd[2]*tg, gc[3]; groundCols(gx,gz,gc);
        float fade=fclampf(tg*0.18f,0.f,1.f);
        for (int c=0;c<3;c++) o[c]=gc[c]*0.9f+(o[c]-gc[c]*0.9f)*fade;
    }
}

// ---------- shading (recursive: floor mirror + acrylic see-through) ----------
static void shade(const float*ro,const float*rd,int depth,float*out);

// Shading normal for a mesh hit: face normal, ray-facing flip (records `enter`
// for acrylic), then optional Gouraud interpolation of the welded vertex normals.
// Factored out so the packet path can compute it per lane before batching shadows.
static int meshHitNormal(const float*ro,const float*rd,float tH,int tri,float*N){
    int mdl=TMDL[tri];
    float P[3]={ro[0]+rd[0]*tH, ro[1]+rd[1]*tH, ro[2]+rd[2]*tH};
    const int *id=IDX+tri*3;
    const float *v0=SV+id[0]*3,*v1=SV+id[1]*3,*v2=SV+id[2]*3;
    float ax=v1[0]-v0[0],ay=v1[1]-v0[1],az=v1[2]-v0[2];
    float bx=v2[0]-v0[0],by=v2[1]-v0[1],bz=v2[2]-v0[2];
    float nx=ay*bz-az*by, ny=az*bx-ax*bz, nz=ax*by-ay*bx;
    float nl=1.f/fsqrt(nx*nx+ny*ny+nz*nz); nx*=nl; ny*=nl; nz*=nl;
    int enter = nx*rd[0]+ny*rd[1]+nz*rd[2] <= 0.f;
    if (!enter){ nx=-nx; ny=-ny; nz=-nz; }
    if (M_SMOOTH[mdl]){              // welded: Gouraud — interpolate vertex normals
        float vpx=P[0]-v0[0], vpy=P[1]-v0[1], vpz=P[2]-v0[2];
        float d00=ax*ax+ay*ay+az*az, d01=ax*bx+ay*by+az*bz, d11=bx*bx+by*by+bz*bz;
        float d20=vpx*ax+vpy*ay+vpz*az, d21=vpx*bx+vpy*by+vpz*bz;
        float den=d00*d11-d01*d01, invd=den!=0.f?1.f/den:0.f;
        float bv=(d11*d20-d01*d21)*invd, bw=(d00*d21-d01*d20)*invd, bu=1.f-bv-bw;
        const float *n0=SN+id[0]*3,*n1=SN+id[1]*3,*n2=SN+id[2]*3;
        float sx=bu*n0[0]+bv*n1[0]+bw*n2[0];
        float sy=bu*n0[1]+bv*n1[1]+bw*n2[1];
        float sz=bu*n0[2]+bv*n1[2]+bw*n2[2];
        float sl=1.f/fsqrt(sx*sx+sy*sy+sz*sz+1e-20f); sx*=sl; sy*=sl; sz*=sl;
        if (sx*rd[0]+sy*rd[1]+sz*rd[2]>0.f){ sx=-sx; sy=-sy; sz=-sz; }
        nx=sx; ny=sy; nz=sz;
    }
    N[0]=nx; N[1]=ny; N[2]=nz; return enter;
}

// Shade a mesh hit given its precomputed shading normal, `enter` flag and sun
// shadow factor (the packet path supplies these so the shadow ray can be batched).
static void shadeMeshHit(const float*ro,const float*rd,float tH,int tri,int depth,
                         const float*Nin,int enter,float sh,float*out){
    int mdl=TMDL[tri];
    int mode=M_MODE[mdl];
    float refl=M_REFL[mdl], gloss=M_GLOSS[mdl];
    float P[3]={ro[0]+rd[0]*tH, ro[1]+rd[1]*tH, ro[2]+rd[2]*tH};
    float nx=Nin[0], ny=Nin[1], nz=Nin[2];

    // color straight from the GLB material (baseColorFactor); no invented pattern
    float alr=COL[tri*3], alg=COL[tri*3+1], alb=COL[tri*3+2];

    float ndl=nx*SUN[0]+ny*SUN[1]+nz*SUN[2]; if (ndl<0.f) ndl=0.f;
    float dif=ndl*sh;
    float hx=SUN[0]-rd[0],hy=SUN[1]-rd[1],hz=SUN[2]-rd[2];
    float hl=1.f/fsqrt(hx*hx+hy*hy+hz*hz); hx*=hl; hy*=hl; hz*=hl;
    float spb=nx*hx+ny*hy+nz*hz; if (spb<0.f) spb=0.f;
    spb*=spb; spb*=spb; spb*=spb; spb*=spb;
    float glossK = (mode==2) ? gloss*0.7f*3.4f : gloss*0.7f;
    float sp=spb*glossK*sh;
    float amb=0.55f+0.45f*ny;
    float cr=alr*(dif*1.30f+amb*0.42f)+sp;
    float cg=alg*(dif*1.22f+amb*0.50f)+sp;
    float cb=alb*(dif*1.05f+amb*0.66f)+sp;

    float dnr=nx*rd[0]+ny*rd[1]+nz*rd[2];
    float ci=-dnr; if (ci<0.f) ci=0.f;
    float f5=1.f-ci; f5=f5*f5; f5=f5*f5*(1.f-ci);
    float rrd[3]={rd[0]-2.f*dnr*nx, rd[1]-2.f*dnr*ny, rd[2]-2.f*dnr*nz};

    if (mode==2){                                   // metal: albedo-tinted env mirror
        float env[3]; envReflect(P,rrd,env);
        float Fm=refl*(0.62f+0.38f*f5); Fm*=0.6f+0.4f*sh;
        float tR=fmin2(alr*1.9f+0.08f,1.f), tG=fmin2(alg*1.9f+0.08f,1.f), tB=fmin2(alb*1.9f+0.08f,1.f);
        float dk=1.f-refl*0.72f;
        cr=cr*dk+(env[0]*tR-cr*dk)*Fm;
        cg=cg*dk+(env[1]*tG-cg*dk)*Fm;
        cb=cb*dk+(env[2]*tB-cb*dk)*Fm;
    } else {                                        // dielectric fresnel sheen
        float F=(0.030f+0.55f*f5)*(refl*2.f); F*=0.5f+0.5f*sh;
        float env[3]; envReflect(P,rrd,env);
        cr+=(env[0]-cr)*F; cg+=(env[1]-cg)*F; cb+=(env[2]-cb)*F;
        if (mode==3){                               // acrylic: see-through behind + fresnel
            float tran=M_TRAN[mdl];
            float bg[3];
            float dir[3]={rd[0],rd[1],rd[2]};
            if (enter){                             // entering the body: bend the ray by its IOR
                float eta=1.f/M_IOR[mdl];
                float cosi=-(nx*rd[0]+ny*rd[1]+nz*rd[2]);
                float k=1.f-eta*eta*(1.f-cosi*cosi);
                if (k>=0.f){
                    float sk=fsqrt(k), coef=eta*cosi-sk;
                    dir[0]=eta*rd[0]+coef*nx; dir[1]=eta*rd[1]+coef*ny; dir[2]=eta*rd[2]+coef*nz;
                    float rl=1.f/fsqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);
                    dir[0]*=rl; dir[1]*=rl; dir[2]*=rl;
                }
            }                                        // leaving the body: pass straight through
            if (depth<2){
                float so[3]={P[0]+dir[0]*0.02f,P[1]+dir[1]*0.02f,P[2]+dir[2]*0.02f};
                shade(so,dir,depth+1,bg);
            } else skyCols(dir,0,bg);
            float kt=tran*(1.f-f5*0.5f);
            // thin tint by albedo, so the body keeps a hint of its color
            cr+=(bg[0]*(0.5f+0.5f*alr)-cr)*kt;
            cg+=(bg[1]*(0.5f+0.5f*alg)-cg)*kt;
            cb+=(bg[2]*(0.5f+0.5f*alb)-cb)*kt;
        }
    }

    if (depth==0){
        float fk=tH*tH, fog=(fk*0.0016f)/(1.f+fk*0.0016f);
        cr+=(0.78f-cr)*fog; cg+=(0.86f-cg)*fog; cb+=(0.98f-cb)*fog;
    }
    out[0]=cr; out[1]=cg; out[2]=cb;
}

static void shadeGround(const float*ro,const float*rd,float tg,int depth,float sh,float*out){
    float P[3]={ro[0]+rd[0]*tg, 0.f, ro[2]+rd[2]*tg};
    float ga[3]; groundCols(P[0],P[2],ga);
    float ndl=SUN[1];
    float dif=ndl*sh, amb=1.f;
    float cr=ga[0]*(dif*1.30f+amb*0.42f);
    float cg=ga[1]*(dif*1.22f+amb*0.50f);
    float cb=ga[2]*(dif*1.05f+amb*0.66f);
    // floor mirror: reflect the line-up (one bounce)
    float rc[3];
    float rrd[3]={rd[0],-rd[1],rd[2]};
    if (depth<1){
        float rro[3]={P[0],0.012f,P[2]};
        shade(rro,rrd,depth+1,rc);
    } else skyCols(rrd,1,rc);
    float ci=-rd[1]; if (ci<0.f) ci=0.f;
    float f5=1.f-ci; f5=f5*f5; f5=f5*f5*(1.f-ci);
    float kR=0.14f+0.60f*f5; if (kR>0.80f) kR=0.80f; kR*=0.55f+0.45f*sh;
    cr+=(rc[0]-cr)*kR; cg+=(rc[1]-cg)*kR; cb+=(rc[2]-cb)*kR;
    if (depth==0){
        float fk=tg*tg, fog=(fk*0.0016f)/(1.f+fk*0.0016f);
        cr+=(0.78f-cr)*fog; cg+=(0.86f-cg)*fog; cb+=(0.98f-cb)*fog;
    }
    out[0]=cr; out[1]=cg; out[2]=cb;
}

// nearest of mesh / ground / sky
// Scalar convenience wrapper (secondary/recursive rays): compute normal + sun
// shadow, then shade. The primary packet path computes these batched instead.
static void shadeMeshHitAuto(const float*ro,const float*rd,float tH,int tri,int depth,float*out){
    float N[3]; int enter=meshHitNormal(ro,rd,tH,tri,N);
    float ndl=N[0]*SUN[0]+N[1]*SUN[1]+N[2]*SUN[2]; if (ndl<0.f) ndl=0.f;
    float sh=1.f;
    if (ndl>0.02f){
        float P[3]={ro[0]+rd[0]*tH, ro[1]+rd[1]*tH, ro[2]+rd[2]*tH};
        float so[3]={P[0]+N[0]*0.006f,P[1]+N[1]*0.006f,P[2]+N[2]*0.006f};
        if (occluded(so,SUN,20.f)) sh=0.25f;
    }
    shadeMeshHit(ro,rd,tH,tri,depth,N,enter,sh,out);
}

static void shade(const float*ro,const float*rd,int depth,float*out){
    float tg = rd[1]<-1e-4f ? (-ro[1]/rd[1]) : 1e9f;
    float tMesh; int tri=traceMesh(ro,rd,(tg<1e8f?tg:1e9f),&tMesh);
    if (tri>=0){ shadeMeshHitAuto(ro,rd,tMesh,tri,depth,out); return; }
    if (tg<1e8f){
        float so[3]={ro[0]+rd[0]*tg, 0.012f, ro[2]+rd[2]*tg};
        float sh=occluded(so,SUN,20.f)?0.3f:1.f;
        shadeGround(ro,rd,tg,depth,sh,out); return;
    }
    skyCols(rd,1,out);
}

// 4-ray packet primary trace: one shared traversal stack, per-lane slab + per-lane
// best. Ray dirs in SoA lanes; each triangle scalar-splatted and tested against all
// 4 rays at once. Writes per-lane nearest t (tHitOut) and hit triangle id (hitId[4],
// -1 = miss). Shading stays scalar (called per lane) so output matches shade().
static void traceMeshP(const float*ro, V3 rd, v4 tmax, v4*tHitOut, int*hitId){
    v4 invx=vdiv(S(1.f),rd.x), invy=vdiv(S(1.f),rd.y), invz=vdiv(S(1.f),rd.z);
    v4 rox=S(ro[0]),roy=S(ro[1]),roz=S(ro[2]);
    v4 best=tmax, hitv=wasm_i32x4_splat(-1);
    if (NNODE==0){ *tHitOut=best; wasm_v128_store(hitId,hitv); return; }
    int st[160],sp=0; st[sp++]=0;
    while (sp>0){
        int node=st[--sp];
        const float *mn=NMIN+node*3,*mx=NMAX+node*3;
        v4 lox=vmul(vsub(S(mn[0]),rox),invx), hix=vmul(vsub(S(mx[0]),rox),invx);
        v4 t0=vmin(lox,hix), t1=vmax(lox,hix);
        v4 loy=vmul(vsub(S(mn[1]),roy),invy), hiy=vmul(vsub(S(mx[1]),roy),invy);
        t0=vmax(t0,vmin(loy,hiy)); t1=vmin(t1,vmax(loy,hiy));
        v4 loz=vmul(vsub(S(mn[2]),roz),invz), hiz=vmul(vsub(S(mx[2]),roz),invz);
        t0=vmax(t0,vmin(loz,hiz)); t1=vmin(t1,vmax(loz,hiz));
        t0=vmax(t0,S(0.f));
        if (!any(vand(vle(t0,t1), vlt(t0,best)))) continue;   // no lane needs this node
        if (NLEFT[node]<0){
            int s=NSTART[node],c=NCOUNT[node];
            for (int p=s;p<s+c;p++){
                v4 e1x=S(PE1X[p]),e1y=S(PE1Y[p]),e1z=S(PE1Z[p]);
                v4 e2x=S(PE2X[p]),e2y=S(PE2Y[p]),e2z=S(PE2Z[p]);
                v4 v0x=S(PV0X[p]),v0y=S(PV0Y[p]),v0z=S(PV0Z[p]);
                v4 px=vfnma(rd.z,e2y, vmul(rd.y,e2z));   // rd x e2
                v4 py=vfnma(rd.x,e2z, vmul(rd.z,e2x));
                v4 pz=vfnma(rd.y,e2x, vmul(rd.x,e2y));
                v4 det=vfma(e1z,pz, vfma(e1y,py, vmul(e1x,px)));
                v4 invd=vdiv(S(1.f),det);
                v4 tvx=vsub(rox,v0x),tvy=vsub(roy,v0y),tvz=vsub(roz,v0z);
                v4 u=vmul(vfma(tvz,pz, vfma(tvy,py, vmul(tvx,px))), invd);
                v4 qx=vfnma(tvz,e1y, vmul(tvy,e1z));   // tv x e1
                v4 qy=vfnma(tvx,e1z, vmul(tvz,e1x));
                v4 qz=vfnma(tvy,e1x, vmul(tvx,e1y));
                v4 vv=vmul(vfma(rd.z,qz, vfma(rd.y,qy, vmul(rd.x,qx))), invd);
                v4 tt=vmul(vfma(e2z,qz, vfma(e2y,qy, vmul(e2x,qx))), invd);
                v4 ok=vgt(vabs(det), S(1e-9f));
                ok=vand(ok, vge(u,S(0.f)));  ok=vand(ok, vle(u,S(1.f)));
                ok=vand(ok, vge(vv,S(0.f))); ok=vand(ok, vle(vadd(u,vv),S(1.f)));
                ok=vand(ok, vgt(tt,S(1e-4f))); ok=vand(ok, vlt(tt,best));
                best=sel(tt,best,ok);
                hitv=sel(wasm_i32x4_splat(PTRI[p]), hitv, ok);
            }
        } else { st[sp++]=NLEFT[node]; st[sp++]=NLEFT[node]+1; }
    }
    *tHitOut=best; wasm_v128_store(hitId,hitv);
}

// 4-ray packet sun-shadow (any-hit). All rays share the sun direction; only the
// origins differ (one per lane). `active` masks the lanes that actually need a
// shadow test. Returns a per-lane mask (all-ones where an occluder was found).
static v4 occludedP(v4 rox, v4 roy, v4 roz, v4 active, float tmax){
    v4 res=wasm_i32x4_splat(0);
    if (NNODE==0) return res;
    v4 ddx=S(SUN[0]),ddy=S(SUN[1]),ddz=S(SUN[2]);
    v4 invx=S(1.f/SUN[0]),invy=S(1.f/SUN[1]),invz=S(1.f/SUN[2]);
    v4 live=active;
    int st[160],sp=0; st[sp++]=0;
    while (sp>0){
        int node=st[--sp];
        const float *mn=NMIN+node*3,*mx=NMAX+node*3;
        v4 lox=vmul(vsub(S(mn[0]),rox),invx), hix=vmul(vsub(S(mx[0]),rox),invx);
        v4 t0=vmin(lox,hix), t1=vmax(lox,hix);
        v4 loy=vmul(vsub(S(mn[1]),roy),invy), hiy=vmul(vsub(S(mx[1]),roy),invy);
        t0=vmax(t0,vmin(loy,hiy)); t1=vmin(t1,vmax(loy,hiy));
        v4 loz=vmul(vsub(S(mn[2]),roz),invz), hiz=vmul(vsub(S(mx[2]),roz),invz);
        t0=vmax(t0,vmin(loz,hiz)); t1=vmin(t1,vmax(loz,hiz));
        t0=vmax(t0,S(0.f)); t1=vmin(t1,S(tmax));
        if (!any(vand(live, vle(t0,t1)))) continue;
        if (NLEFT[node]<0){
            int s=NSTART[node],c=NCOUNT[node];
            for (int p=s;p<s+c;p++){
                v4 e1x=S(PE1X[p]),e1y=S(PE1Y[p]),e1z=S(PE1Z[p]);
                v4 e2x=S(PE2X[p]),e2y=S(PE2Y[p]),e2z=S(PE2Z[p]);
                v4 v0x=S(PV0X[p]),v0y=S(PV0Y[p]),v0z=S(PV0Z[p]);
                v4 px=vfnma(ddz,e2y, vmul(ddy,e2z));
                v4 py=vfnma(ddx,e2z, vmul(ddz,e2x));
                v4 pz=vfnma(ddy,e2x, vmul(ddx,e2y));
                v4 det=vfma(e1z,pz, vfma(e1y,py, vmul(e1x,px)));
                v4 invd=vdiv(S(1.f),det);
                v4 tvx=vsub(rox,v0x),tvy=vsub(roy,v0y),tvz=vsub(roz,v0z);
                v4 u=vmul(vfma(tvz,pz, vfma(tvy,py, vmul(tvx,px))), invd);
                v4 qx=vfnma(tvz,e1y, vmul(tvy,e1z));
                v4 qy=vfnma(tvx,e1z, vmul(tvz,e1x));
                v4 qz=vfnma(tvy,e1x, vmul(tvx,e1y));
                v4 vv=vmul(vfma(ddz,qz, vfma(ddy,qy, vmul(ddx,qx))), invd);
                v4 tt=vmul(vfma(e2z,qz, vfma(e2y,qy, vmul(e2x,qx))), invd);
                v4 ok=vgt(vabs(det), S(1e-9f));
                ok=vand(ok, vge(u,S(0.f)));  ok=vand(ok, vle(u,S(1.f)));
                ok=vand(ok, vge(vv,S(0.f))); ok=vand(ok, vle(vadd(u,vv),S(1.f)));
                ok=vand(ok, vgt(tt,S(1e-3f))); ok=vand(ok, vlt(tt,S(tmax)));
                ok=vand(ok, live);
                res=vor(res,ok); live=vandn(live,ok);
                if (!any(live)) return res;
            }
        } else { st[sp++]=NLEFT[node]; st[sp++]=NLEFT[node]+1; }
    }
    return res;
}

// ---------- frame ----------
void renderMesh(float az, float el, float dist, int w, int h, unsigned char* fb){
    float lx=0.52f,ly=0.64f,lz=0.46f;
    float il=1.f/fsqrt(lx*lx+ly*ly+lz*lz);
    SUN[0]=lx*il; SUN[1]=ly*il; SUN[2]=lz*il;

    skin();
    // Rebuild the BVH topology periodically (and when the mesh set changes);
    // refit bounds on the in-between frames — the skinned pose deforms only
    // modestly frame-to-frame, so the frame-0 partition stays effective.
    static int frameNo=0, lastNT=-1;
    if (NNODE==0 || NT!=lastNT || (frameNo % 16)==0){ buildBVH(); lastNT=NT; }
    else refitBVH();
    frameNo++;
    packTris();   // SoA triangle store in BVH order for 4-wide intersection

    const float tx=FOCX, ty=0.85f, tz=FOCZ;
    float ce=fcos(el), se=fsin(el);
    float cxx=tx+dist*ce*fsin(az), cyy=ty+dist*se, czz=tz+dist*ce*fcos(az);
    float fx=tx-cxx, fy=ty-cyy, fz=tz-czz;
    float fl=1.f/fsqrt(fx*fx+fy*fy+fz*fz); fx*=fl; fy*=fl; fz*=fl;
    float rx=-fz, rz=fx;
    float rl=1.f/fsqrt(rx*rx+rz*rz); rx*=rl; rz*=rl;
    float ux=-rz*fy, uy=rz*fx-rx*fz, uz=rx*fy;
    const float FL=1.8f;
    float ih=2.0f/(float)h;
    float ro[3]={cxx,cyy,czz};
    v4 X4=wasm_f32x4_make(0.f,1.f,2.f,3.f);

    for (int y=0;y<h;y++){
        float pyf=((float)h*0.5f-((float)y+0.5f))*ih;
        unsigned char *row=fb+(unsigned)(y*w)*4u;
        for (int x=0;x<w;x+=4){
            // 4 horizontal primary rays as a SoA packet (matches render.c layout)
            v4 pxf=vmul(vsub(vadd(S((float)x+0.5f),X4), S((float)w*0.5f)), S(ih));
            v4 pyv=S(pyf);
            V3 rd=v3norm(v3(
                vadd(vadd(S(fx*FL), vmul(S(rx),pxf)), vmul(S(ux),pyv)),
                vadd(S(fy*FL), vmul(S(uy),pyv)),
                vadd(vadd(S(fz*FL), vmul(S(rz),pxf)), vmul(S(uz),pyv))));
            v4 down=vlt(rd.y, S(-1e-4f));
            v4 tg=sel(vdiv(S(-cyy), rd.y), S(1e9f), down);   // ground plane t
            v4 tmax=sel(tg, S(1e9f), vlt(tg,S(1e8f)));
            v4 tHitv; int hitId[4];
            traceMeshP(ro, rd, tmax, &tHitv, hitId);
            float rdxA[4],rdyA[4],rdzA[4],tgA[4],tHA[4];
            wasm_v128_store(rdxA,rd.x); wasm_v128_store(rdyA,rd.y); wasm_v128_store(rdzA,rd.z);
            wasm_v128_store(tgA,tg);    wasm_v128_store(tHA,tHitv);

            // Per lane: resolve normal + shadow-ray origin, then batch the 4 sun
            // shadow rays into one packet traversal (they share the sun direction).
            float NA[4][3]; int enterA[4]; int kindA[4];   // 0 sky, 1 mesh, 2 ground
            float soxA[4]={0,0,0,0}, soyA[4]={0,0,0,0}, sozA[4]={0,0,0,0};
            float actA[4]={0,0,0,0};                        // lane needs a shadow test
            for (int l=0; l<4; l++){
                if (x+l>=w){ kindA[l]=0; continue; }
                float rdl[3]={rdxA[l],rdyA[l],rdzA[l]};
                if (hitId[l]>=0){
                    kindA[l]=1;
                    enterA[l]=meshHitNormal(ro,rdl,tHA[l],hitId[l],NA[l]);
                    float ndl=NA[l][0]*SUN[0]+NA[l][1]*SUN[1]+NA[l][2]*SUN[2];
                    if (ndl>0.02f){
                        float tH=tHA[l];
                        soxA[l]=ro[0]+rdl[0]*tH+NA[l][0]*0.006f;
                        soyA[l]=ro[1]+rdl[1]*tH+NA[l][1]*0.006f;
                        sozA[l]=ro[2]+rdl[2]*tH+NA[l][2]*0.006f;
                        actA[l]=1.f;
                    }
                } else if (tgA[l]<1e8f){
                    kindA[l]=2;
                    soxA[l]=ro[0]+rdl[0]*tgA[l]; soyA[l]=0.012f; sozA[l]=ro[2]+rdl[2]*tgA[l];
                    actA[l]=1.f;
                } else kindA[l]=0;
            }
            v4 active=vgt(wasm_v128_load(actA), S(0.f));
            v4 occ=occludedP(wasm_v128_load(soxA),wasm_v128_load(soyA),wasm_v128_load(sozA),
                             active, 20.f);
            int occA[4]; wasm_v128_store(occA, occ);   // -1 (all ones) where occluded

            for (int l=0; l<4 && x+l<w; l++){
                float rdl[3]={rdxA[l],rdyA[l],rdzA[l]}, c[3];
                if (kindA[l]==1){
                    float sh = (actA[l]!=0.f && occA[l]) ? 0.25f : 1.f;
                    shadeMeshHit(ro, rdl, tHA[l], hitId[l], 0, NA[l], enterA[l], sh, c);
                } else if (kindA[l]==2){
                    float sh = occA[l] ? 0.3f : 1.f;
                    shadeGround(ro, rdl, tgA[l], 0, sh, c);
                } else skyCols(rdl, 1, c);
                unsigned char *px=row+(unsigned)(x+l)*4u;
                px[0]=(unsigned)(fsqrt(fclampf(c[0],0.f,1.f))*255.f);
                px[1]=(unsigned)(fsqrt(fclampf(c[1],0.f,1.f))*255.f);
                px[2]=(unsigned)(fsqrt(fclampf(c[2],0.f,1.f))*255.f);
                px[3]=0xff;
            }
        }
    }
}
