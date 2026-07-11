// mesh.c — triangle-mesh renderer for the Quaternius glTF dinosaurs.
// The whole 6-model line-up is skinned each frame, packed into one BVH, and
// scalar-raytraced (primary + sun shadow + floor mirror + acrylic see-through).
// Environment (sky, checker ground, fog, gamma 2.0) and the per-model materials
// mirror render.c so switching between the SDF herd and the mesh line-up is
// seamless. Normals are per-face (flat shading suits the low-poly models).
#include "vec.h"
#include "mesh.h"

// ---------- upload buffers ----------
static float RPOS[MAXV*3];   // rest positions
static int   JIDX[MAXV*4];   // joint indices (global)
static float JW[MAXV*4];     // joint weights
static int   IDX[MAXT*3];    // triangle indices (global)
static float COL[MAXT*3];    // per-triangle color (linear)
static int   TMDL[MAXT];     // per-triangle model id
static float BONE[MAXJ*12];  // per-frame skin matrices (3x4 row-major)
static int   NV = 0, NT = 0, NJ = 0;

float* meshPos(void){ return RPOS; }
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
void meshMat(int i, int mode, float refl, float tran, float ior, float tex, float gloss){
    if (i<0||i>=NMESH) return;
    M_MODE[i]=mode;
    M_REFL[i]=fclampf(refl,0.f,1.f); M_TRAN[i]=fclampf(tran,0.f,1.f);
    M_IOR[i]=fclampf(ior,1.01f,2.5f); M_TEX[i]=fclampf(tex,0.f,1.f);
    M_GLOSS[i]=fclampf(gloss,0.f,1.f);
}

// ---------- skinned vertex store ----------
static float SV[MAXV*3];

static void skin(void){
    for (int v=0; v<NV; v++){
        const float *p = RPOS + v*3;
        const int   *j = JIDX + v*4;
        const float *w = JW   + v*4;
        float ox=0.f, oy=0.f, oz=0.f;
        for (int k=0;k<4;k++){
            float wk = w[k];
            if (wk==0.f) continue;
            const float *b = BONE + j[k]*12;
            ox += wk*(b[0]*p[0]+b[1]*p[1]+b[2]*p[2]+b[3]);
            oy += wk*(b[4]*p[0]+b[5]*p[1]+b[6]*p[2]+b[7]);
            oz += wk*(b[8]*p[0]+b[9]*p[1]+b[10]*p[2]+b[11]);
        }
        SV[v*3]=ox; SV[v*3+1]=oy; SV[v*3+2]=oz;
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

static void nodeBounds(int start,int count,float*mn,float*mx){
    mn[0]=mn[1]=mn[2]=1e30f; mx[0]=mx[1]=mx[2]=-1e30f;
    for (int i=start;i<start+count;i++){
        const int *id=IDX+TRI[i]*3;
        for (int c=0;c<3;c++){ const float *pv=SV+id[c]*3;
            for (int a=0;a<3;a++){ mn[a]=fmin2(mn[a],pv[a]); mx[a]=fmax2(mx[a],pv[a]); } }
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
    int st[160],sp=0; st[sp++]=0;
    while (sp>0){
        int node=st[--sp]; float tn;
        if (!slab(ro,inv,node,best,&tn)) continue;
        if (NLEFT[node]<0){
            int s=NSTART[node],c=NCOUNT[node];
            for (int i=s;i<s+c;i++){
                int t=TRI[i]; const int *id=IDX+t*3;
                const float *v0=SV+id[0]*3,*v1=SV+id[1]*3,*v2=SV+id[2]*3;
                float e1x=v1[0]-v0[0],e1y=v1[1]-v0[1],e1z=v1[2]-v0[2];
                float e2x=v2[0]-v0[0],e2y=v2[1]-v0[1],e2z=v2[2]-v0[2];
                float px=rd[1]*e2z-rd[2]*e2y,py=rd[2]*e2x-rd[0]*e2z,pz=rd[0]*e2y-rd[1]*e2x;
                float det=e1x*px+e1y*py+e1z*pz;
                if (det>-1e-9f&&det<1e-9f) continue;
                float invd=1.f/det;
                float tvx=ro[0]-v0[0],tvy=ro[1]-v0[1],tvz=ro[2]-v0[2];
                float u=(tvx*px+tvy*py+tvz*pz)*invd; if (u<0.f||u>1.f) continue;
                float qx=tvy*e1z-tvz*e1y,qy=tvz*e1x-tvx*e1z,qz=tvx*e1y-tvy*e1x;
                float vv=(rd[0]*qx+rd[1]*qy+rd[2]*qz)*invd; if (vv<0.f||u+vv>1.f) continue;
                float tt=(e2x*qx+e2y*qy+e2z*qz)*invd;
                if (tt>1e-4f&&tt<best){ best=tt; hit=t; }
            }
        } else { st[sp++]=NLEFT[node]; st[sp++]=NLEFT[node]+1; }
    }
    *tHit=best; return hit;
}
static int occluded(const float*ro,const float*rd,float tmax){
    float inv[3]={1.f/rd[0],1.f/rd[1],1.f/rd[2]};
    if (NNODE==0) return 0;
    int st[160],sp=0; st[sp++]=0;
    while (sp>0){
        int node=st[--sp]; float tn;
        if (!slab(ro,inv,node,tmax,&tn)) continue;
        if (NLEFT[node]<0){
            int s=NSTART[node],c=NCOUNT[node];
            for (int i=s;i<s+c;i++){
                int t=TRI[i]; const int *id=IDX+t*3;
                const float *v0=SV+id[0]*3,*v1=SV+id[1]*3,*v2=SV+id[2]*3;
                float e1x=v1[0]-v0[0],e1y=v1[1]-v0[1],e1z=v1[2]-v0[2];
                float e2x=v2[0]-v0[0],e2y=v2[1]-v0[1],e2z=v2[2]-v0[2];
                float px=rd[1]*e2z-rd[2]*e2y,py=rd[2]*e2x-rd[0]*e2z,pz=rd[0]*e2y-rd[1]*e2x;
                float det=e1x*px+e1y*py+e1z*pz;
                if (det>-1e-9f&&det<1e-9f) continue;
                float invd=1.f/det;
                float tvx=ro[0]-v0[0],tvy=ro[1]-v0[1],tvz=ro[2]-v0[2];
                float u=(tvx*px+tvy*py+tvz*pz)*invd; if (u<0.f||u>1.f) continue;
                float qx=tvy*e1z-tvz*e1y,qy=tvz*e1x-tvx*e1z,qz=tvx*e1y-tvy*e1x;
                float vv=(rd[0]*qx+rd[1]*qy+rd[2]*qz)*invd; if (vv<0.f||u+vv>1.f) continue;
                float tt=(e2x*qx+e2y*qy+e2z*qz)*invd;
                if (tt>1e-3f&&tt<tmax) return 1;
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

static void shadeMeshHit(const float*ro,const float*rd,float tH,int tri,int depth,float*out){
    int mdl=TMDL[tri];
    int mode=M_MODE[mdl];
    float refl=M_REFL[mdl], gloss=M_GLOSS[mdl];
    float P[3]={ro[0]+rd[0]*tH, ro[1]+rd[1]*tH, ro[2]+rd[2]*tH};
    const int *id=IDX+tri*3;
    const float *v0=SV+id[0]*3,*v1=SV+id[1]*3,*v2=SV+id[2]*3;
    float ax=v1[0]-v0[0],ay=v1[1]-v0[1],az=v1[2]-v0[2];
    float bx=v2[0]-v0[0],by=v2[1]-v0[1],bz=v2[2]-v0[2];
    float nx=ay*bz-az*by, ny=az*bx-ax*bz, nz=ax*by-ay*bx;
    float nl=1.f/fsqrt(nx*nx+ny*ny+nz*nz); nx*=nl; ny*=nl; nz*=nl;
    if (nx*rd[0]+ny*rd[1]+nz*rd[2]>0.f){ nx=-nx; ny=-ny; nz=-nz; }

    // color straight from the GLB material (baseColorFactor); no invented pattern
    float alr=COL[tri*3], alg=COL[tri*3+1], alb=COL[tri*3+2];

    float ndl=nx*SUN[0]+ny*SUN[1]+nz*SUN[2]; if (ndl<0.f) ndl=0.f;
    float sh=1.f;
    if (ndl>0.02f){
        float so[3]={P[0]+nx*0.006f,P[1]+ny*0.006f,P[2]+nz*0.006f};
        if (occluded(so,SUN,20.f)) sh=0.25f;
    }
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
            if (depth<2){
                float so[3]={P[0]+rd[0]*0.02f,P[1]+rd[1]*0.02f,P[2]+rd[2]*0.02f};
                shade(so,rd,depth+1,bg);
            } else skyCols(rd,0,bg);
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

static void shadeGround(const float*ro,const float*rd,float tg,int depth,float*out){
    float P[3]={ro[0]+rd[0]*tg, 0.f, ro[2]+rd[2]*tg};
    float ga[3]; groundCols(P[0],P[2],ga);
    float ndl=SUN[1], sh=1.f;
    float so[3]={P[0],0.012f,P[2]};
    if (occluded(so,SUN,20.f)) sh=0.3f;
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
static void shade(const float*ro,const float*rd,int depth,float*out){
    float tg = rd[1]<-1e-4f ? (-ro[1]/rd[1]) : 1e9f;
    float tMesh; int tri=traceMesh(ro,rd,(tg<1e8f?tg:1e9f),&tMesh);
    if (tri>=0){ shadeMeshHit(ro,rd,tMesh,tri,depth,out); return; }
    if (tg<1e8f){ shadeGround(ro,rd,tg,depth,out); return; }
    skyCols(rd,1,out);
}

// ---------- frame ----------
void renderMesh(float az, float el, float dist, int w, int h, unsigned char* fb){
    float lx=0.52f,ly=0.64f,lz=0.46f;
    float il=1.f/fsqrt(lx*lx+ly*ly+lz*lz);
    SUN[0]=lx*il; SUN[1]=ly*il; SUN[2]=lz*il;

    skin();
    buildBVH();

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

    for (int y=0;y<h;y++){
        float pyf=((float)h*0.5f-((float)y+0.5f))*ih;
        unsigned char *row=fb+(unsigned)(y*w)*4u;
        for (int x=0;x<w;x++){
            float pxf=((float)x+0.5f-(float)w*0.5f)*ih;
            float rdx=fx*FL+rx*pxf+ux*pyf, rdy=fy*FL+uy*pyf, rdz=fz*FL+rz*pxf+uz*pyf;
            float rn=1.f/fsqrt(rdx*rdx+rdy*rdy+rdz*rdz);
            float ro[3]={cxx,cyy,czz}, rd[3]={rdx*rn,rdy*rn,rdz*rn};
            float c[3]; shade(ro,rd,0,c);
            row[x*4]  =(unsigned)(fsqrt(fclampf(c[0],0.f,1.f))*255.f);
            row[x*4+1]=(unsigned)(fsqrt(fclampf(c[1],0.f,1.f))*255.f);
            row[x*4+2]=(unsigned)(fsqrt(fclampf(c[2],0.f,1.f))*255.f);
            row[x*4+3]=0xff;
        }
    }
}
