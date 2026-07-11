// mesh.c — triangle-mesh renderer for the Quaternius glTF dinosaurs.
// Pipeline per frame: skin vertices with the uploaded bone matrices, rebuild a
// median-split BVH over the skinned triangles, then scalar-raytrace primary +
// shadow + floor-mirror rays. Environment (sky, checker ground, fog, gamma 2.0)
// mirrors render.c so switching between the SDF herd and a mesh model is smooth.
#include "vec.h"
#include "mesh.h"

// ---------- upload buffers ----------
static float RPOS[MAXV*3];   // rest positions
static int   JIDX[MAXV*4];   // joint indices
static float JW[MAXV*4];     // joint weights
static int   IDX[MAXT*3];    // triangle indices
static float COL[MAXT*3];    // per-triangle color (linear)
static float BONE[MAXJ*12];  // per-frame skin matrices (3x4 row-major)
static int   NV = 0, NT = 0, NJ = 0;

float* meshPos(void){ return RPOS; }
int*   meshJoint(void){ return JIDX; }
float* meshWeight(void){ return JW; }
int*   meshIndex(void){ return IDX; }
float* meshColor(void){ return COL; }
float* meshBone(void){ return BONE; }
void meshSetCounts(int nv, int nt, int nj){
    NV = nv < MAXV ? nv : MAXV;
    NT = nt < MAXT ? nt : MAXT;
    NJ = nj < MAXJ ? nj : MAXJ;
}

// ---------- skinned vertex store ----------
static float SV[MAXV*3];     // world-space skinned positions

static void skin(void){
    for (int v=0; v<NV; v++){
        const float *p = RPOS + v*3;
        const int   *j = JIDX + v*4;
        const float *w = JW   + v*4;
        float ox=0.f, oy=0.f, oz=0.f;
        for (int k=0;k<4;k++){
            float wk = w[k];
            if (wk == 0.f) continue;
            const float *b = BONE + j[k]*12;   // rows: [0..3][4..7][8..11]
            float x = b[0]*p[0] + b[1]*p[1] + b[2]*p[2] + b[3];
            float y = b[4]*p[0] + b[5]*p[1] + b[6]*p[2] + b[7];
            float z = b[8]*p[0] + b[9]*p[1] + b[10]*p[2] + b[11];
            ox += wk*x; oy += wk*y; oz += wk*z;
        }
        SV[v*3]=ox; SV[v*3+1]=oy; SV[v*3+2]=oz;
    }
}

// ---------- BVH (median split, rebuilt each frame) ----------
#define MAXNODE (MAXT*2 + 8)
#define LEAF 4
static float NMIN[MAXNODE*3], NMAX[MAXNODE*3];
static int   NLEFT[MAXNODE];   // internal: left child id (right=left+1); leaf: -1
static int   NSTART[MAXNODE], NCOUNT[MAXNODE];
static int   TRI[MAXT];        // triangle index permutation
static float CEN[MAXT*3];      // triangle centroids
static int   NNODE = 0;

static inline float fminf_(float a, float b){ return a<b?a:b; }
static inline float fmaxf_(float a, float b){ return a>b?a:b; }

static void nodeBounds(int start, int count, float *mn, float *mx){
    mn[0]=mn[1]=mn[2]= 1e30f; mx[0]=mx[1]=mx[2]=-1e30f;
    for (int i=start;i<start+count;i++){
        int t = TRI[i];
        const int *id = IDX + t*3;
        for (int c=0;c<3;c++){
            const float *pv = SV + id[c]*3;
            for (int a=0;a<3;a++){
                mn[a]=fminf_(mn[a], pv[a]); mx[a]=fmaxf_(mx[a], pv[a]);
            }
        }
    }
}

static void buildBVH(void){
    // per-triangle centroids
    for (int t=0;t<NT;t++){
        const int *id = IDX + t*3;
        const float *a=SV+id[0]*3, *b=SV+id[1]*3, *c=SV+id[2]*3;
        CEN[t*3]  =(a[0]+b[0]+c[0])*(1.f/3.f);
        CEN[t*3+1]=(a[1]+b[1]+c[1])*(1.f/3.f);
        CEN[t*3+2]=(a[2]+b[2]+c[2])*(1.f/3.f);
        TRI[t]=t;
    }
    NNODE = 0;
    // explicit work stack of (node, start, count)
    int stkNode[64], stkStart[64], stkCount[64], sp=0;
    int root = NNODE++;
    stkNode[sp]=root; stkStart[sp]=0; stkCount[sp]=NT; sp++;
    while (sp>0){
        sp--;
        int node=stkNode[sp], start=stkStart[sp], count=stkCount[sp];
        nodeBounds(start, count, NMIN+node*3, NMAX+node*3);
        NSTART[node]=start; NCOUNT[node]=count; NLEFT[node]=-1;
        if (count<=LEAF) continue;
        // widest centroid axis
        float cmn[3]={1e30f,1e30f,1e30f}, cmx[3]={-1e30f,-1e30f,-1e30f};
        for (int i=start;i<start+count;i++){
            const float *cc = CEN + TRI[i]*3;
            for (int a=0;a<3;a++){ cmn[a]=fminf_(cmn[a],cc[a]); cmx[a]=fmaxf_(cmx[a],cc[a]); }
        }
        int axis=0; float ext=cmx[0]-cmn[0];
        if (cmx[1]-cmn[1]>ext){ axis=1; ext=cmx[1]-cmn[1]; }
        if (cmx[2]-cmn[2]>ext){ axis=2; ext=cmx[2]-cmn[2]; }
        if (ext < 1e-8f) continue;   // degenerate -> leaf
        float mid = 0.5f*(cmn[axis]+cmx[axis]);
        // partition TRI[start..start+count) by centroid[axis] < mid
        int i=start, jN=start+count-1;
        while (i<=jN){
            if (CEN[TRI[i]*3+axis] < mid) i++;
            else { int tmp=TRI[i]; TRI[i]=TRI[jN]; TRI[jN]=tmp; jN--; }
        }
        int leftCount = i-start;
        if (leftCount==0 || leftCount==count) leftCount = count/2;  // fallback median
        int l = NNODE++, r = NNODE++;
        NLEFT[node]=l;
        stkNode[sp]=l; stkStart[sp]=start;           stkCount[sp]=leftCount;       sp++;
        stkNode[sp]=r; stkStart[sp]=start+leftCount; stkCount[sp]=count-leftCount; sp++;
    }
}

// ---------- ray/box + ray/triangle ----------
static inline int slab(const float *ro, const float *inv, int node, float tmax, float *tnear){
    float t0=0.f, t1=tmax;
    const float *mn=NMIN+node*3, *mx=NMAX+node*3;
    for (int a=0;a<3;a++){
        float lo=(mn[a]-ro[a])*inv[a], hi=(mx[a]-ro[a])*inv[a];
        if (lo>hi){ float s=lo; lo=hi; hi=s; }
        if (lo>t0) t0=lo;
        if (hi<t1) t1=hi;
        if (t0>t1) return 0;
    }
    *tnear=t0; return 1;
}

// nearest triangle along ray; returns tri index or -1. writes hit t.
static int traceMesh(const float *ro, const float *rd, float tmax, float *tHit){
    float inv[3]={1.f/rd[0], 1.f/rd[1], 1.f/rd[2]};
    float best=tmax; int hit=-1;
    if (NNODE==0) return -1;
    int stack[64], sp=0; stack[sp++]=0;
    while (sp>0){
        int node=stack[--sp];
        float tn;
        if (!slab(ro, inv, node, best, &tn)) continue;
        if (NLEFT[node]<0){
            int s=NSTART[node], c=NCOUNT[node];
            for (int i=s;i<s+c;i++){
                int t=TRI[i];
                const int *id=IDX+t*3;
                const float *v0=SV+id[0]*3, *v1=SV+id[1]*3, *v2=SV+id[2]*3;
                float e1x=v1[0]-v0[0], e1y=v1[1]-v0[1], e1z=v1[2]-v0[2];
                float e2x=v2[0]-v0[0], e2y=v2[1]-v0[1], e2z=v2[2]-v0[2];
                float px=rd[1]*e2z-rd[2]*e2y, py=rd[2]*e2x-rd[0]*e2z, pz=rd[0]*e2y-rd[1]*e2x;
                float det=e1x*px+e1y*py+e1z*pz;
                if (det>-1e-9f && det<1e-9f) continue;
                float invd=1.f/det;
                float tvx=ro[0]-v0[0], tvy=ro[1]-v0[1], tvz=ro[2]-v0[2];
                float u=(tvx*px+tvy*py+tvz*pz)*invd;
                if (u<0.f||u>1.f) continue;
                float qx=tvy*e1z-tvz*e1y, qy=tvz*e1x-tvx*e1z, qz=tvx*e1y-tvy*e1x;
                float vv=(rd[0]*qx+rd[1]*qy+rd[2]*qz)*invd;
                if (vv<0.f||u+vv>1.f) continue;
                float tt=(e2x*qx+e2y*qy+e2z*qz)*invd;
                if (tt>1e-4f && tt<best){ best=tt; hit=t; }
            }
        } else {
            stack[sp++]=NLEFT[node];
            stack[sp++]=NLEFT[node]+1;
        }
    }
    *tHit=best; return hit;
}

// any-hit shadow test (early out)
static int occluded(const float *ro, const float *rd, float tmax){
    float inv[3]={1.f/rd[0], 1.f/rd[1], 1.f/rd[2]};
    if (NNODE==0) return 0;
    int stack[64], sp=0; stack[sp++]=0;
    while (sp>0){
        int node=stack[--sp];
        float tn;
        if (!slab(ro, inv, node, tmax, &tn)) continue;
        if (NLEFT[node]<0){
            int s=NSTART[node], c=NCOUNT[node];
            for (int i=s;i<s+c;i++){
                int t=TRI[i];
                const int *id=IDX+t*3;
                const float *v0=SV+id[0]*3, *v1=SV+id[1]*3, *v2=SV+id[2]*3;
                float e1x=v1[0]-v0[0], e1y=v1[1]-v0[1], e1z=v1[2]-v0[2];
                float e2x=v2[0]-v0[0], e2y=v2[1]-v0[1], e2z=v2[2]-v0[2];
                float px=rd[1]*e2z-rd[2]*e2y, py=rd[2]*e2x-rd[0]*e2z, pz=rd[0]*e2y-rd[1]*e2x;
                float det=e1x*px+e1y*py+e1z*pz;
                if (det>-1e-9f && det<1e-9f) continue;
                float invd=1.f/det;
                float tvx=ro[0]-v0[0], tvy=ro[1]-v0[1], tvz=ro[2]-v0[2];
                float u=(tvx*px+tvy*py+tvz*pz)*invd;
                if (u<0.f||u>1.f) continue;
                float qx=tvy*e1z-tvz*e1y, qy=tvz*e1x-tvx*e1z, qz=tvx*e1y-tvy*e1x;
                float vv=(rd[0]*qx+rd[1]*qy+rd[2]*qz)*invd;
                if (vv<0.f||u+vv>1.f) continue;
                float tt=(e2x*qx+e2y*qy+e2z*qz)*invd;
                if (tt>1e-3f && tt<tmax) return 1;
            }
        } else {
            stack[sp++]=NLEFT[node];
            stack[sp++]=NLEFT[node]+1;
        }
    }
    return 0;
}

// ---------- environment (scalar mirror of render.c) ----------
static float SUN[3];   // set in renderMesh

static inline float ffloorf(float x){ return __builtin_floorf(x); }
static float hashs(float ix, float iy){
    float s = fsin(ix*127.1f + iy*311.7f) * 43758.547f;
    return s - ffloorf(s);
}
static float noises(float x, float y){
    float ix=ffloorf(x), iy=ffloorf(y);
    float fx=x-ix, fy=y-iy;
    float ux=fx*fx*(3.f-2.f*fx), uy=fy*fy*(3.f-2.f*fy);
    float a=hashs(ix,iy), b=hashs(ix+1.f,iy);
    float c=hashs(ix,iy+1.f), d=hashs(ix+1.f,iy+1.f);
    return (a+(b-a)*ux) + ((c+(d-c)*ux) - (a+(b-a)*ux))*uy;
}
static void skyCols(const float *rd, int clouds, float *o){
    float sy = fclampf(rd[1]*1.7f, 0.f, 1.f);
    float r = 0.82f+(0.30f-0.82f)*sy;
    float g = 0.89f+(0.52f-0.89f)*sy;
    float b = 1.00f+(0.88f-1.00f)*sy;
    float sd = rd[0]*SUN[0]+rd[1]*SUN[1]+rd[2]*SUN[2];
    if (sd<0.f) sd=0.f;
    float g2=sd*sd; g2*=g2; g2*=g2; g2*=g2; g2*=g2; g2*=g2;   // sd^32-ish glow
    r += g2*0.9f; g += g2*0.75f; b += g2*0.45f;
    if (clouds){
        float iy = 1.f/(ffabs(rd[1])+0.10f);
        float u = rd[0]*iy*0.9f, v = rd[2]*iy*0.9f;
        float n = noises(u,v)*0.62f + noises(u*2.6f+11.3f, v*2.6f)*0.38f;
        float cov = fclampf((n-0.50f)*3.0f,0.f,1.f) * fclampf((rd[1]-0.03f)*7.f,0.f,1.f);
        cov *= 0.85f;
        r += (0.99f-r)*cov; g += (0.99f-g)*cov; b += (1.00f-b)*cov;
    }
    o[0]=r; o[1]=g; o[2]=b;
}
static void groundCols(float x, float z, float *o){
    float gu=ffloorf(x*1.1f), gv=ffloorf(z*1.1f);
    float gs=gu+gv;
    float ck=gs-2.f*ffloorf(gs*0.5f);
    float grime=noises(x*0.7f, z*0.7f)*0.10f;
    o[0]=(0.31f+(0.41f-0.31f)*ck)+grime;
    o[1]=(0.35f+(0.45f-0.35f)*ck)+grime;
    o[2]=(0.26f+(0.35f-0.26f)*ck)+grime;
}

// ---------- frame ----------
void renderMesh(float az, float el, float dist, int w, int h, unsigned char* fb){
    // sun (same as the SDF scene)
    float lx=0.52f, ly=0.64f, lz=0.46f;
    float il=1.f/fsqrt(lx*lx+ly*ly+lz*lz);
    SUN[0]=lx*il; SUN[1]=ly*il; SUN[2]=lz*il;

    skin();
    buildBVH();

    // camera basis (mirror render.c)
    const float tx=0.f, ty=0.85f, tz=0.f;
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
            float rdx=fx*FL+rx*pxf+ux*pyf;
            float rdy=fy*FL+uy*pyf;
            float rdz=fz*FL+rz*pxf+uz*pyf;
            float rn=1.f/fsqrt(rdx*rdx+rdy*rdy+rdz*rdz);
            rdx*=rn; rdy*=rn; rdz*=rn;
            float ro[3]={cxx,cyy,czz}, rd[3]={rdx,rdy,rdz};

            // ground plane y=0
            float tg = rdy<-1e-4f ? (-cyy/rdy) : 1e9f;
            float tMesh; int tri=traceMesh(ro, rd, (tg<1e8f?tg:1e9f), &tMesh);

            float cr,cg,cb;
            float hitT;
            if (tri>=0){
                hitT=tMesh;
                float P[3]={cxx+rdx*tMesh, cyy+rdy*tMesh, czz+rdz*tMesh};
                // geometric normal (flat shading)
                const int *id=IDX+tri*3;
                const float *v0=SV+id[0]*3, *v1=SV+id[1]*3, *v2=SV+id[2]*3;
                float ax=v1[0]-v0[0], ay=v1[1]-v0[1], az_=v1[2]-v0[2];
                float bx=v2[0]-v0[0], by=v2[1]-v0[1], bz=v2[2]-v0[2];
                float nx=ay*bz-az_*by, ny=az_*bx-ax*bz, nz=ax*by-ay*bx;
                float nl=1.f/fsqrt(nx*nx+ny*ny+nz*nz); nx*=nl; ny*=nl; nz*=nl;
                if (nx*rdx+ny*rdy+nz*rdz > 0.f){ nx=-nx; ny=-ny; nz=-nz; }  // face camera
                float ndl=nx*SUN[0]+ny*SUN[1]+nz*SUN[2]; if (ndl<0.f) ndl=0.f;
                // shadow ray to sun
                float sh=1.f;
                if (ndl>0.02f){
                    float so[3]={P[0]+nx*0.006f, P[1]+ny*0.006f, P[2]+nz*0.006f};
                    if (occluded(so, SUN, 20.f)) sh=0.25f;
                }
                float dif=ndl*sh;
                // specular (Blinn, pow 16)
                float hx=SUN[0]-rdx, hy=SUN[1]-rdy, hz=SUN[2]-rdz;
                float hl=1.f/fsqrt(hx*hx+hy*hy+hz*hz); hx*=hl; hy*=hl; hz*=hl;
                float sp=nx*hx+ny*hy+nz*hz; if (sp<0.f) sp=0.f;
                sp*=sp; sp*=sp; sp*=sp; sp*=sp; sp*=0.35f*sh;
                const float *al=COL+tri*3;
                float amb=0.55f+0.45f*ny;
                cr=al[0]*(dif*1.30f+amb*0.42f)+sp;
                cg=al[1]*(dif*1.22f+amb*0.50f)+sp;
                cb=al[2]*(dif*1.05f+amb*0.66f)+sp;
                // fresnel sky reflection (subtle sheen)
                float ci=-(nx*rdx+ny*rdy+nz*rdz); if (ci<0.f) ci=0.f;
                float f5=1.f-ci; f5=f5*f5; f5=f5*f5*(1.f-ci);
                float F=(0.03f+0.55f*f5)*0.24f;
                float refx=rdx-2.f*(nx*rdx+ny*rdy+nz*rdz)*nx;
                float refy=rdy-2.f*(nx*rdx+ny*rdy+nz*rdz)*ny;
                float refz=rdz-2.f*(nx*rdx+ny*rdy+nz*rdz)*nz;
                float rdir[3]={refx,refy,refz}, env[3]; skyCols(rdir,0,env);
                cr+=(env[0]-cr)*F; cg+=(env[1]-cg)*F; cb+=(env[2]-cb)*F;
            } else if (tg<1e8f){
                hitT=tg;
                float P[3]={cxx+rdx*tg, 0.f, czz+rdz*tg};
                float ga[3]; groundCols(P[0],P[2],ga);
                // sun on flat ground (N=+Y)
                float ndl=SUN[1];
                float sh=1.f;
                float so[3]={P[0], 0.012f, P[2]};
                if (occluded(so, SUN, 20.f)) sh=0.3f;
                float dif=ndl*sh, amb=1.f;
                cr=ga[0]*(dif*1.30f+amb*0.42f);
                cg=ga[1]*(dif*1.22f+amb*0.50f);
                cb=ga[2]*(dif*1.05f+amb*0.66f);
                // floor mirror: reflect the mesh
                float rro[3]={P[0],0.012f,P[2]}, rrd[3]={rdx,-rdy,rdz};
                float rt; int rtri=traceMesh(rro,rrd,1e9f,&rt);
                float rc[3]; skyCols(rrd,1,rc);
                if (rtri>=0){
                    float RP[3]={rro[0]+rrd[0]*rt, rro[1]+rrd[1]*rt, rro[2]+rrd[2]*rt};
                    const int *id=IDX+rtri*3;
                    const float *v0=SV+id[0]*3,*v1=SV+id[1]*3,*v2=SV+id[2]*3;
                    float ax=v1[0]-v0[0],ay=v1[1]-v0[1],az_=v1[2]-v0[2];
                    float bx=v2[0]-v0[0],by=v2[1]-v0[1],bz=v2[2]-v0[2];
                    float nx=ay*bz-az_*by, ny=az_*bx-ax*bz, nz=ax*by-ay*bx;
                    float nl=1.f/fsqrt(nx*nx+ny*ny+nz*nz); nx*=nl; ny*=nl; nz*=nl;
                    if (nx*rrd[0]+ny*rrd[1]+nz*rrd[2]>0.f){ nx=-nx; ny=-ny; nz=-nz; }
                    float rndl=nx*SUN[0]+ny*SUN[1]+nz*SUN[2]; if (rndl<0.f) rndl=0.f;
                    const float *al=COL+rtri*3;
                    rc[0]=al[0]*(rndl*1.1f+0.34f);
                    rc[1]=al[1]*(rndl*1.05f+0.38f);
                    rc[2]=al[2]*(rndl*0.95f+0.46f);
                    (void)RP;
                }
                float ci=-rdy; if (ci<0.f) ci=0.f;
                float f5=1.f-ci; f5=f5*f5; f5=f5*f5*(1.f-ci);
                float kR=0.14f+0.60f*f5; if (kR>0.80f) kR=0.80f;
                kR*=0.55f+0.45f*sh;
                cr+=(rc[0]-cr)*kR; cg+=(rc[1]-cg)*kR; cb+=(rc[2]-cb)*kR;
            } else {
                float sk[3]; skyCols(rd,1,sk);
                cr=sk[0]; cg=sk[1]; cb=sk[2];
                unsigned int rI=(unsigned)(fsqrt(fclampf(cr,0.f,1.f))*255.f);
                unsigned int gI=(unsigned)(fsqrt(fclampf(cg,0.f,1.f))*255.f);
                unsigned int bI=(unsigned)(fsqrt(fclampf(cb,0.f,1.f))*255.f);
                row[x*4]=rI; row[x*4+1]=gI; row[x*4+2]=bI; row[x*4+3]=0xff;
                continue;
            }

            // fog toward sky (match render.c)
            float fk=hitT*hitT;
            float fog=(fk*0.0016f)/(1.f+fk*0.0016f);
            cr+=(0.78f-cr)*fog; cg+=(0.86f-cg)*fog; cb+=(0.98f-cb)*fog;

            unsigned int rI=(unsigned)(fsqrt(fclampf(cr,0.f,1.f))*255.f);
            unsigned int gI=(unsigned)(fsqrt(fclampf(cg,0.f,1.f))*255.f);
            unsigned int bI=(unsigned)(fsqrt(fclampf(cb,0.f,1.f))*255.f);
            row[x*4]=rI; row[x*4+1]=gI; row[x*4+2]=bI; row[x*4+3]=0xff;
        }
    }
}
