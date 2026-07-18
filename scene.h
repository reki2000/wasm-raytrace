// scene.h — combined scene renderer: unifies the SDF herd (render.c) and the
// glTF mesh line-up (mesh.c) into a single frame. One primary visibility
// pass picks the nearer of an SDF sphere-march hit and a mesh BVH hit per
// pixel; from there, lighting, shadows, the dielectric/metal fresnel blend
// and the floor-mirror reflection all run over a unified surface (position,
// normal, albedo, material) regardless of which renderer produced it, so an
// SDF dinosaur and a polygon dinosaur can share the screen, shadow each
// other, and show up in each other's floor reflection.
//
// Acrylic refraction is the one place the two representations can't share
// code: the SDF path marches the implicit distance field through the body
// to find the back face, which has no equivalent on a triangle mesh. Mesh
// acrylic hits instead bend the ray by Snell's law at the entry triangle
// and sample the combined scene once along the exit ray (see scene.c).
#ifndef SCENE_H
#define SCENE_H

// Rebuilds both scenes' per-frame state: the SDF herd (animTick + animate)
// and the mesh line-up (skin + BVH + packTris). Must run before sceneRows
// for this t; single-threaded, main instance only (same contract as
// renderPrep / meshPrep).
void scenePrep(float t);

// Renders rows [y0,y1) of the combined frame into fb (RGBA8, w*h). Read-only
// over scene state after scenePrep, so safe to call concurrently for
// disjoint row ranges from multiple threads (same contract as renderRows /
// renderMeshRows).
void sceneRows(float az, float el, float dist, int w, int h, unsigned char *fb, int y0, int y1);

// Convenience: scenePrep + full-frame sceneRows, single-threaded.
void renderScene(float t, float az, float el, float dist, int w, int h, unsigned char *fb);

#endif
