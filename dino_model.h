// dino_model.h — the dinosaur herd scene: builds this demo's 3 species
// (geometry + animation parameters) into the renderer's primitive store,
// and provides their procedural surface colors.
#ifndef DINO_MODEL_H
#define DINO_MODEL_H
#include "vec.h"

// rebuild the whole herd for time t: registers prims/eyes, fills the
// instance registry (DB/DPR/DXW/DZW) and sets the sun direction
void animate(float t);

// species albedo (textured), given exclusive instance masks m0/m1/m2
C3 dinoAlbedo(V3 P, V3 N, v4 m0, v4 m1, v4 m2);

#endif
