// anim.h — animation engine: global clock, no-slip gait, herd kinematics.
// Everything is an analytic function of time t (stateless between frames).
#ifndef ANIM_H
#define ANIM_H

extern const float VG;   // ground / herd reference speed
extern float GT;         // global time (chewing, jaw, cloud drift)

// herd member kinematics: x(t) = x0 + D*sin(u + a*sin u), u = w t + ph (skewed sine)
// speed(t) = VG + D*w*cos(u + a*sin u)*(1 + a*cos u)
typedef struct { float x, dist, speed, run; } Kin;
Kin kin(float t, float x0, float D, float w, float ph, float a);

// piecewise gait: stance (phase<ds) moves the foot linearly in traveled distance
// -> exact rolling contact when cadence K = pi*ds/A (A: half-stride, body space).
// Returns foot x-offset in [-A,A]; sets lift 0..1 (0 during stance).
float gaitFoot(float ph, float ds, float A, float *lift);

// advance the animation clock (sets GT)
void animTick(float t);

#endif
