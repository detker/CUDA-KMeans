#pragma once

#include <iostream>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cassert>

#include "error_utils.h"

#define D_VIZ 3

struct Mat4 {
    float m[16];                                    // 16 elements
    Mat4() { memset(m, 0, 16 * sizeof(float)); }      // default constructor sets all to 0
    static Mat4 identity() { Mat4 M; M.m[0] = M.m[5] = M.m[10] = M.m[15] = 1.0f; return M; } // identity matrix
};

static Mat4 multiply(const Mat4& A, const Mat4& B);

static Mat4 perspective(float fovy, float aspect, float zn, float zf);

static Mat4 translate(float x, float y, float z);

static Mat4 rotateX(float a);

static Mat4 rotateY(float a);

static Mat4 scale(float s);

static GLuint compileShader(GLenum type, const char* src);

__global__ void fillVBOKernel(const double* points, int* assignments, int N, int K, float minx, float maxx, float miny, float maxy, float minz, float maxz, float* outPos, unsigned char* outCol);

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

static void compute_bounds(const std::vector<double>& pts, int N, float& minx, float& maxx, float& miny, float& maxy, float& minz, float& maxz);

int render(const double* points, const double* d_points, int* d_assignments, int N, int K);