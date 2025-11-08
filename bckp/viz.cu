#include "viz.cuh"


__global__ void fillVBOKernel(const double* points, int* assignments, int N, int K, float minx, float maxx, float miny, float maxy, float minz, float maxz, float* outPos, unsigned char* outCol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    double px = points[idx * D_VIZ + 0];
    double py = points[idx * D_VIZ + 1];
    double pz = points[idx * D_VIZ + 2];
    // Normalize point positions to [-1, 1] cube
    float nx = (maxx > minx) ? (float)((px - minx) / (maxx - minx) * 2.0 - 1.0) : 0.0f;
    float ny = (maxy > miny) ? (float)((py - miny) / (maxy - miny) * 2.0 - 1.0) : 0.0f;
    float nz = (maxz > minz) ? (float)((pz - minz) / (maxz - minz) * 2.0 - 1.0) : 0.0f;
    outPos[idx * D_VIZ + 0] = nx;
    outPos[idx * D_VIZ + 1] = ny;
    outPos[idx * D_VIZ + 2] = nz;
    int cluster = assignments[idx];
    unsigned char r = (unsigned char)((cluster * 37) % 256);
    unsigned char g = (unsigned char)((cluster * 91) % 256);
    unsigned char b = (unsigned char)((cluster * 53) % 256);
    outCol[idx * (D_VIZ + 1) + 0] = r;
    outCol[idx * (D_VIZ + 1) + 1] = g;
    outCol[idx * (D_VIZ + 1) + 2] = b;
    outCol[idx * (D_VIZ + 1) + 3] = 255; // alpha
}

// multiply two Mat4's A * B and return result (column-major math)
static Mat4 multiply(const Mat4& A, const Mat4& B) {
    Mat4 R;                                          // result matrix
    for (int r = 0;r < 4;r++) {                           // for each row
        for (int c = 0;c < 4;c++) {                       // for each column
            float v = 0.0f;                            // accumulator
            for (int k = 0;k < 4;k++) v += A.m[k * 4 + r] * B.m[c * 4 + k]; // compute dot product (column-major)
            R.m[c * 4 + r] = v;                       // store result
        }
    }
    return R;                                       // return multiplied matrix
}

// construct a perspective projection matrix using fovy radians and aspect ratio
static Mat4 perspective(float fovy, float aspect, float zn, float zf) {
    float f = 1.0f / tanf(fovy * 0.5f);             // focal factor
    Mat4 M;                                         // result matrix
    M.m[0] = f / aspect;                             // fx
    M.m[5] = f;                                      // fy
    M.m[10] = (zf + zn) / (zn - zf);                 // z scaling for depth
    M.m[11] = -1.0f;                                 // perspective division element
    M.m[14] = (2.0f * zf * zn) / (zn - zf);          // z translation term
    return M;                                       // return projection matrix
}

// translation matrix by (x,y,z)
static Mat4 translate(float x, float y, float z) {
    Mat4 M = Mat4::identity();                      // start with identity
    M.m[12] = x;                                    // set translation x (column-major index 12)
    M.m[13] = y;                                    // translation y
    M.m[14] = z;                                    // translation z
    return M;                                       // return translation matrix
}

// rotation around X axis by angle a (radians)
static Mat4 rotateX(float a) {
    Mat4 M = Mat4::identity();                      // identity
    float c = cosf(a), s = sinf(a);                 // compute cos/sin
    M.m[5] = c; M.m[9] = -s;                        // set rotation elements (column-major)
    M.m[6] = s; M.m[10] = c;                        // set rotation elements
    return M;                                       // return rotation matrix
}

// rotation around Y axis by angle a (radians)
static Mat4 rotateY(float a) {
    Mat4 M = Mat4::identity();                      // identity
    float c = cosf(a), s = sinf(a);                 // cos/sin
    M.m[0] = c; M.m[8] = s;                         // set elements
    M.m[2] = -s; M.m[10] = c;                       // set elements
    return M;                                       // return rotation matrix
}

// uniform scale matrix by factor s
static Mat4 scale(float s) {
    Mat4 M = Mat4::identity();                      // identity
    M.m[0] = M.m[5] = M.m[10] = s;                        // scale diagonal
    return M;                                       // return scale matrix
}

// ---------------------------- GLSL shader sources ----------------------------

// vertex shader for points: takes position (vec3) and color (vec4), applies MVP and sets point size
const char* vs_src =
"#version 330 core\n"                                 // GLSL version
"layout(location=0) in vec3 position;\n"              // input attribute 0 = position
"layout(location=1) in vec4 colorIn;\n"               // input attribute 1 = color (RGBA)
"out vec4 vColor;\n"                                  // out variable to fragment shader
"uniform mat4 MVP;\n"                                 // uniform Model-View-Projection matrix
"void main(){ gl_Position = MVP * vec4(position,1.0); vColor = colorIn / 255.0; gl_PointSize = 3.0; }\n";
// note: colorIn is bytes packed into normalized attribute; dividing by 255 converts to [0,1]

// fragment shader for points: output interpolated color
const char* fs_src =
"#version 330 core\n"
"in vec4 vColor; out vec4 fragColor; void main(){ fragColor = vColor; }\n";

// vertex shader for wireframe cube lines; position only
const char* vs_line =
"#version 330 core\nlayout(location=0) in vec3 position; uniform mat4 MVP; void main(){ gl_Position = MVP * vec4(position,1.0); }\n";

// fragment shader for lines; single uniform color
const char* fs_line =
"#version 330 core\nout vec4 fragColor; uniform vec3 uColor; void main(){ fragColor = vec4(uColor,1.0); }\n";

static GLuint compileShader(GLenum type, const char* src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE)
    {
        char buffer[512];
        glGetShaderInfoLog(shader, 512, nullptr, buffer);
        ERR(buffer);
    }
    return shader;
}

// ---------------------------- Camera controls ----------------------------
static double lastX = 0.0, lastY = 0.0;
static bool dragging = false;
static float yaw = 0.5f, pitch = -0.3f;
static float distanceCamera = 3.0f;

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (dragging)
    {
        double dx = xpos - lastX;
        double dy = ypos - lastY;
        yaw += (float)(dx * 0.007f);
        pitch += (float)(dy * 0.007f);
        if (pitch > 1.4f) pitch = 1.4f;
        if (pitch < -1.4f) pitch = -1.4f;
    }
    lastX = xpos;
    lastY = ypos;
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            dragging = true;
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            lastX = xpos;
            lastY = ypos;
        }
        else if (action == GLFW_RELEASE)
        {
            dragging = false;
        }
    }
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    distanceCamera *= (yoffset > 0) ? 0.9f : 1.1f;
    if (distanceCamera < 0.3f) distanceCamera = 0.3f;
    if (distanceCamera > 50.0f) distanceCamera = 50.0f;
}


static void compute_bounds(const std::vector<double>& pts, int N, float& minx, float& maxx, float& miny, float& maxy, float& minz, float& maxz) {
    if (pts.empty()) return;
    minx = (float)pts[0];
    maxx = (float)pts[0];
    miny = (float)pts[1];
    maxy = (float)pts[1];
    minz = (float)pts[2];
    maxz = (float)pts[2];
    for (size_t i = 1;i < N;++i)
    {
        minx = fminf(minx, (float)pts[i * D_VIZ]); maxx = fmaxf(maxx, (float)pts[i * D_VIZ]);
        miny = fminf(miny, (float)pts[i * D_VIZ + 1]); maxy = fmaxf(maxy, (float)pts[i * D_VIZ + 1]);
        minz = fminf(minz, (float)pts[i * D_VIZ + 2]); maxz = fmaxf(maxz, (float)pts[i * D_VIZ + 2]);
    }

    float px = fmaxf(1e-6f, (maxx - minx) * 0.01f);
    float py = fmaxf(1e-6f, (maxy - miny) * 0.01f);
    float pz = fmaxf(1e-6f, (maxz - minz) * 0.01f);
    minx -= px; maxx += px;
    miny -= py; maxy += py;
    minz -= pz; maxz += pz;
}


int render(const double* points, const double* d_points, int* d_assignments, int N, int K) {

    float minx, maxx, miny, maxy, minz, maxz;
    const std::vector<double> vec_points(points, points + N * D_VIZ);
    compute_bounds(vec_points, N, minx, maxx, miny, maxy, minz, maxz);

    if (!glfwInit()) {
        ERR("Failed to innit GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow* window = glfwCreateWindow(1200, 800, "KMeans Visualization", nullptr, nullptr);
    if (!window)
    {
        ERR("glfwCreateWindow failed");
    }
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        ERR("Failed to initialize GLEW");
    }

    std::cout << "OpenGL + CUDA initialized!\n";

    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);

    GLuint vboPos = 0, vboCol = 0, vao = 0, vaoLines = 0, vboLines = 0;
    glGenBuffers(1, &vboPos);
    glGenBuffers(1, &vboCol);
    glGenVertexArrays(1, &vao);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vboPos);
    glBufferData(GL_ARRAY_BUFFER, N * D_VIZ * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, vboCol);
    glBufferData(GL_ARRAY_BUFFER, N * (D_VIZ + 1) * sizeof(GLubyte), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_FALSE, 0, (void *)0);

    glBindVertexArray(0);

    {
        float cubeVerts[] = {
            -1,-1,-1,  +1,-1,-1,  +1,-1,-1,  +1,+1,-1,
            +1,+1,-1,  -1,+1,-1,  -1,+1,-1,  -1,-1,-1,
            -1,-1,+1,  +1,-1,+1,  +1,-1,+1,  +1,+1,+1,
            +1,+1,+1,  -1,+1,+1,  -1,+1,+1,  -1,-1,+1,
            -1,-1,-1,  -1,-1,+1,  +1,-1,-1,  +1,-1,+1,
            +1,+1,-1,  +1,+1,+1,  -1,+1,-1,  -1,+1,+1
        }; // 24 vertices total, each triple is a vertex coordinate
        glGenVertexArrays(1, &vaoLines);                  // create VAO for line geometry
        glGenBuffers(1, &vboLines);                       // create VBO for cube lines
        glBindVertexArray(vaoLines);                      // bind line VAO
        glBindBuffer(GL_ARRAY_BUFFER, vboLines);          // bind line VBO
        glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVerts), cubeVerts, GL_STATIC_DRAW); // upload cube verts
        glEnableVertexAttribArray(0);                     // enable attribute 0 (reused for lines)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); // layout: vec3 position
        glBindVertexArray(0);                             // unbind VAO
    }

    cudaGraphicsResource* cudaPosRes = nullptr, * cudaColRes = nullptr;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPosRes, vboPos, cudaGraphicsRegisterFlagsNone));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaColRes, vboCol, cudaGraphicsRegisterFlagsNone));

    GLuint vs = compileShader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fs_src);
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    GLuint vsL = compileShader(GL_VERTEX_SHADER, vs_line);
    GLuint fsL = compileShader(GL_FRAGMENT_SHADER, fs_line);
    GLuint programL = glCreateProgram();
    glAttachShader(programL, vsL);
    glAttachShader(programL, fsL);
    glLinkProgram(programL);

    GLint loc = glGetAttribLocation(program, "colorIn");
    printf("colorIn location = %d\n", loc);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);

    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPosRes, 0));
    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaColRes, 0));
    float* d_vboPos = nullptr;
    unsigned char* d_vboCol = nullptr;
    size_t posSize = 0, colSize = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_vboPos, &posSize, cudaPosRes));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_vboCol, &colSize, cudaColRes));
    assert(posSize >= (size_t)N * D_VIZ * sizeof(float));
    assert(colSize >= (size_t)N * (D_VIZ + 1) * sizeof(unsigned char));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // kernel launch
    fillVBOKernel << < blocksPerGrid, threadsPerBlock >> > (d_points, d_assignments, N, K, minx, maxx, miny, maxy, minz, maxz, d_vboPos, d_vboCol);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaColRes, 0));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPosRes, 0));

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        float aspect = (float)width / (float)height;
        Mat4 proj = perspective(45.0f * (3.14159265f / 180.0f), aspect, 0.1f, 100.0f);
        Mat4 T = translate(0.0f, 0.0f, -distanceCamera);
        Mat4 R = multiply(rotateX(pitch), rotateY(yaw));
        Mat4 MVP = multiply(proj, multiply(T, R));

        glViewport(0, 0, width, height);
        glClearColor(0.06f, 0.06f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(program);
        GLuint locMVP = glGetUniformLocation(program, "MVP");
        glUniformMatrix4fv(locMVP, 1, GL_FALSE, MVP.m);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, N);
        glBindVertexArray(0);

        glUseProgram(programL);
        GLuint locMVPL = glGetUniformLocation(programL, "MVP");
        glUniformMatrix4fv(locMVPL, 1, GL_FALSE, MVP.m);
        GLuint locColor = glGetUniformLocation(programL, "uColor");
        glUniform3f(locColor, 1.0f, 0.2f, 0.8f);
        glBindVertexArray(vaoLines);
        glDrawArrays(GL_LINES, 0, 24);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
    }

    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaPosRes));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaColRes));

    glDeleteBuffers(1, &vboPos);
    glDeleteBuffers(1, &vboCol);
    glDeleteVertexArrays(1, &vao);
    glDeleteVertexArrays(1, &vaoLines);
    glDeleteBuffers(1, &vboLines);

    glDeleteProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);
    glDeleteProgram(programL);
    glDeleteShader(vsL);
    glDeleteShader(fsL);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}