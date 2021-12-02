#pragma warning(disable:4819)

// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// CUDA includes and interop headers
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>      // includes cuda.h and cuda_runtime_api.h

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define MAX(a,b) ((a > b) ? a : b)

#define USE_SIMPLE_FILTER 0

#define MAX_EPSILON_ERROR 5.0f
#define THRESHOLD  0.15f

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "test_10.ppm",
    "test_14.ppm",
    "test_18.ppm",
    "test_22.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_10.ppm",
    "ref_14.ppm",
    "ref_18.ppm",
    "ref_22.ppm",
    NULL
};

const char *image_filename = "test.ppm";
float sigma = 10.0f;
int order = 0;
int nthreads = 64;  // number of threads per block

unsigned int width, height;
unsigned int *h_img = NULL;
unsigned int *d_img = NULL;
unsigned int *d_temp = NULL;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint texid = 0;   // texture

cudaGraphicsResource_t cuda_vbo_resource;

StopWatchInterface *timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
unsigned int frameCount = 0;

int *pArgc = NULL;
char **pArgv = NULL;

bool runBenchmark = false;

const char *sSDKsample = "Gaussian Sample";

extern "C"
void transpose(unsigned int *d_src, unsigned int *d_dest, unsigned int width, int height);

extern "C"
void gaussianFilterRGBA(unsigned int *d_src, unsigned int *d_dest, unsigned int *d_temp, int width, int height, float sigma, int nthreads);

void cleanup();

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "%s (sigma=%4.2f): %3.1f fps", sSDKsample, sigma, ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = ftoi(MAX(ifps, 1.f));
        sdkResetTimer(&timer);
    }
}

void idle()
{
    glutPostRedisplay();
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    checkCudaErrors(cudaFree(d_img));
    checkCudaErrors(cudaFree(d_temp));
}

void reshape(int x, int y)
{
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void initCudaBuffers()
{
    unsigned int size = width * height * sizeof(unsigned int);

    // allocate device memory
    checkCudaErrors(cudaMalloc((void **) &d_img, size));
    checkCudaErrors(cudaMalloc((void **) &d_temp, size));

    checkCudaErrors(cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice));

    sdkCreateTimer(&timer);
}


void initGLBuffers()
{
    // create pixel buffer object to store final image
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, h_img, GL_STREAM_DRAW_ARB);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void
benchmark(int iterations)
{
    // allocate memory for result
    unsigned int *d_result;
    unsigned int size = width * height * sizeof(unsigned int);
    checkCudaErrors(cudaMalloc((void **) &d_result, size));

    // warm-up
    gaussianFilterRGBA(d_img, d_result, d_temp, width, height, sigma, nthreads);

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStartTimer(&timer);

    // execute the kernel
    for (int i = 0; i < iterations; i++)
    {
        gaussianFilterRGBA(d_img, d_result, d_temp, width, height, sigma, nthreads);
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);

    // check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n", (width*height*iterations / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);

    checkCudaErrors(cudaFree(d_result));
}

bool
runSingleTest(const char *ref_file, const char *exec_path)
{
    // allocate memory for result
    int nTotalErrors = 0;
    unsigned int *d_result;
    unsigned int size = width * height * sizeof(unsigned int);
    checkCudaErrors(cudaMalloc((void **) &d_result, size));

    // warm-up
    gaussianFilterRGBA(d_img, d_result, d_temp, width, height, sigma, nthreads);

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStartTimer(&timer);

    gaussianFilterRGBA(d_img, d_result, d_temp, width, height, sigma, nthreads);
    checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("Kernel execution failed");
    sdkStopTimer(&timer);

    unsigned char *h_result = (unsigned char *)malloc(width*height*4);
    checkCudaErrors(cudaMemcpy(h_result, d_result, width*height*4, cudaMemcpyDeviceToHost));

    char dump_file[1024];
    sprintf(dump_file, "test_%02d.ppm", (int)sigma);
    sdkSavePPM4ub(dump_file, h_result, width, height);

    if (!sdkComparePPM(dump_file, sdkFindFilePath(ref_file, exec_path), MAX_EPSILON_ERROR, THRESHOLD, false))
    {
        nTotalErrors++;
    }

    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n", (width*height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);

    checkCudaErrors(cudaFree(d_result));
    free(h_result);

    printf("Summary: %d errors!\n", nTotalErrors);

    printf(nTotalErrors == 0 ? "Test passed\n": "Test failed!\n");
    return (nTotalErrors == 0);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;
    char *ref_file = NULL;

    printf("%s Starting...\n\n", sSDKsample);

    printf("NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "file"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
            fpsLimit = frameCheckNumber;
        }
    }

    // Get the path of the filename
    char *filename;

    if (getCmdLineArgumentString(argc, (const char **) argv, "image", &filename))
    {
        image_filename = filename;
    }

    // load image
    char *image_path = sdkFindFilePath(image_filename, argv[0]);

    if (image_path == NULL)
    {
        fprintf(stderr, "Error unable to find and load image file: '%s'\n", image_filename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPPM4ub(image_path, (unsigned char **)&h_img, &width, &height);

    if (!h_img)
    {
        printf("Error unable to load PPM file: '%s'\n", image_path);
        exit(EXIT_FAILURE);
    }

    printf("Loaded '%s', %d x %d pixels\n", image_path, width, height);

    if (checkCmdLineFlag(argc, (const char **)argv, "threads"))
    {
        nthreads = getCmdLineArgumentInt(argc, (const char **) argv, "threads");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "sigma"))
    {
        sigma = getCmdLineArgumentFloat(argc, (const char **) argv, "sigma");
    }

    int device;
    struct cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    runBenchmark = true;

    findCudaDevice(argc, (const char **)argv);
    

    initCudaBuffers();

    if (ref_file)
    {
        printf("(Automated Testing)\n");
        bool testPassed = runSingleTest(ref_file, argv[0]);

        cleanup();
        exit(testPassed ? EXIT_SUCCESS : EXIT_FAILURE);
    }

    if (runBenchmark)
    {
        printf("(Run Benchmark)\n");
        benchmark(100);

        cleanup();
        exit(EXIT_SUCCESS);
    }

    initGLBuffers();
    glutMainLoop();

    exit(EXIT_SUCCESS);
}
