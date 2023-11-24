#include "cnrt.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif
void foo(int bin_size, uint32_t *A_row, uint32_t *A_col,
                        uint32_t *B_row, uint32_t *B_col, float *A_val, float *B_val, uint32_t *C_row, float *C_val, int *sv);
#ifdef __cplusplus
}
#endif

uint32_t readfile(uint32_t *&A_row, uint32_t *&A_col, float *&A_val) {
    std::fstream f("m.txt");
    std::vector<uint32_t> row, col;
    std::vector<float> val;
    uint32_t r, c;
    float v;
    while (f >> r >> c >> v) {
        row.push_back(r);
        col.push_back(c);
        val.push_back(v);
    }
    auto s = row.size();
    A_row = new uint32_t[s];
    A_col = new uint32_t[s];
    A_val = new float[s];
    uint32_t i = 0;
    for (const auto& temp : row)
        A_row[i++] = temp;
    i = 0;
    
    for (const auto& temp : col)
        A_col[i++] = temp;
    i = 0;

    for (const auto& temp : val)
        A_val[i++] = temp * 100;

    row.clear();
    col.clear();
    val.clear();
    return s;
}

int main() {
    int val = 0;
    cnrtDim3_t dim = {16, 32, 32};
    cnrtFunctionType_t ktype = CNRT_FUNC_TYPE_UNION4;

    struct timeval start, end;
    float time_use;

    CNRT_CHECK(cnrtInit(0));
    cnrtDev_t dev;
    CNRT_CHECK(cnrtGetDeviceHandle(&dev, 0));
    CNRT_CHECK(cnrtSetCurrentDevice(dev));

    cnrtKernelInitParam_t init_param;
    CNRT_CHECK(cnrtCreateKernelInitParam(&init_param));
    CNRT_CHECK(cnrtInitKernelMemory((const void*)foo, init_param));

    uint32_t *A_row, *A_col;
    float *A_val;
    uint32_t len = readfile(A_row, A_col, A_val);

    uint32_t *d_row1 = NULL, *d_row2 = NULL, *d_col1 = NULL, *d_col2 = NULL;
    float *d_val1 = NULL, *d_val2 = NULL;

    uint32_t *C_row, *C_col;
    float *C_val;
    uint32_t *d_outrow = NULL, *d_outcol = NULL;
    float *d_outval = NULL;
    int *dsv = NULL;
    //std::cout << "len = " << len << std::endl;

    CNRT_CHECK(cnrtMalloc((void **)&d_row1, len * sizeof(uint32_t)));
    CNRT_CHECK(cnrtMalloc((void **)&d_row2, len * sizeof(uint32_t)));
    CNRT_CHECK(cnrtMalloc((void **)&d_col1, len * sizeof(uint32_t)));
    CNRT_CHECK(cnrtMalloc((void **)&d_col2, len * sizeof(uint32_t)));
    CNRT_CHECK(cnrtMalloc((void **)&d_val1, len * sizeof(float)));
    CNRT_CHECK(cnrtMalloc((void **)&d_val2, len * sizeof(float)));
    CNRT_CHECK(cnrtMalloc((void **)&d_outrow, len * sizeof(uint32_t)));
    CNRT_CHECK(cnrtMalloc((void **)&d_outcol, len * sizeof(uint32_t)));
    CNRT_CHECK(cnrtMalloc((void **)&d_outval, len * sizeof(float)));
    CNRT_CHECK(cnrtMalloc((void **)&dsv, len * len * sizeof(int)));

    CNRT_CHECK(cnrtMemcpy(d_row1, A_row, sizeof(uint32_t) * len, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_col1, A_col, sizeof(uint32_t) * len, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_row2, A_row, sizeof(uint32_t) * len, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_col2, A_col, sizeof(uint32_t) * len, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_val1, A_val, sizeof(float) * len, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_val2, A_val, sizeof(float) * len, CNRT_MEM_TRANS_DIR_HOST2DEV));

    cnrtQueue_t pQueue;
    CNRT_CHECK(cnrtCreateQueue(&pQueue));
    cnrtKernelParamsBuffer_t params;
    CNRT_CHECK(cnrtGetKernelParamsBuffer(&params));

    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params, &len, sizeof(uint32_t)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params, &d_row1, sizeof(uint32_t*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params, &d_col1, sizeof(uint32_t*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params, &d_row2, sizeof(uint32_t*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params, &d_col2, sizeof(uint32_t*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params, &d_val1, sizeof(float*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params, &d_val2, sizeof(float*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params, &d_outrow, sizeof(uint32_t*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params, &d_outval, sizeof(float*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params, &dsv, sizeof(int**)));

    gettimeofday(&start, NULL);
    CNRT_CHECK(cnrtInvokeKernel_V3((void *)&foo, init_param, dim, params, ktype, pQueue, NULL));
    gettimeofday(&end, NULL);
    time_use = ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) / (1000 * 1000.0);

    std::cout << "scatter vector (origin) consumes " << time_use << " ms on MLU." << std::endl;

    CNRT_CHECK(cnrtFree(d_col1));
    CNRT_CHECK(cnrtFree(d_col2));
    CNRT_CHECK(cnrtFree(d_row1));
    CNRT_CHECK(cnrtFree(d_row2));
    CNRT_CHECK(cnrtFree(d_val1));
    CNRT_CHECK(cnrtFree(d_val2));
    CNRT_CHECK(cnrtFree(d_outrow));
    CNRT_CHECK(cnrtFree(d_outval));
    CNRT_CHECK(cnrtFree(dsv));
    free(A_row);
    free(A_col);
    free(A_val);
    return 0;
}