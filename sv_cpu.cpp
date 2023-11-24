#include <iostream>
#include <fstream>
#include <vector>
#include <sys/time.h>
#include <cstring>

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
    uint32_t *A_row, *A_col;
    float *A_val;

    struct timeval start, end;
    float time_use;

    auto len = readfile(A_row, A_col, A_val);

    int *sv = new int[len];
    memset(sv, -1, sizeof sv);
    int nnz = 0;
    uint32_t *C_row, *C_col;
    float *C_val;
    C_row = new uint32_t[len];
    C_col = new uint32_t[len];
    C_val = new float[len];
    gettimeofday(&start, NULL);
    for (int i = 0; i < len-1; ++i) {
        for (int a_index = A_row[i]; a_index < A_row[i+1] && a_index < len; ++a_index) {
            auto a = A_val[a_index];
            auto j = A_col[a_index];
            for (int b_index = A_row[j]; j < len-1 && b_index < A_row[j+1] && b_index < len; ++b_index) {
                auto k = A_col[b_index];
                if (k >= len) continue;
                if (sv[k] == -1) {
                    if (nnz >= len) continue;
                    C_col[nnz] = k;
                    C_val[nnz] = a * A_val[b_index];
                    sv[k] = nnz;
                    ++nnz;
                } 
                else {
                    auto p = sv[k];
                    if (p >= len || p < 0) continue;
                    C_col[p] = k;
                    C_val[p] = a * A_val[b_index];
                }
            }
        }
        C_row[i+1] = nnz;
        memset(sv, -1, sizeof sv);
    }
    gettimeofday(&end, NULL);

    time_use = ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) / 1000.0;
    std::cout << "scatter-vector consumes " << time_use << " ms on CPU." << std::endl;

    delete []A_col;
    delete []A_val;
    delete []A_row;
    delete []C_col;
    delete []C_val;
    delete []C_row;
    delete []sv;

    return 0;
}   