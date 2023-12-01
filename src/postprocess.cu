#include "centerutils.h"
#include "config.h"
#include "postprocess.h"

namespace centernet {

dim3 cudaGridSize(uint n) {
    uint k = (n - 1) / config::BLOCK + 1;
    uint x = k;
    uint y = 1;
    if (x > 65535) {
        x = ceil(sqrt(x));
        y = (n - 1) / (x * config::BLOCK) + 1;
    }
    dim3 d = {x, y, 1};
    return d;
}

// 从 hm, wh, reg 三张特征图中计算出目标 bbox
__global__ void forwardKernel(const float *hm,
                              const float *wh,
                              const float *reg,
                              float *output,
                              const int w,
                              const int h,
                              const int classes,
                              const int kernel_size,
                              const float vis_thresh) {
    int idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (idx >= w * h * classes) return;
    int padding = (kernel_size - 1) / 2;
    int offset = -padding;
    int stride = w * h;
    int grid_x = idx % w;
    int grid_y = (idx / w) % h;
    int cls = idx / w / h;
    int l, m;
    int reg_index = idx - cls * stride;
    float c_x, c_y;
    if (hm[idx] > vis_thresh) {
        float max = -1;
        int max_index = 0;
        for (l = 0; l < kernel_size; ++l) {
            for (m = 0; m < kernel_size; ++m) {
                int cur_x = offset + l + grid_x;
                int cur_y = offset + m + grid_y;
                int cur_index = cur_y * w + cur_x + stride * cls;
                int valid = (cur_x >= 0 && cur_x < w && cur_y >= 0 && cur_y < h);
                float val = (valid != 0) ? hm[cur_index] : -1;
                max_index = (val > max) ? cur_index : max_index;
                max = (val > max) ? val : max;
            }
        }
        if (idx == max_index) {
            int res_count = (int)atomicAdd(output, 1);
            char *data = (char *)output + sizeof(float) + res_count * sizeof(util::Detection);
            util::Detection *det = (util::Detection *)(data);
            c_x = grid_x + reg[reg_index];
            c_y = grid_y + reg[reg_index + stride];
            det->box.x1 = (c_x - wh[reg_index] / 2) * 4;
            det->box.y1 = (c_y - wh[reg_index + stride] / 2) * 4;
            det->box.x2 = (c_x + wh[reg_index] / 2) * 4;
            det->box.y2 = (c_y + wh[reg_index + stride] / 2) * 4;
            det->class_id = cls;
            det->prob = hm[idx];
        }
    }
}
void centerNetPostProcess(const float *hm,
                          const float *wh,
                          const float *reg,
                          float *output,
                          const int w,
                          const int h,
                          const int classes,
                          const int kernel_size,
                          const float vis_thresh) {
    uint num = w * h * classes;
    forwardKernel<<<cudaGridSize(num), config::BLOCK>>>(hm, wh, reg, output, w, h, classes,
                                                        kernel_size, vis_thresh);
}
} // namespace centernet
