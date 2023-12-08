#ifndef CENTERRT_DET_POSTPROCESS_H
#define CENTERRT_DET_POSTPROCESS_H

namespace centernet {
void centerNetPostProcess(const float *hm,
                          const float *wh,
                          const float *reg,
                          float *output,
                          const int w,
                          const int h,
                          const int classes,
                          const int kernel_size,
                          const float vis_thresh);
} // namespace centernet

#endif