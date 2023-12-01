#ifndef CENTERRT_CONFIG_H
#define CENTERRT_CONFIG_H

namespace centernet {
namespace config {
constexpr inline int input_w = 512;
constexpr inline int input_h = 512;
constexpr inline int channel = 3;
constexpr inline int class_num = 1;
constexpr inline float vis_thresh = 0.3;
constexpr inline int kernel_size = 3;
constexpr inline int BLOCK = 512;
const inline char* class_name[] = {"drone"};
} // namespace config

} // namespace centernet

#endif
