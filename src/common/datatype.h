#ifndef CENTERRT_COMMON_DATATYPE_H
#define CENTERRT_COMMON_DATATYPE_H

namespace common {
struct Box {
    float x1;
    float y1;
    float x2;
    float y2;
};

struct Detection {
    Box box;
    int class_id;
    float prob;
};
} // namespace common

#endif