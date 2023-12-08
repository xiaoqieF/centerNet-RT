#ifndef CENTERRT_TRACK_DATATYPE_H
#define CENTERRT_TRACK_DATATYPE_H

#include <eigen3/Eigen/Dense>
#include <opencv2/core.hpp>
#include <vector>

namespace botsort {
constexpr uint8_t kDetElements = 4;
constexpr uint8_t kStateSpaceDim = 8;
constexpr uint8_t kMeasureSpaceDim = 4;
using DetVec = Eigen::Matrix<float, kDetElements, 1>;
using HomoGraphyMatrix = Eigen::Matrix<float, 3, 3>;
using CostMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

struct AssociationData {
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_track_indices;
    std::vector<int> unmatched_det_indices;
};

} // namespace botsort

#endif