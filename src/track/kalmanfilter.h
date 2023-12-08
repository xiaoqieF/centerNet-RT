#ifndef CENTERRT_TRACK_KALMANFILTER_H
#define CENTERRT_TRACK_KALMANFILTER_H

#include <cstdint>
#include <eigen3/Eigen/Dense>

#include "datatype.h"

namespace botsort {
class KalmanFilter {
public:
    using StateSpaceVec = Eigen::Matrix<float, kStateSpaceDim, 1>;
    using ErrorCovMatrix = Eigen::Matrix<float, kStateSpaceDim, kStateSpaceDim>;
    using MeasSpaceVec = Eigen::Matrix<float, kMeasureSpaceDim, 1>;
    using MeasCovMatrix = Eigen::Matrix<float, kMeasureSpaceDim, kMeasureSpaceDim>;

    static constexpr double chi2inv95[10] = {0,      3.8415, 5.9915, 7.8147, 9.4877,
                                             11.070, 12.592, 14.067, 15.507, 16.919};
    explicit KalmanFilter(double dt);
    // 通过第一个检测结果初始化 Kalman 滤波器，返回初始状态估计值和误差协方差矩阵
    std::pair<StateSpaceVec, ErrorCovMatrix> init(const DetVec& measurement);
    void predict(StateSpaceVec& mean, ErrorCovMatrix& conv);
    std::pair<MeasSpaceVec, MeasCovMatrix> project(const StateSpaceVec& mean,
                                                   const ErrorCovMatrix& covariance);
    std::pair<StateSpaceVec, ErrorCovMatrix> update(const StateSpaceVec& mean,
                                                    const ErrorCovMatrix& covariance,
                                                    const DetVec& measurement);

private:
    static constexpr float std_weight_position_ = 1.0 / 20;
    static constexpr float std_weight_velocity_ = 1.0 / 160;

    Eigen::Matrix<float, kStateSpaceDim, kStateSpaceDim> state_trans_mat_; // 状态转移矩阵 A
    Eigen::Matrix<float, kMeasureSpaceDim, kStateSpaceDim> measure_mat_;   // 观测矩阵 H
};
} // namespace botsort

#endif
