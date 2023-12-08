#include "kalmanfilter.h"

namespace botsort {
KalmanFilter::KalmanFilter(double dt) {
    //  A = [[1,0,0,0,1,0,0,0],   H = [[1,0,0,0,0,0,0,0],
    //       [0,1,0,0,0,1,0,0],        [0,1,0,0,0,0,0,0],
    //       [0,0,1,0,0,0,1,0],        [0,0,1,0,0,0,0,0],
    //       [0,0,0,1,0,0,0,1],        [0,0,0,1,0,0,0,0]]
    //       [0,0,0,0,1,0,0,0],
    //       [0,0,0,0,0,1,0,0],
    //       [0,0,0,0,0,0,1,0],
    //       [0,0,0,0,0,0,0,1]]
    measure_mat_.setIdentity();
    state_trans_mat_.setIdentity();
    for (Eigen::Index i = 0; i < 4; ++i) {
        state_trans_mat_(i, i + 4) = static_cast<float>(dt);
    }
}

std::pair<KalmanFilter::StateSpaceVec, KalmanFilter::ErrorCovMatrix> KalmanFilter::init(
    const DetVec& measurement) {
    StateSpaceVec mean_state_space;
    mean_state_space.head(4) = measurement.head(4);
    mean_state_space.tail(4).setZero();

    float w = measurement(2), h = measurement(3);
    StateSpaceVec std_dev;
    std_dev.head(4) = 2 * std_weight_position_ * (Eigen::Vector4f(w, h, w, h).array());
    std_dev.tail(4) = 10 * std_weight_velocity_ * (Eigen::Vector4f(w, h, w, h).array());
    ErrorCovMatrix covariance = std_dev.array().square().matrix().asDiagonal();
    return {mean_state_space, covariance};
}

void KalmanFilter::predict(StateSpaceVec& mean, ErrorCovMatrix& covariance) {
    // 构建过程噪声 Q
    Eigen::VectorXf std_combined;
    std_combined.resize(kStateSpaceDim);
    std_combined << mean(2), mean(3), mean(2), mean(3), mean(2), mean(3), mean(2), mean(3);
    std_combined.head(4).array() *= std_weight_position_;
    std_combined.tail(4).array() *= std_weight_velocity_;
    ErrorCovMatrix motion_cov = std_combined.array().square().matrix().asDiagonal();

    // 1. X_k = AX_{k-1}
    mean = state_trans_mat_ * mean;
    // 2. P_k^- = AP_{k-1}A^T + Q_k
    covariance = state_trans_mat_ * covariance * state_trans_mat_.transpose() + motion_cov;
}

std::pair<KalmanFilter::MeasSpaceVec, KalmanFilter::MeasCovMatrix> KalmanFilter::project(
    // 构建测量噪声 R
    const StateSpaceVec& mean,
    const ErrorCovMatrix& covariance) {
    // 测量噪声尽可能小
    MeasSpaceVec innovation_cov =
        std_weight_position_ *
        Eigen::Vector4f(mean(2), mean(3), mean(2), mean(3)).array().square().matrix() * 0.01;
    MeasCovMatrix innovation_cov_diag = innovation_cov.asDiagonal();

    // HX_k^-
    MeasSpaceVec mean_projected = measure_mat_ * mean;
    // HP_k^-H^T + R
    MeasCovMatrix covariance_projected =
        measure_mat_ * covariance * measure_mat_.transpose() + innovation_cov_diag;
    return {mean_projected, covariance_projected};
}

std::pair<KalmanFilter::StateSpaceVec, KalmanFilter::ErrorCovMatrix> KalmanFilter::update(
    const StateSpaceVec& mean, const ErrorCovMatrix& covariance, const DetVec& measurement) {
    auto [projected_mean, projected_covariance] = project(mean, covariance);

    // 3. 利用 Cholesky 分解求解卡尔曼增益
    // K_k (HP_k^-H^T + R) = P_k^-H^T  ===> (HP_k^-H^T + R) K_k^T = (P_k^-H^T)^T
    // 即求解方程组 Ax = b,
    Eigen::Matrix<float, kMeasureSpaceDim, kStateSpaceDim> b =
        (covariance * measure_mat_.transpose()).transpose();
    Eigen::Matrix<float, kStateSpaceDim, kMeasureSpaceDim> kalman_gain =
        projected_covariance.llt().solve(b).transpose();
    // 4. X_k = X_k^- + K_k (z_k - HX_k^-)
    StateSpaceVec mean_updated = mean + kalman_gain * (measurement - projected_mean);
    // 这个地方和公式不一样，作者说是采用另一种写法: P_k = P_k^- - K_k(HP_k^-H^T + R)K_k^T
    // 5. P_k = P_k^- - K_kHP_k^-
    ErrorCovMatrix covariance_updated =
        covariance - kalman_gain * projected_covariance * kalman_gain.transpose();
    return {mean_updated, covariance_updated};
}
} // namespace botsort
