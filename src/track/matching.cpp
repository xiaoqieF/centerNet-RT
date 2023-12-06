#include "matching.h"

#include <iostream>

#include "lapjv.h"

namespace botsort {
namespace matching {
namespace detail {
double lapjv(CostMatrix& cost,
             std::vector<int>& rowsol,
             std::vector<int>& colsol,
             bool extend_cost,
             float cost_limit,
             bool return_cost = true) {
    std::vector<std::vector<float>> cost_c;

    for (Eigen::Index i = 0; i < cost.rows(); i++) {
        std::vector<float> row;
        for (Eigen::Index j = 0; j < cost.cols(); j++) {
            row.push_back(cost(i, j));
        }
        cost_c.push_back(row);
    }

    std::vector<std::vector<float>> cost_c_extended;

    int n_rows = static_cast<int>(cost.rows());
    int n_cols = static_cast<int>(cost.cols());
    rowsol.resize(n_rows);
    colsol.resize(n_cols);

    int n = 0;
    if (n_rows == n_cols) {
        n = n_rows;
    } else {
        if (!extend_cost) {
            std::cout << "set extend_cost=True" << std::endl;
            exit(0);
        }
    }

    if (extend_cost || cost_limit < LONG_MAX) {
        n = n_rows + n_cols;
        cost_c_extended.resize(n);
        for (int i = 0; i < cost_c_extended.size(); i++) cost_c_extended[i].resize(n);

        if (cost_limit < LONG_MAX) {
            for (int i = 0; i < cost_c_extended.size(); i++) {
                for (int j = 0; j < cost_c_extended[i].size(); j++) {
                    cost_c_extended[i][j] = cost_limit / 2.0;
                }
            }
        } else {
            float cost_max = -1;
            for (int i = 0; i < cost_c.size(); i++) {
                for (int j = 0; j < cost_c[i].size(); j++) {
                    if (cost_c[i][j] > cost_max) cost_max = cost_c[i][j];
                }
            }
            for (int i = 0; i < cost_c_extended.size(); i++) {
                for (int j = 0; j < cost_c_extended[i].size(); j++) {
                    cost_c_extended[i][j] = cost_max + 1;
                }
            }
        }

        for (int i = n_rows; i < cost_c_extended.size(); i++) {
            for (int j = n_cols; j < cost_c_extended[i].size(); j++) {
                cost_c_extended[i][j] = 0;
            }
        }
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
                cost_c_extended[i][j] = cost_c[i][j];
            }
        }

        cost_c.clear();
        cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
    }

    double** cost_ptr;
    cost_ptr = new double*[n];
    for (int i = 0; i < n; i++) cost_ptr[i] = new double[n];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cost_ptr[i][j] = cost_c[i][j];
        }
    }

    int* x_c = new int[n];
    int* y_c = new int[n];

    int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
    if (ret != 0) {
        std::cout << "Calculate Wrong!" << std::endl;
        exit(0);
    }

    double opt = 0.0;

    if (n != n_rows) {
        for (int i = 0; i < n; i++) {
            if (x_c[i] >= n_cols) x_c[i] = -1;
            if (y_c[i] >= n_rows) y_c[i] = -1;
        }
        for (int i = 0; i < n_rows; i++) {
            rowsol[i] = x_c[i];
        }
        for (int i = 0; i < n_cols; i++) {
            colsol[i] = y_c[i];
        }

        if (return_cost) {
            for (int i = 0; i < rowsol.size(); i++) {
                if (rowsol[i] != -1) {
                    opt += cost_ptr[i][rowsol[i]];
                }
            }
        }
    } else if (return_cost) {
        for (int i = 0; i < rowsol.size(); i++) {
            opt += cost_ptr[i][rowsol[i]];
        }
    }

    for (int i = 0; i < n; i++) {
        delete[] cost_ptr[i];
    }
    delete[] cost_ptr;
    delete[] x_c;
    delete[] y_c;

    return opt;
}
} // namespace detail

CostMatrix iouDistance(const STrackList& tracks, const STrackList& detections) {
    size_t num_tracks = tracks.size();
    size_t num_detections = detections.size();
    CostMatrix cost_matrix = Eigen::MatrixXf::Zero(num_tracks, num_detections);

    for (int i = 0; i < num_tracks; ++i) {
        for (int j = 0; j < num_detections; ++j) {
            cost_matrix(i, j) = 1.0f - iou(tracks[i]->tlwh(), detections[j]->tlwh());
        }
    }
    return cost_matrix;
}

void fuseScore(CostMatrix& cost_matrix, const STrackList& detections) {
    if (cost_matrix.rows() == 0 || cost_matrix.cols() == 0) {
        return;
    }
    for (auto i = 0; i < cost_matrix.rows(); ++i) {
        for (auto j = 0; j < cost_matrix.cols(); ++j) {
            cost_matrix(i, j) = 1.0f - (1.0f - cost_matrix(i, j)) * detections[j]->score();
        }
    }
}

AssociationData lineraAssignment(CostMatrix& cost_matrix, float thresh) {
    AssociationData res;
    if (cost_matrix.size() == 0) {
        for (auto i = 0; i < cost_matrix.rows(); ++i) {
            res.unmatched_track_indices.push_back(i);
        }
        for (auto i = 0; i < cost_matrix.cols(); ++i) {
            res.unmatched_det_indices.push_back(i);
        }
        return res;
    }
    std::vector<int> rowsol, colsol;
    double total_cost = detail::lapjv(cost_matrix, rowsol, colsol, true, thresh);

    for (int i = 0; i < rowsol.size(); i++) {
        if (rowsol[i] >= 0) {
            res.matches.emplace_back(i, rowsol[i]);
        } else {
            res.unmatched_track_indices.emplace_back(i);
        }
    }

    for (int i = 0; i < colsol.size(); i++) {
        if (colsol[i] < 0) {
            res.unmatched_det_indices.emplace_back(i);
        }
    }

    return res;
}

float iou(const std::vector<float>& tlwh_a, const std::vector<float>& tlwh_b) {
    float left = std::max(tlwh_a[0], tlwh_b[0]);
    float top = std::max(tlwh_a[1], tlwh_b[1]);
    float right = std::min(tlwh_a[0] + tlwh_a[2], tlwh_b[0] + tlwh_b[2]);
    float bottom = std::min(tlwh_a[1] + tlwh_a[3], tlwh_b[1] + tlwh_b[3]);
    float area_inter = std::max(right - left + 1, 0.0f) * std::max(bottom - top + 1, 0.0f);
    float area_a = (tlwh_a[2] + 1) * (tlwh_a[3] + 1);
    float area_b = (tlwh_b[2] + 1) * (tlwh_b[3] + 1);
    return area_inter / (area_a + area_b - area_inter);
}

} // namespace matching

} // namespace botsort
