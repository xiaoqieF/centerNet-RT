#ifndef CENTERNET_TRACK_MATCHING_H
#define CENTERNET_TRACK_MATCHING_H

#include <memory>
#include <vector>

#include "datatype.h"
#include "strack.h"

namespace botsort {
namespace matching {
using STrackList = std::vector<std::shared_ptr<STrack>>;

CostMatrix iouDistance(const STrackList& tracks, const STrackList& detections);

void fuseScore(CostMatrix& cost_matrix, const STrackList& detections);

AssociationData lineraAssignment(CostMatrix& cost_matrix, float thresh);

float iou(const std::vector<float>& tlwh_a, const std::vector<float>& tlwh_b);

} // namespace matching
} // namespace botsort

#endif