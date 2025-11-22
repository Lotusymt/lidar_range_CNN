#pragma once
#include "Types.hpp"
#include <vector>

namespace slam {

class LoopClosureManager {
public:
    LoopClosureManager(float similarity_threshold, int min_separation);

    // Add a new keyframe embedding
    void addKeyframe(int keyframe_id, const std::vector<float>& embedding);

    // Given current embedding, find candidate past keyframes
    std::vector<LoopClosureCandidate> findCandidates(const std::vector<float>& embedding,
                                                     int current_keyframe_id) const;

private:
    struct KeyframeEntry {
        int id;
        std::vector<float> embedding;
    };

    std::vector<KeyframeEntry> keyframes_;
    float similarity_threshold_;
    int min_separation_;  // avoid matching to very recent frames

    static double cosineSimilarity(const std::vector<float>& a,
                                   const std::vector<float>& b);
};

} // namespace slam
