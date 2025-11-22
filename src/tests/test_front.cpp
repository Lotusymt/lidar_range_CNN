#include "FrontEnd.hpp"
#include "KittiLoader.hpp"
#include <iostream>

int main() {
    using namespace slam;

    std::string base = "/path/to/KITTI/raw";   // adjust
    std::string seq  = "00";  // example

    KittiLoader loader(base, seq);
    FrontEnd frontend;

    LidarScan scan;
    int frame_idx = 0;

    while (loader.loadNextScan(scan)) {
        PoseDelta edge = frontend.processFrame(scan);

        Pose3D pose = frontend.currentPose();

        std::cout << "Frame " << frame_idx << ": "
                  << "t = [" << pose.t.x() << ", "
                              << pose.t.y() << ", "
                              << pose.t.z() << "], "
                  << "q = [" << pose.q.w() << ", "
                              << pose.q.x() << ", "
                              << pose.q.y() << ", "
                              << pose.q.z() << "]\n";

        ++frame_idx;
    }

    return 0;
}
