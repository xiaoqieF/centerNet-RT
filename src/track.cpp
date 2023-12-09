#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "common/datatype.h"
#include "det/centernet.h"
#include "det/centerutils.h"
#include "det/config.h"
#include "track/botsort.h"

std::vector<float> prepareImage(cv::Mat& img) {
    int channel = centernet::config::channel;
    int input_w = centernet::config::input_w;
    int input_h = centernet::config::input_h;
    float scale = cv::min(float(input_w) / img.cols, float(input_h) / img.rows);
    auto scale_size = cv::Size(img.cols * scale, img.rows * scale);
    cv::Mat resized;
    cv::resize(img, resized, scale_size, 0, 0);

    cv::Mat cropped(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Rect rect((input_w - scale_size.width) / 2, (input_h - scale_size.height) / 2,
                  scale_size.width, scale_size.height);
    resized.copyTo(cropped(rect));

    cv::Mat img_float;
    cropped.convertTo(img_float, CV_32FC3, 1. / 255);

    std::vector<float> res(input_h * input_w * channel);
    auto data = res.data();
    int channel_len = input_h * input_w;
    // HWC -> CHW
    std::vector<cv::Mat> input_channels{
        cv::Mat(input_h, input_w, CV_32FC1, data + channel_len * 2),
        cv::Mat(input_h, input_w, CV_32FC1, data + channel_len * 1),
        cv::Mat(input_h, input_w, CV_32FC1, data + channel_len * 0)};
    cv::split(img_float, input_channels);
    return res;
}

void plot_tracks(cv::Mat& frame,
                 std::vector<common::Detection>& detections,
                 std::vector<std::shared_ptr<botsort::STrack>>& tracks,
                 int fps) {
    static std::map<int, cv::Scalar> track_colors;
    cv::Scalar detection_color = cv::Scalar(255, 0, 0);
    for (const auto& det : detections) {
        cv::Rect rect(det.box.x1, det.box.y1, det.box.x2 - det.box.x1, det.box.y2 - det.box.y1);
        cv::rectangle(frame, rect, detection_color, 1);
    }

    for (const std::shared_ptr<botsort::STrack>& track : tracks) {
        std::vector<float> bbox_tlwh = track->tlwh();
        cv::Scalar color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);

        if (track_colors.find(track->trackId()) == track_colors.end()) {
            track_colors[track->trackId()] = color;
        } else {
            color = track_colors[track->trackId()];
        }

        cv::rectangle(frame,
                      cv::Rect(static_cast<int>(bbox_tlwh[0]), static_cast<int>(bbox_tlwh[1]),
                               static_cast<int>(bbox_tlwh[2]), static_cast<int>(bbox_tlwh[3])),
                      color, 2);
        cv::putText(frame, std::to_string(track->trackId()),
                    cv::Point(static_cast<int>(bbox_tlwh[0]), static_cast<int>(bbox_tlwh[1])),
                    cv::FONT_HERSHEY_SIMPLEX, 0.75, color, 2);

        cv::rectangle(frame, cv::Rect(10, 10, 20, 20), detection_color, -1);
        cv::putText(frame, "Detection", cv::Point(40, 25), cv::FONT_HERSHEY_SIMPLEX, 0.75,
                    detection_color, 2);
        cv::putText(frame, "FPS: " + std::to_string(fps), cv::Point(20, 65),
                    cv::FONT_HERSHEY_SIMPLEX, 0.75, detection_color, 2);
    }
}

void checkVideo(cv::VideoCapture& cap) {
    if (!cap.isOpened()) {
        std::cout << "Can't open video " << std::endl;
        exit(-1);
    }
    std::cout << "视频中图像的宽度=" << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "视频中图像的高度=" << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "视频帧率=" << cap.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "视频的总帧数=" << cap.get(cv::CAP_PROP_FRAME_COUNT);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: ./track <engine_path> <config_path> <video_path>" << std::endl;
        return -1;
    }
    centernet::CenterEngine engine(argv[1]);
    std::unique_ptr<botsort::BoTSORT> botSort = std::make_unique<botsort::BoTSORT>(argv[2]);
    cv::VideoCapture cap(argv[3]);
    checkVideo(cap);
    cv::namedWindow("result", cv::WINDOW_NORMAL);
    cv::resizeWindow("result", 1920, 1080);
    std::unique_ptr<float[]> output_data(new float[engine.outputBufferSize()]);
    cv::Mat img;
    std::chrono::microseconds avg_det_time(0);
    std::chrono::microseconds avg_track_time(0);
    int frame_id = 0;
    while (true) {
        cap >> img;
        if (img.empty()) {
            break;
        }
        ++frame_id;
        auto t0 = std::chrono::steady_clock::now();
        auto net_input = prepareImage(img);
        engine.infer(net_input.data(), output_data.get());
        int num_det = static_cast<int>(output_data[0]);
        std::vector<common::Detection> results(num_det);
        memcpy(results.data(), &output_data[1], num_det * sizeof(common::Detection));
        centernet::util::correctBox(results, img.cols, img.rows);
        // std::cout << "Det result: " << results.size() << std::endl;
        auto t1 = std::chrono::steady_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
        std::cout << "Detection Cost: " << dur.count() << " microseconds" << std::endl;
        auto track_res = botSort->track(results, img);
        auto t2 = std::chrono::steady_clock::now();
        auto dur1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        plot_tracks(img, results, track_res, 1'000'000 / (dur + dur1).count());

        std::cout << "Tracking Cost: " << dur1.count() << " microseconds" << std::endl;
        std::cout << "Total: " << (dur + dur1).count() << " microseconds" << std::endl;
        avg_det_time = avg_det_time + std::chrono::duration_cast<std::chrono::microseconds>(
                                          1.0 / frame_id * (dur - avg_det_time));
        avg_track_time = avg_track_time + std::chrono::duration_cast<std::chrono::microseconds>(
                                              1.0 / frame_id * ((dur + dur1) - avg_track_time));
        cv::imshow("result", img);
        cv::waitKey(1);
    }
    std::cout << "Avg det time: " << avg_det_time.count() << " microseconds" << std::endl;
    std::cout << "Avg track time: " << avg_track_time.count() << " microseconds" << std::endl;
    std::cout << "Avg FPS: " << 1'000'000 / avg_track_time.count() << std::endl;

    return 0;
}