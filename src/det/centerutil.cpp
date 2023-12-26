#include <sstream>

#include "centerutils.h"
#include "config.h"

namespace centernet {
namespace util {
std::vector<float> prepareImage(const cv::Mat& img) {
    int channel = config::channel;
    int input_w = config::input_w;
    int input_h = config::input_h;
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

void correctBox(std::vector<common::Detection>& results, const int img_w, const int img_h) {
    int input_w = centernet::config::input_w;
    int input_h = centernet::config::input_h;
    float scale =
        std::min(static_cast<float>(input_w) / img_w, static_cast<float>(input_h) / img_h);
    float dx = (input_w - scale * img_w) / 2;
    float dy = (input_h - scale * img_h) / 2;

    for (auto& item : results) {
        float x1 = (item.box.x1 - dx) / scale;
        float y1 = (item.box.y1 - dy) / scale;
        float x2 = (item.box.x2 - dx) / scale;
        float y2 = (item.box.y2 - dy) / scale;
        item.box.x1 = std::max(0.0f, x1);
        item.box.y1 = std::max(0.0f, y1);
        item.box.x2 = std::min(x2, static_cast<float>(img_w - 1));
        item.box.y2 = std::min(y2, static_cast<float>(img_h - 1));
    }
}

void drawImg(const std::vector<common::Detection>& results,
             cv::Mat& img,
             const std::vector<cv::Scalar>& color) {
    int box_think = (img.rows + img.cols) * 0.001;
    float label_scale = img.rows * 0.0009;
    int base_line;
    for (const auto& det : results) {
        std::string label;
        std::stringstream ss;
        ss << centernet::config::class_name[det.class_id] << " " << det.prob << std::endl;
        std::getline(ss, label);
        auto size = cv::getTextSize(label, cv::FONT_HERSHEY_COMPLEX, label_scale, 1, &base_line);
        cv::rectangle(img, cv::Point(det.box.x1, det.box.y1), cv::Point(det.box.x2, det.box.y2),
                      color[det.class_id], box_think * 2, 8, 0);
        cv::putText(img, label, cv::Point(det.box.x2, det.box.y2 - size.height),
                    cv::FONT_HERSHEY_COMPLEX, label_scale, color[det.class_id], box_think / 2, 8,
                    0);
    }
}
} // namespace util
} // namespace centernet