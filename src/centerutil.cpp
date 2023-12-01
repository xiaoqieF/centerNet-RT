#include <sstream>

#include "centerutils.h"
#include "config.h"

namespace util {

void correctBox(std::vector<Detection>& results, const int img_w, const int img_h) {
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

void drawImg(const std::vector<Detection>& results,
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