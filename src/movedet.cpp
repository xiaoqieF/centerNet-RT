#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace std;

// 稀疏光流
void sparseOptFlow(const std::string& video_path) {
    VideoCapture capture(video_path);
    if (!capture.isOpened()) {
        // error in opening the video input
        cerr << "Unable to open file!" << endl;
    }
    // Create some random colors
    vector<Scalar> colors;
    RNG rng;
    for (int i = 0; i < 100; i++) {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r, g, b));
    }
    Mat old_frame, old_gray;
    vector<Point2f> p0, p1;
    // Take first frame and find corners in it
    capture >> old_frame;
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
    // Create a mask image for drawing purposes
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());
    while (true) {
        Mat frame, frame_gray;
        capture >> frame;
        if (frame.empty()) break;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        // 计算光流点
        vector<uchar> status;
        vector<float> err;
        // 设置迭代终止条件
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15, 15), 2, criteria);
        vector<Point2f> good_new;
        for (uint i = 0; i < p0.size(); i++) {
            // Select good points
            if (status[i] == 1) {
                good_new.push_back(p1[i]);
                // draw the tracks
                //                line(mask,p1[i], p0[i], Scalar(0,0,255), 2);
                //                circle(frame, p1[i], 5, Scalar(0,0,255), -1);
                circle(mask, p1[i], 10, Scalar(0, 0, 255), -1);
            }
        }
        Mat img;
        add(frame, mask, img);
        imshow("Frame", img);
        int keyboard = waitKey(1);
        if (keyboard == 'q' || keyboard == 27) break;
        // Now update the previous frame and previous points
        old_gray = frame_gray.clone();
        p0 = good_new;
    }
}

// 背景消减
void backDiff(const std::string& video_path) {
    Mat frame;      // 当前帧
    Mat fgMaskMOG2; // 通过MOG2方法得到的掩码图像fgmask
    Mat segm;       // frame的副本
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();
    VideoCapture capture(video_path); // 参数为0，默认从摄像头读取视频
    if (!capture.isOpened()) {
        cout << "Unable to open the camera! " << endl;
        // EXIT_FAILURE 可以作为exit()的参数来使用，表示没有成功地执行一个程序,其值为1
        exit(EXIT_FAILURE);
    }
    while (true) {
        if (!capture.read(frame)) {
            cout << "Unable to read next frame." << endl;
            exit(0);
        }
        pMOG2->apply(frame, fgMaskMOG2); // 更新背景模型
                                         // getStructuringElement构造形态学使用的kernel
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        // 使用形态学的开运算做背景的去除
        morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_OPEN, kernel);

        frame.copyTo(segm); // 建立一个当前frame的副本
        findContours(fgMaskMOG2, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE,
                     Point(0, 0)); // 检测轮廓
        std::cout << contours.size() << std::endl;
        vector<vector<Point>> contours_poly(contours.size());
        vector<Point2f> center(contours.size());
        vector<float> radius(contours.size());
        for (int i = 0; i < contours.size(); i++) {
            // findContours后的轮廓信息contours可能过于复杂不平滑，
            // 可以用approxPolyDP函数对该多边形曲线做适当近似
            approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
            // 得到轮廓的外包络圆
            minEnclosingCircle(contours_poly[i], center[i], radius[i]);
        }
        // 对所得到的轮廓进行一定的筛选
        for (int i = 0; i < contours.size(); i++) {
            if (contourArea(contours[i]) > 10) {
                circle(segm, center[i], (int)radius[i], Scalar(100, 100, 0), 2, 8, 0);
                break;
            }
        }
        // 得到当前是第几帧
        stringstream ss;
        //        rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
        //                  cv::Scalar(255,255,255), -1);
        ss << capture.get(CAP_PROP_POS_FRAMES);
        string frameNumberString = ss.str();
        putText(frame, frameNumberString.c_str(), cv::Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0));
        // 显示
        imshow("frame", frame);
        imshow("Segm", segm);
        imshow("FG Mask MOG 2", fgMaskMOG2);
        int key;
        key = waitKey(5);
        if (key == 'q' || key == 'Q' || key == 27) break;
    }
    capture.release();
    destroyAllWindows();
}

// 逐帧差分
Mat frameDiff(Mat frame1, Mat frame2) {
    Mat result = frame2.clone();
    Mat gray1, gray2;
    cvtColor(frame1, gray1, COLOR_BGR2GRAY);
    cvtColor(frame2, gray2, COLOR_BGR2GRAY);
    Mat diff;
    absdiff(gray1, gray2, diff);
    // imshow("absdiss", diff);
    threshold(diff, diff, 40, 255, THRESH_BINARY);
    imshow("threshold", diff);
    // medianBlur(diff, diff, 5);
    // imshow("medianBlur", diff);
    // Mat element = getStructuringElement(MORPH_RECT, Size(2, 2));
    Mat element2 = getStructuringElement(MORPH_RECT, Size(5, 5));
    // erode(diff, diff, element);
    dilate(diff, diff, element2);
    imshow("dilate", diff);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarcy;
    findContours(diff, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE,
                 Point(0, 0)); // 查找轮廓
    vector<vector<Point>> contours_poly(contours.size());
    vector<Rect> boundRect(contours.size()); // 定义外接矩形集合
    // drawContours(img2, contours, -1, Scalar(0, 0, 255), 1, 8);  //绘制轮廓
    int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        // 对图像轮廓点进行多边形拟合：轮廓点组成的点集，输出的多边形点集，精度（即两个轮廓点之间的距离），输出多边形是否封闭
        approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
        boundRect[i] = boundingRect(Mat(contours_poly[i]));
        if (boundRect[i].width > 5 && boundRect[i].width < 40 && boundRect[i].height > 5 &&
            boundRect[i].height < 40) { // 轮廓筛选
            x0 = boundRect[i].x;
            y0 = boundRect[i].y;
            w0 = boundRect[i].width;
            h0 = boundRect[i].height;
            rectangle(result, Point(x0, y0), Point(x0 + w0, y0 + h0), Scalar(0, 255, 0), 2, 8, 0);
        }
    }
    return result;
}

void doFrameDiff(const std::string& video_path) {
    VideoCapture cap(video_path);
    if (!cap.isOpened()) // 检查打开是否成功
        return;
    Mat frame;
    Mat tmp;
    Mat result;
    int count = 0;
    while (1) {
        cap >> frame;
        if (frame.empty()) // 检查视频是否结束
            break;
        else {
            count++;
            if (count == 1)
                result = frameDiff(frame, frame);
            else
                result = frameDiff(tmp, frame);
            imshow("frameDiff", result);
            tmp = frame.clone();
            if (waitKey(20) == 27) break;
        }
    }
    cap.release();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./move_det <video_path>" << std::endl;
        return -1;
    }
    doFrameDiff(argv[1]);

    // backDiff(argv[1]);

    // sparseOptFlow(argv[1]);
}