#include <iostream>
#include <csignal>
#include <opencv4/opencv2/opencv.hpp>

typedef struct VideoInfo
{
    int fps;
    int height;
    int width;
};

bool is_running = true;

void start_capture_from_rtsp(cv::VideoCapture capture, std::queue<cv::Mat> &buffer)
{

    cv::Mat frame;

    // 将 cv 读到的每一帧传入子进程
    while (is_running)
    {
        capture >> frame;
        if (frame.empty())
        {
            continue;
        }
        while (buffer.size() >= 30)
            ;
        buffer.push(frame.clone());
    }
}
