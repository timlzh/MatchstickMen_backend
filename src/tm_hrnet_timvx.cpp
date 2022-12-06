/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2021, OPEN AI LAB
 * Author: qtang@openailab.com
 * Author: stevenwudi@fiture.com
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <thread>
#include <mutex>
#include <jsoncpp/json/json.h>
#include "json_util.h"
#include "common.h"
#include "points_in_polygon.h"
#include "http.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"
#include "capture_from_rtsp.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/stat.h>
#include <fstream>
#include <fcntl.h>

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1
#define LETTERBOX_ROWS 256
#define LETTERBOX_COLS 256
#define MODEL_CHANNELS 3
#define HEATMAP_CHANNEL 16
#define OPENCV_TO_HRNET "/home/khadas/workspace/FIFO/OPENCV_TO_HRNET"
#define PORT 6005

std::queue<cv::Mat> framebuffer;

typedef struct
{
    float x;
    float y;
    float score;
} ai_point_t;

struct skeleton
{
    int connection[2];
    int left_right_neutral;
};

typedef struct
{
    std::vector<ai_point_t> keypoints;
    std::vector<ai_point_t> prepoints;
    int32_t img_width = 0;
    int32_t img_heigh = 0;
    uint64_t timestamp = 0;
} ai_body_parts_s;

typedef struct FrameInfo
{
    int people_exsist;
    int frame_count;
    cv::Mat frame;
    ai_body_parts_s pose;
};

class FrameQueue
{

private:
    std::queue<FrameInfo> queue;
    std::mutex mu;
    int current_size;
    int max_size;

public:
    FrameQueue(int queue_size)
    {
        current_size = 0;
        max_size = queue_size;
    }
    void Push(FrameInfo info)
    {
        while (true)
        {
            if (current_size < max_size)
            {
                //only one writer, so lock after reading current_size if ok
                mu.lock();
                //std::queue is not thread safe
                queue.push(info);
                current_size++;
                mu.unlock();
                break;
            }
            printf("[info] queue is full.\n");
        }
    }
    FrameInfo Pop()
    {
        mu.lock();
        FrameInfo info = queue.front();
        queue.pop();
        current_size--;
        mu.unlock();
        return info;
    }
    bool Empty()
    {
        return current_size == 0;
    }
};

std::vector<skeleton> pairs = {{0, 1, 0},
                               {1, 2, 0},
                               {3, 4, 1},
                               {4, 5, 1},
                               {2, 6, 0},
                               {3, 6, 1},
                               {6, 7, 2},
                               {7, 8, 2},
                               {8, 9, 2},
                               {13, 7, 1},
                               {10, 11, 0},
                               {7, 12, 0},
                               {12, 11, 0},
                               {13, 14, 1},
                               {14, 15, 1}};

std::vector<Area> coveringArea;
std::vector<WarningArea> warningArea;
int stay_count = 0;
int is_counting = 0;
int idc = 0;
/*for socket*/
int sock;
#define MAX_STAY_COUNT 10
#define MAX_STAY_R 10
FrameQueue queue(300);
// cv::VideoWriter output;

void update_config_clockly()
{
    while (true)
    {
        sleep(1);
        coveringArea = readCoveringArea();
        warningArea = readWarningArea();
    }
}

void draw_result(cv::Mat img, ai_body_parts_s &pose);
void picture_out(cv::Mat &out)
{
    std::vector<unsigned char> buffer;
    cv::imencode(".jpg", out, buffer);
    int img_size = buffer.size();
    send(sock, &img_size, sizeof(img_size), 0);
    send(sock, buffer.data(), img_size, 0);
}

void draw_frame_and_send()
{

    while (true)
    {
        while (!queue.Empty())
        {

            FrameInfo frameinfo = queue.Pop();
            char filename[100];
            sprintf(filename, "/home/khadas/MatchstickMen/drawed/out-%d.jpg", frameinfo.frame_count);
            printf("picture-%d is drawed\n", frameinfo.frame_count);
            if (frameinfo.people_exsist == 1)
            {
                draw_result(frameinfo.frame, frameinfo.pose);
            }
            picture_out(frameinfo.frame);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        printf("[info] queue is empty.\n");
    }
}

void FindMax2D(float *buf, int width, int height, int *max_idx_width, int *max_idx_height, float *max_value, int c)
{
    float *ptr = buf;
    *max_value = -10.f;
    *max_idx_width = 0;
    *max_idx_height = 0;
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            float score = ptr[c * height * width + h * height + w];
            if (score > *max_value)
            {
                *max_value = score;
                *max_idx_height = h;
                *max_idx_width = w;
            }
        }
    }
}

bool isInRange(float x1, float y1, float x2, float y2)
{
    if ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) > MAX_STAY_R * MAX_STAY_R)
    {
        return false;
    }
    return true;
}

bool isMoving(const std::vector<ai_point_t> &pose1, const std::vector<ai_point_t> &pose2)
{
    for (int c = 0; c < HEATMAP_CHANNEL; ++c)
    {
        if (!isInRange(pose1[c].x, pose1[c].y, pose2[c].x, pose2[c].y))
        {
            return false;
        }
    }
    return true;
}
void PostProcess(float *data, ai_body_parts_s &pose, int img_h, int img_w, int *people_exsist)
{
    int heatmap_width = img_w / 4;
    int heatmap_height = img_h / 4;
    int max_idx_width, max_idx_height;
    float max_score;
    float avg_socre = 0;

    ai_point_t kp;
    pose.keypoints.clear();
    for (int c = 0; c < HEATMAP_CHANNEL; ++c)
    {
        FindMax2D(data, heatmap_width, heatmap_height, &max_idx_width, &max_idx_height, &max_score, c);
        kp.x = (float)max_idx_width / (float)heatmap_width;
        kp.y = (float)max_idx_height / (float)heatmap_height;
        kp.score = max_score;
        pose.keypoints.push_back(kp);
        avg_socre += pose.keypoints[c].score;

        //        std::cout << "x: " << pose.keypoints[c].x * 64 << ", y: " << pose.keypoints[c].y * 64 << ", score: "
        //                  << pose.keypoints[c].score << std::endl;
    }
    avg_socre /= HEATMAP_CHANNEL;
    if (avg_socre > 0.6)
    {
        *people_exsist = 1;
    }
    if (*people_exsist == 1)
    {
        std::vector<cv::Point> cv_pose;
        for (auto &&item : pose.keypoints)
        {
            cv_pose.push_back(cv::Point(item.x, item.y));
        }

        // if (AllPointsInPolygon(cv_pose, warningArea[0].points))
        // {
        //     if (is_counting)
        //     {
        //         if (!isMoving(pose.prepoints, pose.keypoints))
        //         {
        //             if (++stay_count > MAX_STAY_COUNT)
        //             {
        //                 //
        //                 HttpRequest *Http;
        //                 char http_return[4096] = {0};
        //                 std::string http_msg = "http://47.108.213.171:8003/warnFromCamera?idc=";
        //                 http_msg += std::to_string(idc);
        //                 if (Http->HttpGet(http_msg.data(), http_return))
        //                 {
        //                     std::cout << http_return << std::endl;
        //                 }
        //             }
        //         }
        //         else
        //         {
        //             is_counting = 0;
        //             stay_count = 0;
        //         }
        //     }
        //     else
        //     {
        //         is_counting = 1;
        //         stay_count = 1;
        //         pose.prepoints = pose.keypoints;
        //     }
        // }
    }
    else
    {
        stay_count = 0;
        is_counting = 0;
    }
}

void draw_result(cv::Mat img, ai_body_parts_s &pose)
{
    /* recover process to draw */
    float scale_letterbox;
    int resize_rows;
    int resize_cols;

    if ((LETTERBOX_ROWS * 1.0 / img.rows) < (LETTERBOX_COLS * 1.0 / img.cols))
        scale_letterbox = LETTERBOX_ROWS * 1.0 / img.rows;
    else
        scale_letterbox = LETTERBOX_COLS * 1.0 / img.cols;

    resize_cols = int(scale_letterbox * img.cols);
    resize_rows = int(scale_letterbox * img.rows);

    int tmp_h = (LETTERBOX_ROWS - resize_rows) / 2;
    int tmp_w = (LETTERBOX_COLS - resize_cols) / 2;

    float ratio_x = (float)img.rows / resize_rows;
    float ratio_y = (float)img.cols / resize_cols;

    for (int i = 0; i < HEATMAP_CHANNEL; i++)
    {
        int x = (int)((pose.keypoints[i].x * LETTERBOX_COLS - tmp_w) * ratio_x);
        int y = (int)((pose.keypoints[i].y * LETTERBOX_ROWS - tmp_h) * ratio_y);

        x = std::max(std::min(x, (img.cols - 1)), 0);
        y = std::max(std::min(y, (img.rows - 1)), 0);

        cv::circle(img, cv::Point(x, y), 4, cv::Scalar(0, 0, 0), cv::FILLED);
    }

    cv::Scalar color;
    cv::Point pt1;
    cv::Point pt2;
    cv::Point body1;
    cv::Point body2;
    for (auto &element : pairs)
    {
        switch (element.left_right_neutral)
        {
        case 0:
            color = cv::Scalar(0, 0, 0);
            break;
        case 1:
            color = cv::Scalar(0, 0, 0);
            break;
        default:
            color = cv::Scalar(0, 0, 0);
        }

        int x1 = (int)((pose.keypoints[element.connection[0]].x * LETTERBOX_COLS - tmp_w) * ratio_x);
        int y1 = (int)((pose.keypoints[element.connection[0]].y * LETTERBOX_ROWS - tmp_h) * ratio_y);
        int x2 = (int)((pose.keypoints[element.connection[1]].x * LETTERBOX_COLS - tmp_w) * ratio_x);
        int y2 = (int)((pose.keypoints[element.connection[1]].y * LETTERBOX_ROWS - tmp_h) * ratio_y);

        x1 = std::max(std::min(x1, (img.cols - 1)), 0);
        y1 = std::max(std::min(y1, (img.rows - 1)), 0);
        x2 = std::max(std::min(x2, (img.cols - 1)), 0);
        y2 = std::max(std::min(y2, (img.rows - 1)), 0);

        pt1 = cv::Point(x1, y1);
        pt2 = cv::Point(x2, y2);
        if (pose.keypoints[element.connection[0]].x == pose.keypoints[6].x &&
            pose.keypoints[element.connection[0]].y == pose.keypoints[6].y)
        {
            body1 = pt1;
            body2 = pt2;
            continue;
        }
        cv::line(img, pt1, pt2, color, 80);
    }

    cv::line(img, body1, body2, color, 200);

    //point 0 is left foot
    //point 15 is right hand
    //point 13,12,11 are shoulders and right arm
    //point 6 is body
    int x = (int)((pose.keypoints[9].x * LETTERBOX_COLS - tmp_w) * ratio_x);
    int y = (int)((pose.keypoints[9].y * LETTERBOX_ROWS - tmp_h) * ratio_y);
    x = std::max(std::min(x, (img.cols - 1)), 0);
    y = std::max(std::min(y, (img.rows - 1)), 0);
    cv::ellipse(img, cv::Point(x, y + 60), cv::Size(90, 110), 0, 0, 360, cv::Scalar(0, 0, 0), cv::FILLED);
    for (auto &&i : coveringArea)
    {
        cv::rectangle(img, cv::Point(i.left, i.top), cv::Point(i.right, i.bottom), cv::Scalar(0, 0, 0), cv::FILLED);
    }
}

void get_input_uint8_data_square(cv::Mat &img, uint8_t *input_data, float *mean, float *scale,
                                 float input_scale, int zero_point)
{

    /* letterbox process to support different letterbox size */
    float scale_letterbox;
    // Currenty we only support square input.
    int resize_rows;
    int resize_cols;
    if ((LETTERBOX_ROWS * 1.0 / img.rows) < (LETTERBOX_COLS * 1.0 / img.cols * 1.0))
        scale_letterbox = 1.0 * LETTERBOX_ROWS / img.rows;
    else
        scale_letterbox = 1.0 * LETTERBOX_COLS / img.cols;

    resize_cols = int(scale_letterbox * img.cols);
    resize_rows = int(scale_letterbox * img.rows);
    cv::resize(img, img, cv::Size(resize_cols, resize_rows));
    img.convertTo(img, CV_32FC3);
    // Generate a gray image for letterbox
    cv::Mat img_new(LETTERBOX_COLS, LETTERBOX_ROWS, CV_32FC3,
                    cv::Scalar(0.5 / scale[0] + mean[0], 0.5 / scale[1] + mean[1], 0.5 / scale[2] + mean[2]));

    int top = (LETTERBOX_ROWS - resize_rows) / 2;
    int bot = (LETTERBOX_ROWS - resize_rows + 1) / 2;
    int left = (LETTERBOX_COLS - resize_cols) / 2;
    int right = (LETTERBOX_COLS - resize_cols + 1) / 2;
    // Letterbox filling
    cv::copyMakeBorder(img, img_new, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    //    cv::imwrite("hrnet_lb_image.jpg", img_new); // for letterbox test
    float *img_data = (float *)img_new.data;

    /* nhwc to nchw */
    for (int h = 0; h < LETTERBOX_ROWS; h++)
    {
        for (int w = 0; w < LETTERBOX_COLS; w++)
        {
            for (int c = 0; c < MODEL_CHANNELS; c++)
            {
                int in_index = h * LETTERBOX_COLS * MODEL_CHANNELS + w * MODEL_CHANNELS + c;
                int out_index = c * LETTERBOX_ROWS * LETTERBOX_COLS + h * LETTERBOX_COLS + w;
                float input_temp = (img_data[in_index] - mean[c]) * scale[c];
                /* quant to uint8 */
                int udata = (round)(input_temp / input_scale + (float)zero_point);
                if (udata > 255)
                    udata = 255;
                else if (udata < 0)
                    udata = 0;
                input_data[out_index] = udata;
            }
        }
    }
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-o output_file] [-r repeat_count] [-t thread_count]\n");
}

void make_fifos(const char *fifo_dir, __mode_t mode)
{
    if (access(fifo_dir, F_OK) == -1)
    {
        if (mkfifo(fifo_dir, O_RDONLY | O_CREAT) != 0 && (errno != EEXIST))
        {
            fprintf(stderr, "Open fifo failed.\n");
            exit(-1);
        }
    }
}

std::string int_to_string(int x)
{
    std::string ans("");
    while (x)
    {
        char now = x % 10 + '0';
        x /= 10;
        ans += now;
    }
    std::reverse(ans.begin(), ans.end());
    return ans;
}
/*
 * 图像转成字符流输出
 */

int main(int argc, char *argv[])
{
    read_config(&idc, coveringArea);
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    char *model_file = nullptr;
    char *output_file = nullptr;
    char *hostname;
    int img_h = LETTERBOX_COLS;
    int img_w = LETTERBOX_ROWS;
    /*for socket*/
    int valread;
    struct sockaddr_in serv_addr;
    // char hostname[100] = "uestc-nu.nodes.naiv.fun";
    // char hostname[100] = "localhost";
    char ip[100];

    ai_body_parts_s pose;

    float mean[3] = {123.67f, 116.28f, 103.53f};
    float scale[3] = {0.017125f, 0.017507f, 0.017429f};

    int res;
    while ((res = getopt(argc, argv, "m:r:t:d:")) != -1)
    {
        switch (res)
        {
        case 'm':
            model_file = optarg;
            break;
        case 'd':
            hostname = optarg;
            break;
        case 'r':
            repeat_count = atoi(optarg);
            break;
        case 't':
            num_thread = atoi(optarg);
            break;
        case 'h':
            show_usage();
            return 0;
        default:
            break;
        }
    }

    /* check files */
    if (model_file == nullptr)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (hostname == nullptr)
    {
        fprintf(stderr, "Error:hostname not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file))
        return -1;

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_UINT8;
    opt.affinity = 0;

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create VeriSilicon TIM-VX backend */
    context_t timvx_context = create_context("timvx", 1);
    int rtt = set_context_device(timvx_context, "TIMVX", nullptr, 0);
    if (0 > rtt)
    {
        fprintf(stderr, " add_context_device VSI DEVICE failed.\n");
        return -1;
    }
    /* create graph, load tengine model xxx.tmfile */

    graph_t graph = create_graph(timvx_context, "tengine", model_file);
    if (graph == nullptr)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w}; // nchw

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    std::vector<uint8_t> input_data(img_size);
    if (input_tensor == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_data.data(), img_size) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    //获取域名对应的ip
    struct hostent *he;
    struct in_addr **addr_list;

    if ((he = gethostbyname(hostname)) == NULL)
    {
        // get the host info
        herror("gethostbyname");
        return 1;
    }

    addr_list = (struct in_addr **)he->h_addr_list;

    for (int i = 0; addr_list[i] != NULL; i++)
    {
        //Return the first one;
        strcpy(ip, inet_ntoa(*addr_list[i]));
    }

    printf("ip adress: %s\n", ip);
    char buffer[1024] = {0};
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("\n Socket creation error \n");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    // Convert IPv4 and IPv6 addresses from text to binary form
    if (inet_pton(AF_INET, ip, &serv_addr.sin_addr) <= 0)
    {
        printf(
            "\nInvalid address/ Address not supported \n");
        return -1;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr,
                sizeof(serv_addr)) < 0)
    {
        printf("\nConnection Failed \n");
        return -1;
    }
    //send(sock, hello, strlen(hello), 0);

    std::thread updater(update_config_clockly);
    std::thread sender(draw_frame_and_send);
    cv::VideoCapture cap("rtmp://localhost/live/livestream");
    // cv::VideoCapture cap(1);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera." << std::endl;
        exit(-1);
    }
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS);
    // int width = 1920,height=1080,fps=30;

    std::thread capture(start_capture_from_rtsp, cap, std::ref(framebuffer));

    VideoInfo videoinfo;
    videoinfo.width = width;
    videoinfo.height = height;
    videoinfo.fps = fps;
    send(sock, &videoinfo, sizeof(VideoInfo), 0);

    for (int frame_count = 0;; frame_count++)
    {
        //read a image
        while (framebuffer.empty())
            ;
        cv::Mat src = framebuffer.front();
        framebuffer.pop();

        cv::Mat bak(src);

        /* prepare process input data, set the data mem to input tensor */
        float input_scale = 0.f;
        int input_zero_point = 0;
        get_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1);
        get_input_uint8_data_square(src, input_data.data(), mean, scale, input_scale, input_zero_point);

        /* run graph */
        double min_time = DBL_MAX;
        double max_time = DBL_MIN;
        double total_time = 0.;
        for (int i = 0; i < repeat_count; i++)
        {
            double start = get_current_time();
            if (run_graph(graph, 1) < 0)
            {
                fprintf(stderr, "Run graph failed\n");
                return -1;
            }
            double end = get_current_time();
            double cur = end - start;
            total_time += cur;
            min_time = std::min(min_time, cur);
            max_time = std::max(max_time, cur);
        }
        //fprintf(stderr, "Repeat [%d] min %.3f ms, max %.3f ms, avg %.3f ms\n", repeat_count, min_time, max_time,
        //        total_time / repeat_count);

        /* get output tensor */
        tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
        uint8_t *output_u8 = (uint8_t *)get_tensor_buffer(output_tensor);
        int output_size = get_tensor_buffer_size(output_tensor) / sizeof(uint8_t);
        /* dequant */
        float output_scale = 0.f;
        int output_zero_point = 0;
        get_tensor_quant_param(output_tensor, &output_scale, &output_zero_point, 1);
        // float* output_data = ( float* )malloc(output_size * sizeof(float));
        std::vector<float> output_data(output_size);
        for (int i = 0; i < output_size; i++)
        {
            output_data[i] = ((float)output_u8[i] - (float)output_zero_point) * output_scale;
        }
        int people_exsist = 0;
        PostProcess(output_data.data(), pose, img_h, img_w, &people_exsist);
        /* write some visualisation  */
        FrameInfo frameinfo;
        frameinfo.people_exsist = people_exsist;
        frameinfo.frame_count = frame_count;
        frameinfo.pose = pose;
        frameinfo.frame = bak.clone();
        queue.Push(frameinfo);
    }
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();
    return 0;
}
