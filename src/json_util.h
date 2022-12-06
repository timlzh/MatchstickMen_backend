#include <jsoncpp/json/json.h>
#include <fstream>
#include <iostream>
#include <vector>
#include "points_in_polygon.h"

typedef struct Area
{
    int bottom;
    int top;
    int left;
    int right;
};

typedef struct WarningArea
{
    std::vector<cv::Point> points;
};

std::vector<Area> readCoveringArea();
std::vector<WarningArea> readWarningArea();
void read_config(int *idc, std::vector<Area> &coveringArea);