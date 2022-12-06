#include "points_in_polygon.h"

//points是要判断的点，polygon里存放的是多边形顶点顺时针的坐标
//return true is point in polygon
int PointInPolygon(cv::Point point, const std::vector<cv::Point> &polygon)
{
    bool check = false;
    int len = polygon.size();
    int lastVertex = len - 1;
    for (int i = 0; i < len; i++)
    {
        //判断点是否在直线上
        if ((polygon[i].y < point.y && polygon[lastVertex].y >= point.y) || (polygon[lastVertex].y < point.y && polygon[i].y >= point.y))
        {
            //x = x1 +  (y - y1) * (x2 - x1) / (y2 - y1);
            if (polygon[i].x + (point.y - polygon[i].y) * (polygon[lastVertex].x - polygon[i].x) / (polygon[lastVertex].y - polygon[i].y) < point.x)
                check = !check;
        }
        lastVertex = i;
    }

    return check;
}

//return true if all points in polygon
int AllPointsInPolygon(std::vector<cv::Point> points, const std::vector<cv::Point> &polygon)
{
    for (auto &&item : polygon)
    {
        if (!PointInPolygon(item, polygon))
        {
            return false;
        }
    }
    return true;
}
