#include <bits/stdc++.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//从目标点出发引一条射线，看这条射线和多边形所有边的交点数目：
//（1）如果有奇数个交点，则说明在内部；
//（2）如果有偶数个交点，则说明在外部.//

//points是要判断的点，polygon里存放的是多边形顶点顺时针的坐标
int PointInPolygon(cv::Point point, const std::vector<cv::Point> &polygon);
int AllPointsInPolygon(std::vector<cv::Point> points, const std::vector<cv::Point> &polygon);
