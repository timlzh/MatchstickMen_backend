#include "json_util.h"

std::vector<Area> readCoveringArea()
{
    Json::Reader reader;
    Json::Value root;
    std::vector<Area> result;

    //从文件中读取，保证当前文件有json文件
    std::ifstream in("/home/khadas/workspace/express-server/code/publicForkhadas/conf/coverArea.json", std::ios::binary);

    if (!in.is_open())
    {
        printf("Error opening file\n");
        return result;
    }

    if (reader.parse(in, root))
    {
        for (unsigned int i = 0; i < root.size(); i++)
        {
            Area drawArea;
            drawArea.bottom = root[i]["bottom"].asInt();
            drawArea.top = root[i]["top"].asInt();
            drawArea.left = root[i]["left"].asInt();
            drawArea.right = root[i]["right"].asInt();
            result.push_back(drawArea);
        }
    }
    in.close();
    return result;
}

std::vector<WarningArea> readWarningArea()
{
    Json::Reader reader;
    Json::Value root;
    std::vector<WarningArea> result;

    //从文件中读取，保证当前文件有json文件
    std::ifstream in("/home/khadas/workspace/express-server/code/publicForkhadas/conf/warnArea.json", std::ios::binary);

    if (!in.is_open())
    {
        printf("Error opening file\n");
        return result;
    }

    if (reader.parse(in, root))
    {
        for (unsigned int i = 0; i < root.size(); i++)
        {
            WarningArea warningArea;
            Json::Value points = root[i]["points"];
            for (int i = 0; i < points.size(); i++)
            {
                warningArea.points.push_back(cv::Point(points[i]["x"].asInt(), points[i]["y"].asInt()));
            }
            result.push_back(warningArea);
        }
    }
    in.close();
    printf("quit\n");
    return result;
}

void read_config(int *idc, std::vector<Area> &coveringArea)
{
    coveringArea = readCoveringArea();
    Json::Reader reader;
    Json::Value root;

    //从文件中读取，保证当前文件有json文件
    std::ifstream in("/home/khadas/workspace/express-server/code/publicForkhadas/conf/idc.json", std::ios::binary);

    if (!in.is_open())
    {
        printf("Error opening file\n");
    }

    if (reader.parse(in, root))
    {
        *idc = root["idc"].asInt();
    }
    in.close();
}