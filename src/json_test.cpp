#include <iostream>
#include "json_util.h"

int main()
{
    auto result = readWarningArea();
    for (auto &&i : result)
    {
        printf("points:\n");

        for (auto &&j : i.points)
        {
            printf("x=%d,y=%d", j.x, j.y);
        }
        printf("\n");
    }
}