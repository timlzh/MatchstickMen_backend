#include "http.h"
#include <string.h>
#include <iostream>
int main(void)
{
    int idc = 1;
    HttpRequest *Http;
    char http_return[4096] = {0};
    std::string http_msg = "http://47.108.213.171:8003/warnFromCamera?idc=";
    http_msg += std::to_string(idc);
    if (Http->HttpGet(http_msg.data(), http_return))
    {
        std::cout << http_return << std::endl;
    }
    return 0;
}