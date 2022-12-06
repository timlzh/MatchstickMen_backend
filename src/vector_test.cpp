#include <bits/stdc++.h>
using namespace std;
int main()
{
    std::vector<int> vec1, vec2;
    vec1.push_back(1);
    vec1.push_back(2);
    vec2 = vec1;
    vec2[0] = 2;
    std::cout << vec1[0] << " " << vec1[1] << std::endl;
    std::cout << vec2[0] << " " << vec2[1] << std::endl;
    std::cout << &vec1[0] << " " << &vec1[1] << std::endl;
    std::cout << &vec2[0] << " " << &vec2[1] << std::endl;
}