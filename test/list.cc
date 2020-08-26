#include<iostream>
#include<list>

using namespace std;

int main()
{
    list<int> lt;
    for(int i=0; i<4;i++)
        lt.push_back(i);

    list<int>::iterator iter;

    for(iter=lt.begin(); iter != lt.end(); ++iter)
    {
        std::cout<<*iter<<std::endl;
    }

    return 0;
    
}
