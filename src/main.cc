#include<iostream>
#include"dataset.h"
#include"frame.h"
#include<tuple>
#include<iostream>

int main(int argc, char* argv[])
{
    std::srand(time(0));
    DATASET::Dataset ds(argv[1],argv[2]); // load data
    std::string arg1 = ds.next();
    std::string arg2 = ds.next();

    FRAME::FramePair fp(arg2,arg1);

    //fp.showPairs();
    fp.compute();
    return 0;
}
