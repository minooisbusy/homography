#include<iostream>
#include"dataset.h"
#include"frame.h"

int main(int argc, char* argv[])
{
    DATASET::Dataset ds(argv[1],argv[2]);
    FRAME::FramePair fp(ds.next(),ds.next());

    fp.showPairs();
    fp.compute();
    std::cout<<"Hello world"<<std::endl;

    return 0;
}
