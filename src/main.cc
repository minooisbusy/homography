#include<iostream>
#include"dataset.h"
#include"frame.h"
#include<tuple>
#include<iostream>

int main(int argc, char* argv[])
{
    std::srand(time(0));
    DATASET::Dataset ds(argv[1],argv[2]); // load data
    FRAME::FramePair fp(ds.next(),ds.next());

    //fp.showPairs();
    fp.compute();
    return 0;
}
