#include "dataset.h"

namespace DATASET 
{
// Constructor
Dataset::Dataset(char* _base, char* _scene):base(_base), scene(_scene)
{
    //Validation test
    path = base+scene;
    cout<<path<<endl;
    iter = fs::directory_iterator(path);
    bool flag_valid = false;
    unsigned int _n_images=0;

    if(fs::is_directory(path))
    {
        cout<<"Dataset Validation Test Complete"<<endl;
        cout<<"Dataset Path:\n"<<base<<endl;
        cout<<"Scene name:\n"<<scene<<endl;
    }
    else
    {
        cout<<"Dataset Validation Fail"<<endl;
        cout<<"Dataset Path:\n"<<base<<endl;
        cout<<"Scene name:\n"<<scene<<endl;
    }
    
    count = 0;
    n_images=0;
    fs::directory_iterator end_iter;
    for(auto& p:fs::directory_iterator(path))
    {
        if (iter->path().extension() == ".jpg")
        {
            ++_n_images;
        }
    }
    while (iter != fs::end(iter))
    {
    const fs::directory_entry& entry = *iter; filenames.push_back(entry.path()); ++iter;
    }
    std::sort(filenames.begin(), filenames.end());
    bValid = &flag_valid;
    n_images = &_n_images;
    cout<<"Total #image= "<<*n_images<<endl;
    
}

// Methods
std::string Dataset::next()
{
    std::string res = filenames[count];
    count +=1;
    return res;

}
}

