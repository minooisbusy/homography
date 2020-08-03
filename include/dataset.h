#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;
namespace fs = std::filesystem;
namespace DATASET
{
class Dataset
{
private:
    std::vector<std::string> filenames;
    fs::path path;
    std::string base = "";
    std::string scene = "";
    fs::directory_iterator iter; // path 저장.
public:
    const bool *bValid;
    unsigned int count;
    const unsigned int *n_images;
public:
    explicit Dataset();
    explicit Dataset(char* _base, char* _scene);

    std::string next();
};
}
