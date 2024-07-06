#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <experimental/filesystem>

#define TEST_PATHMANAGER 0

using namespace std;
namespace fs = std::experimental::filesystem;

class PathManager
{
public:
    PathManager(const string &s);

    bool beingCurFile;
    void nextFile();
    int getImagesNumber();
    // getter
    string getFilename();
    string getCurImgPath();

private:
    fs::path imgDir; // the directory of images
    vector<string> filenamesWFormat; // the filenames of images, with format
    int curIdx; // the index for current file
    string curFilenameWoFormat; // the current filename without suffix
    string curFileSuffix; // the suffix of the current file
    string _getFileSuffix();
};
