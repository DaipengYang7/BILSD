#include "PathManager.h"

PathManager::PathManager(const string &s)
{
    this->imgDir = fs::path(s);
    fs::path parentDir = this->imgDir.parent_path();
    fs::directory_iterator files(this->imgDir);
    for(auto& it: files)
        this->filenamesWFormat.push_back(it.path().filename().string());
    this->curIdx = -1;
    this->nextFile();
#if TEST_PATHMANAGER
    cout << this->imgDir << endl;
#endif
}

void PathManager::nextFile()
{
    this->curIdx++;
    if(this->curIdx < this->filenamesWFormat.size())
    {
        this->beingCurFile = true;
        this->curFilenameWoFormat = this->getFilename();
        this->curFileSuffix = this->_getFileSuffix();
    }
    else
    {
        this->beingCurFile = false;
        this->curFilenameWoFormat = "";
        this->curFileSuffix = "";
    }
}

int PathManager::getImagesNumber()
{
    return this->filenamesWFormat.size();
}

string PathManager::getFilename()
{
    // 这里假设文件名存在且格式正确，没有考虑意外情况
    string t = this->filenamesWFormat[curIdx];
    return t.substr(0, t.find("."));
}

string PathManager::getCurImgPath()
{
    return (this->imgDir / (this->curFilenameWoFormat + "." + this->curFileSuffix)).string();
}

string PathManager::_getFileSuffix()
{
    // 这里假设文件名存在且格式正确，没有考虑意外情况
    string t = filenamesWFormat[curIdx];
    int npos = t.find(".");
    return t.substr(npos + 1, t.size() - npos);
}
