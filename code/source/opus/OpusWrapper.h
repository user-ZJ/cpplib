#pragma once
#include <vector>

namespace BASE_NAMESPACE{

int OpusEncode(const std::vector<short> &audiodata,int sampleRate,std::vector<unsigned char> &buff);

int OpusDecode(const std::vector<unsigned char> &opusdata,int sampleRate,std::vector<short> &buff);

}