#pragma once

#include <fstream.h>  

std::vector<char> file_to_buff(const char *path){
    std::ifstream in (filename, ios::in|ios::binary|ios::ate);
    long size = in.tellg(); 
    in.seekg (0, ios::beg); 
    std::vector<char> buffer(size);
    in.read(buffer.data(), size);
    in.close();
    return buffer;
}