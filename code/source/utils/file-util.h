/*
 * @Author: zack 
 * @Date: 2021-10-13 10:04:28 
 * @Last Modified by: zack
 * @Last Modified time: 2021-10-15 17:48:26
 */
#pragma once

#include <fstream>
#include <sstream>
#include <sys/stat.h>

namespace BASE_NAMESPACE {

// 读取文件，存储到buffer中
inline std::vector<char> file_to_buff(const char *path) {
  std::ifstream in(path, std::ios::in | std::ios::binary | std::ios::ate);
  if (!in.good()) {
    std::cout << "ERROR:file not exist\n";
    return {};
  }
  long size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  in.read(buffer.data(), size);
  in.close();
  return buffer;
}

inline std::stringstream file_to_ss(const char *path) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  std::stringstream buffer;
  if (!in.good()) {
    std::cout << "ERROR:file not exist\n";
    return buffer;
  }
  buffer << in.rdbuf();
  in.close();
  return buffer;
}

// 判断文件是否存在
inline bool is_exist(const char *path) {
  struct stat buffer;
  return (stat(path, &buffer) == 0);
}

//获取文件大小
int64_t get_file_size(const char *path) {
    struct stat path_stat;

    memset(&path_stat, 0, sizeof(path_stat));
    if (stat(path, &path_stat) == 0) {
        /* Stat returns size taken up by directory entry, so return 0 */
        if (S_ISDIR(path_stat.st_mode))
            return 0;

        return path_stat.st_size;
    }   
    return 0;                                                                                                                                                                                                                           
}



}; // namespace BASE_NAMESPACE