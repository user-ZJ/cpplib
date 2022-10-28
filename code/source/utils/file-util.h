/*
 * @Author: zack
 * @Date: 2021-10-13 10:04:28
 * @Last Modified by: zack
 * @Last Modified time: 2022-09-21 15:18:34
 */
#ifndef BASE_FILE_UTIL_H_
#define BASE_FILE_UTIL_H_

#include "logging.h"
#include <fstream>
#include <sstream>
#include <sys/stat.h>

namespace BASE_NAMESPACE {

// 读取文件，存储到buffer中
inline std::vector<char> file_to_buff(const char *path) {
  std::ifstream in(path, std::ios::in | std::ios::binary | std::ios::ate);
  if (!in.good()) {
    LOG(ERROR) << "file not exist," << std::string(path);
    return {};
  }
  long size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  in.read(buffer.data(), size);
  in.close();
  return buffer;
}

inline std::string file_to_str(const char *path){
  auto buffer = file_to_buff(path);
  return std::string(buffer.begin(),buffer.end()); 
}

inline std::stringstream file_to_ss(const char *path) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  std::stringstream buffer;
  if (!in.good()) {
    LOG(ERROR) << "file not exist," << std::string(path);
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
inline int64_t get_file_size(const char *path) {
  struct stat path_stat;

  memset(&path_stat, 0, sizeof(path_stat));
  if (stat(path, &path_stat) == 0) {
    /* Stat returns size taken up by directory entry, so return 0 */
    if (S_ISDIR(path_stat.st_mode)) return 0;

    return path_stat.st_size;
  }
  return 0;
}

inline std::vector<std::string> get_file_lines(const char *path) {
  std::vector<std::string> lines;
  std::ifstream in(path);
  std::string l;
  while (getline(in, l)) {
    lines.push_back(l);
  }
  return lines;
}

inline int write_to_file(const char *path,const std::vector<char> &buff){
  LOG(INFO)<<"write "<<path;
  std::ofstream out (path, std::ios::out|std::ios::binary);
  out.write(buff.data(),buff.size());
  out.close();
  return 0;
}

inline int write_to_file(const char *path,const std::string &msg){
  LOG(INFO)<<"write "<<path;
  std::ofstream out (path, std::ios::out|std::ios::binary);
  out.write(msg.data(),msg.length());
  out.close();
  return 0;
}

};  // namespace BASE_NAMESPACE

#endif