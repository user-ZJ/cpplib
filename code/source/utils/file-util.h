/*
 * @Author: zack
 * @Date: 2021-10-13 10:04:28
 * @Last Modified by: zack
 * @Last Modified time: 2022-09-21 15:18:34
 */
#ifndef BASE_FILE_UTIL_H_
#define BASE_FILE_UTIL_H_

#include "logging.h"
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace BASE_NAMESPACE {

class MappedFile {
 public:
  MappedFile(const std::string &path) {
    int fd = open(path.c_str(), O_RDONLY);
    CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

    struct stat sb;
    CHECK(fstat(fd, &sb) == 0) << strerror(errno);
    size = sb.st_size;

    data = (char *)mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    CHECK(data != MAP_FAILED) << strerror(errno);

    CHECK(close(fd) == 0) << strerror(errno);
  }
  ~MappedFile() {
    CHECK(munmap(data, size) == 0) << strerror(errno);
  }

 public:
  char *data;
  size_t size;
};

// 读取文件，存储到buffer中
inline std::vector<char> file_to_buff(const char *path) {
  std::ifstream in(path, std::ios::in | std::ios::binary | std::ios::ate);
  if (!in.good()) {
    LOG(ERROR) << "file not exist:" << std::string(path);
    return {};
  }
  long size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  in.read(buffer.data(), size);
  in.close();
  return buffer;
}

inline std::string file_to_str(const char *path) {
  auto buffer = file_to_buff(path);
  return std::string(buffer.begin(), buffer.end());
}

inline std::stringstream file_to_ss(const char *path) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  std::stringstream buffer;
  if (!in.good()) {
    LOG(ERROR) << "file not exist:" << std::string(path);
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

inline int write_lines_file(const char *path, const std::vector<std::string> &data) {
  std::ofstream out(path);
  if (out.is_open()) {
    for (int i = 0; i < data.size(); i++) {
      out << data[i];
      if(data.size()-1!=i)
        out<<"\n";
    }
    out.close();
    return 0;
  }
  return 1;
}

template <typename T>
inline int writeBinaryFile(const char *path, const std::vector<T> &buff) {
  LOG(INFO) << "write " << path;
  std::ofstream out(path, std::ios::out | std::ios::binary);
  out.write(buff.data(), buff.size() * sizeof(T));
  out.close();
  return 0;
}

inline int writeBinaryFile(const char *path, const std::string &msg) {
  LOG(INFO) << "write " << path;
  std::ofstream out(path, std::ios::out | std::ios::binary);
  out.write(msg.data(), msg.length());
  out.close();
  return 0;
}

template <typename T>
inline int readBinaryFile(const char *path, std::vector<T> &data) {
  LOG(INFO) << "read " << path;
  std::ifstream in(path, std::ios::in | std::ios::binary | std::ios::ate);
  if (!in.good()) {
    LOG(ERROR) << "file not exist:" << std::string(path);
    return {};
  }
  long size = in.tellg();
  in.seekg(0, std::ios::beg);
  data.resize(size / sizeof(T));
  in.read(data.data(), size);
  in.close();
  return 0;
}

template <typename T>
int writeTextFile(const char *path, const std::vector<T> &data, int seg = 0) {
  std::ofstream out(path);
  if (out.is_open()) {
    for (int i = 0; i < data.size(); i++) {
      out << std::to_string(data[i]) << " ";
      if (seg != 0 && (i + 1) % seg == 0) out << "\n";
    }
    out << "\n";
    out.close();
    return 0;
  }
  return 1;
}

template <typename T>
int writeTextFile(const char *path, const std::vector<std::vector<T>> &data) {
  std::ofstream out(path);
  if (out.is_open()) {
    for (auto &vec : data) {
      for (auto &t : vec)
        out << std::to_string(t) << " ";
      out << "\n";
    }

    out << "\n";
    out.close();
    return 0;
  }
  return 1;
}

inline int writeTextFile(const char *path,const std::string &str) {
  std::ofstream out(path);
  if (out.is_open()) {
    out<<str;
    out.close();
    return 0;
  }
  return 1;
}


template <typename T>
int readTextFile(std::string filepath, std::vector<T> &data) {
  std::ifstream in(filepath.c_str());
  if (in.is_open()) {
    data.clear();
    std::string tt;
    float d;
    while (in >> tt) {
      d = stof(tt);
      data.push_back(static_cast<T>(d));
    }
    in.close();
    return 0;
  }
  return 1;
}

};  // namespace BASE_NAMESPACE

#endif