#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace DMAI {

template <typename T>
int writeToFile(std::vector<T> data, int seg, std::string filepath) {
  std::ofstream out(filepath.c_str());
  if (out.is_open()) {
    for (int i = 0; i < data.size(); i++) {
      out << std::to_string(data[i]) << " ";
      if (seg != 0 && (i + 1) % seg == 0)
        out << "\n";
    }

    out << "\n";
    out.close();
    return 0;
  }
  return 1;
}

template <typename T>
int writeToFile(std::vector<std::vector<T>> data, std::string filepath) {
  std::ofstream out(filepath.c_str());
  if (out.is_open()) {
    for (auto vec : data) {
      for (auto t : vec)
        out << std::to_string(t) << " ";
      out << "\n";
    }

    out << "\n";
    out.close();
    return 0;
  }
  return 1;
}


// template <typename T>
int readFromFile(std::string filepath,std::vector<float> *data) {
  std::ifstream in(filepath.c_str());
  if (in.is_open()) {
    data->clear();
    std::string tt;
    float d;
    while(in>>tt) {
      d = stof(tt);
      data->push_back(d);
    }
    in.close();
    return 0;
  }
  return 1;
}

};  // namespace DMAI
