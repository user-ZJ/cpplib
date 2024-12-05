#include "utils/flags.h"
#include "utils/logging.h"
#include <cereal/archives/json.hpp>
#include <cereal/archives/optional_json.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>
#include <fstream>
#include <optional>

namespace cereal {
//! Saving for std::map<std::string, std::string> for text based archives
// Note that this shows off some internal cereal traits such as EnableIf,
// which will only allow this template to be instantiated if its predicates
// are true
template <class Archive, class T, class C, class A,
          traits::EnableIf<traits::is_text_archive<Archive>::value> = traits::sfinae>
inline void save(Archive &ar, std::map<std::string, T, C, A> const &map) {
  for (const auto &i : map)
    ar(cereal::make_nvp(i.first, i.second));
}

//! Loading for std::map<std::string, std::string> for text based archives
template <class Archive, class T, class C, class A,
          traits::EnableIf<traits::is_text_archive<Archive>::value> = traits::sfinae>
inline void load(Archive &ar, std::map<std::string, T, C, A> &map) {
  map.clear();

  auto hint = map.begin();
  while (true) {
    const auto namePtr = ar.getNodeName();

    if (!namePtr) break;

    std::string key = namePtr;
    T value;
    ar(value);
    hint = map.emplace_hint(hint, std::move(key), std::move(value));
  }
}
}  // namespace cereal

struct MyData {
  int x, y, z;
  float t;
  std::string word = "你好";

  std::string serialize() {
    std::stringstream ss;
    try {
      {
        cereal::JSONOutputArchive archive(ss);
        save(archive);
      }
    }
    catch (std::exception &e) {
      LOG(ERROR) << e.what();
    }
    return ss.str();
  }

  void deserialize(const std::string &str) {
    try {
      std::stringstream ss(str);
      cereal::JSONInputArchive archive(ss);
      load(archive);
    }
    catch (std::exception &e) {
      LOG(ERROR) << e.what();
    }
  }

  template <class Archive>
  void save(Archive &ar) const {
    ar(CEREAL_NVP(x), CEREAL_NVP(y), CEREAL_NVP(z), CEREAL_NVP(word));
  }

  template <class Archive>
  void load(Archive &ar) {
    ar(cereal::make_optional_nvp("x", x));
    ar(cereal::make_optional_nvp("z", z));
    ar(cereal::make_optional_nvp("t", t));
    ar(CEREAL_NVP(y));
  }

  MyData() : x(0), y(0), z(0) {}
};

struct MyData1 {
  int t = 0;
  std::unique_ptr<MyData> data;

  std::string serialize() {
    std::stringstream ss;
    try {
      {
        cereal::JSONOutputArchive archive(ss);
        save(archive);
      }
    }
    catch (std::exception &e) {
      LOG(ERROR) << e.what();
    }
    return ss.str();
  }

  void deserialize(const std::string &str) {
    try {
      std::stringstream ss(str);
      cereal::JSONInputArchive archive(ss);
      load(archive);
    }
    catch (std::exception &e) {
      LOG(ERROR) << e.what();
    }
  }

  template <class Archive>
  void save(Archive &ar) const {
    ar(CEREAL_NVP(t));
    ar(cereal::make_nvp("data", data));
  }

  template <class Archive>
  void load(Archive &ar) {
    ar(cereal::make_optional_nvp("t", t));
    ar(cereal::make_nvp("data", data));
  }
};

int main(int argc, char *argv[]) {
  google::EnableLogCleaner(30);  // keep your logs for 30 days
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);  // 初始化 glog
  google::SetStderrLogging(google::GLOG_ERROR);
  google::InstallFailureSignalHandler();
  std::stringstream ss;
  // MyData myData;
  // {
  //   cereal::JSONOutputArchive outarchive(std::cout);
  //   myData.save(outarchive);
  // }

  // std::string sjson = R"({"y":1,"z": 3,"i":8})";
  // ss << sjson;
  // {
  //   cereal::JSONInputArchive inarchive(ss);
  //   myData.load(inarchive);
  // }
  // LOG(INFO) << myData.x << " " << myData.y << " " << myData.z;

  MyData1 sdata, sdata2;
  sdata.data = std::make_unique<MyData>();
  std::string str = sdata.serialize();
  LOG(INFO) << str;
  sdata2.deserialize(str);
  LOG(INFO) << sdata2.serialize();

  return 0;
}