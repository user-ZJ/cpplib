#pragma once
#include "utils/file-util.h"
#include "utils/logging.h"
#include <cereal/archives/json.hpp>
#include <cereal/archives/optional_json.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>
#include <optional>
#include <string>

#ifndef MESSAGE_DESERIALIZE
#define MESSAGE_DESERIALIZE                                                    \
  int deserialize(const std::string &str) {                                    \
    try {                                                                      \
      std::stringstream ss(str);                                               \
      cereal::JSONInputArchive archive(ss);                                    \
      serialize(archive);                                                      \
      return 0;                                                                \
    } catch (std::exception & e) {                                             \
      LOG(ERROR) << e.what();                                                  \
      return 1;                                                                \
    }                                                                          \
  }
#endif

#ifndef MESSAGE_SERIALIZE
#define MESSAGE_SERIALIZE                                                      \
  std::string serialize() {                                                    \
    std::stringstream ss;                                                      \
    try {                                                                      \
      cereal::JSONOutputArchive archive(ss,cereal::JSONOutputArchive::Options{5});                                   \
      serialize(archive);                                                      \
    } catch (std::exception & e) {                                             \
      LOG(ERROR) << e.what();                                                  \
    }                                                                          \
    return ss.str();                                                           \
  }
#endif

struct RequestParam{
  std::string name;
  int max_token;
  int max_max_token;
  int min_max_token;
  float temperature;
  float max_temperature;
  float min_temperature;
  float top_p;
  float max_top_p;
  float min_top_p;
  float frequency_penalty;
  float max_frequency_penalty;
  float min_frequency_penalty;
  float presence_penalty;
  float max_presence_penalty;
  float min_presence_penalty;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(name));
    ar(CEREAL_NVP(max_token));
    ar(CEREAL_NVP(max_max_token));
    ar(CEREAL_NVP(min_max_token));
    ar(CEREAL_NVP(temperature));
    ar(CEREAL_NVP(max_temperature));
    ar(CEREAL_NVP(min_temperature));
    ar(CEREAL_NVP(top_p));
    ar(CEREAL_NVP(max_top_p));
    ar(CEREAL_NVP(min_top_p));
    ar(CEREAL_NVP(frequency_penalty));
    ar(CEREAL_NVP(max_frequency_penalty));
    ar(CEREAL_NVP(min_frequency_penalty));
    ar(CEREAL_NVP(presence_penalty));
    ar(CEREAL_NVP(max_presence_penalty));
    ar(CEREAL_NVP(min_presence_penalty));
  }
};

struct ModelConfig {
  std::string name;
  uint32_t hash;
  std::string description;
  uint32_t created = static_cast<uint32_t>(std::time(nullptr));
  std::map<std::string, std::string> url;
  std::string api_type;
  std::string api_key;
  std::vector<RequestParam> request_params;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(name));
    ar(CEREAL_NVP(url));
    ar(CEREAL_NVP(api_type));
    ar(CEREAL_NVP(request_params));
    ar(cereal::make_optional_nvp("api_key", api_key));
    ar(cereal::make_optional_nvp("hash", hash));
  }
  MESSAGE_SERIALIZE;
  int deserialize(const std::string &str) {                                    
    try {                                                                      
      std::stringstream ss(str);                                               
      cereal::JSONInputArchive archive(ss);                                    
      serialize(archive);
      std::hash<std::string> hasher;
      hash = hasher(name);                                       
      return 0;                                                                
    } catch (std::exception & e) {                                             
      LOG(ERROR) << e.what();                                                  
      return 1;                                                                
    }                                                                          
  }
};

class ModelMaps {
public:
  std::map<std::string, ModelConfig> models;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(models));
  }
  MESSAGE_SERIALIZE;
  MESSAGE_DESERIALIZE;

  static ModelMaps &instance(){
    static ModelMaps ins("../data/config.json");
    return ins;
  }
  
private:
  ModelMaps(const std::string &config) {
    CHECK(DMAI::is_exist(config.c_str())) << config + " not exist!";
    std::string config_str = DMAI::file_to_str(config.c_str());
    deserialize(config_str);
    LOG(INFO) << "load config success";
  }
 
};

namespace cereal {
//! Saving for std::map<std::string, std::string> for text based archives
// Note that this shows off some internal cereal traits such as EnableIf,
// which will only allow this template to be instantiated if its predicates
// are true
template <
    class Archive, class C, class A,
    traits::EnableIf<traits::is_text_archive<Archive>::value> = traits::sfinae>
inline void save(Archive &ar,
                 std::map<std::string, ModelConfig, C, A> const &map) {
  for (const auto &i : map)
    ar(cereal::make_nvp(i.first, i.second));
}

template <
    class Archive, class C, class A,
    traits::EnableIf<traits::is_text_archive<Archive>::value> = traits::sfinae>
inline void save(Archive &ar,
                 std::map<std::string, std::string, C, A> const &map) {
  for (const auto &i : map)
    ar(cereal::make_nvp(i.first, i.second));
}

//! Loading for std::map<std::string, std::string> for text based archives
template <
    class Archive, class C, class A,
    traits::EnableIf<traits::is_text_archive<Archive>::value> = traits::sfinae>
inline void load(Archive &ar, std::map<std::string, std::string, C, A> &map) {
  map.clear();

  auto hint = map.begin();
  while (true) {
    const auto namePtr = ar.getNodeName();

    if (!namePtr)
      break;

    std::string key = namePtr;
    std::string value;
    ar(value);
    hint = map.emplace_hint(hint, std::move(key), std::move(value));
  }
}

template <
    class Archive, class C, class A,
    traits::EnableIf<traits::is_text_archive<Archive>::value> = traits::sfinae>
inline void load(Archive &ar, std::map<std::string, ModelConfig, C, A> &map) {
  map.clear();

  auto hint = map.begin();
  while (true) {
    const auto namePtr = ar.getNodeName();

    if (!namePtr)
      break;

    std::string key = namePtr;
    ModelConfig value;
    ar(value);
    hint = map.emplace_hint(hint, std::move(key), std::move(value));
  }
}
} // namespace cereal


