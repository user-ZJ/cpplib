#include "cppjieba/Jieba.hpp"
#include <string>
#include <vector>

namespace BASE_NAMESPACE {

// const char* const DICT_PATH = "../src/frontend/data/jiebadict/jieba.dict.utf8";
// const char* const HMM_PATH = "../src/frontend/data/jiebadict/hmm_model.utf8";
// const char* const USER_DICT_PATH = "../src/frontend/data/jiebadict/user.dict.utf8";
// const char* const IDF_PATH = "../src/frontend/data/jiebadict/idf.utf8";
// const char* const STOP_WORD_PATH = "../src/frontend/data/jiebadict/stop_words.utf8";

class JiebaSingleton {
 public:
  static JiebaSingleton &instance() {
    static JiebaSingleton s;
    return s;
  }

  int loadDict(const std::string &dictPath) {
    std::string DICT_PATH = dictPath + "/jieba.dict.utf8";
    std::string HMM_PATH = dictPath + "/hmm_model.utf8";
    std::string USER_DICT_PATH = dictPath + "/user.dict.utf8";
    std::string IDF_PATH = dictPath + "/idf.utf8";
    std::string STOP_WORD_PATH = dictPath + "/stop_words.utf8";
    jb.reset(new cppjieba::Jieba(DICT_PATH.c_str(), HMM_PATH.c_str(), USER_DICT_PATH.c_str(), IDF_PATH.c_str(),
                                 STOP_WORD_PATH.c_str()));
    return 0;
  }

  std::vector<std::string> cut(const std::string &s) {
    std::vector<std::string> words;
    jb->Cut(s, words, true);
    return words;
  }

 private:
  std::shared_ptr<cppjieba::Jieba> jb;
  JiebaSingleton() {}
  ~JiebaSingleton() {}
};

};  // namespace BASE_NAMESPACE