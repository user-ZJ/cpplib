#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace BASE_NAMESPACE {

class Tokenizer {
 public:
  Tokenizer(bool padding = false);
  ~Tokenizer();
  //加载词表，词表较大，建议进行预加载，不要每次调用都加载
  //加载成功，返回0
  int loadVocab(const std::string &vocabPath);
  int loadVocab(const char *buffer, const size_t &size);
  /**
   * @brief 将句子拆分为单个字/子词，通过max_len参数控制子句的最大长度
   *
   * @param text 待处理文本
   * @param max_len 子句最大长度
   * @return std::vector<std::vector<std::string>>
   *         子句中被切分的字/子词集合
   */
  std::vector<std::vector<std::string>> tokenize(const std::string &text, int max_len = 120);

  //将句子拆分为单字,并转换为词表对应的id
  std::vector<std::vector<int>> encode(const std::string &text);
  std::vector<std::vector<int>> encode(const std::vector<std::vector<std::string>> &tokens);
  std::vector<std::string> decode(const std::vector<int> &preds);
  std::string decode(int pred);
  int getLabelLen();

 private:
  // word拆分成subword，主要用来将英文单词拆分为子词
  std::vector<std::string> wordPieceTokenize(const std::string &word);
  bool vocabLoaded = false;  //是否加载了词表
  bool needPadding;
  std::unordered_map<std::string, int> token_id_dict;
  std::unordered_map<int, std::string> id_token_dict;
};

};  // namespace BASE_NAMESPACE