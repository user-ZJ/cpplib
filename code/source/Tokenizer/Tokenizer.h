/*
 * @Author: zack 
 * @Date: 2021-11-01 11:09:51 
 * @Last Modified by: zack
 * @Last Modified time: 2021-11-01 11:22:08
 */
#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace BASE_NAMESPACE {

#define MAX_SEQ_LEN 122
#define MAX_TEXT_LEN 120

class TokenizerBase {
 public:
  TokenizerBase()=default;
  virtual ~TokenizerBase()=default;
  virtual int loadVocab(std::string vocabPath) = 0;
  virtual int loadVocab(const char *buffer, size_t size) = 0;
  //将句子拆分为单字
  virtual std::vector<std::vector<std::wstring>> tokenize(const std::wstring &text) = 0;
  //将句子拆分为单字,并转换为词表对应的id
  virtual std::vector<std::vector<float>> encode(const std::wstring &text) = 0;
  virtual std::vector<std::vector<float>> encode(const std::vector<std::vector<std::wstring>> &tokens) = 0;
  virtual std::vector<std::wstring> decode(const std::vector<int> &preds) = 0;
  virtual std::wstring decode(int pred) = 0;
  virtual int getLabelLen() = 0;
};

class ZHTokenizer : public TokenizerBase {
 public:
  ZHTokenizer(bool padding = false);
  ~ZHTokenizer();
  //加载词表，词表较大，建议进行预加载，不要每次调用都加载
  //加载成功，返回0
  int loadVocab(std::string vocabPath) override;
  int loadVocab(const char *buffer, size_t size) override;
  //将句子拆分为单字
  std::vector<std::vector<std::wstring>> tokenize(const std::wstring &text) override;
  //word内分成subword
  std::vector<std::wstring> wordPieceTokenize(const std::wstring &word);
  //将句子拆分为单字,并转换为词表对应的id
  std::vector<std::vector<float>> encode(const std::wstring &text) override;
  std::vector<std::vector<float>> encode(const std::vector<std::vector<std::wstring>> &tokens) override;
  std::vector<std::wstring> decode(const std::vector<int> &preds) override;
  std::wstring decode(int pred) override;
  int getLabelLen() override {
    int len = static_cast<int>(id_to_label.size());
    return len;
  }

 private:
  bool vocabLoaded = false;
  bool needPadding;
  std::unordered_map<std::wstring, int> vocab_dict;
  std::unordered_map<int, std::wstring> id_to_label;
};

class ENTokenizer : public TokenizerBase {
 public:
  ENTokenizer(bool padding = false);
  ~ENTokenizer();
  //加载词表，词表较大，建议进行预加载，不要每次调用都加载
  //加载成功，返回0
  int loadVocab(std::string vocabPath) override;
  int loadVocab(const char *buffer, size_t size) override;
  //将句子拆分为单字
  std::vector<std::vector<std::wstring>> tokenize(const std::wstring &text) override;
  //word内分成subword
  std::vector<std::wstring> wordPieceTokenize(const std::wstring &word);
  //将句子拆分为单字,并转换为词表对应的id
  std::vector<std::vector<float>> encode(const std::wstring &text) override;
  std::vector<std::vector<float>> encode(const std::vector<std::vector<std::wstring>> &tokens) override;
  std::vector<std::wstring> decode(const std::vector<int> &preds) override;
  std::wstring decode(int pred) override;
  int getLabelLen() override {
    return static_cast<int>(id_to_label.size());
  }

 private:
  bool vocabLoaded = false;
  bool needPadding;
  std::unordered_map<std::wstring, int> vocab_dict;
  std::unordered_map<int, std::wstring> id_to_label;
};

};