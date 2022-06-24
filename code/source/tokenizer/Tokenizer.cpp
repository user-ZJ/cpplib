#include "Tokenizer.h"
#include "file-util.h"
#include "logging.h"
#include "regex-util.h"
#include "string-util.h"
#include <fstream>
#include <iostream>
#include <libgen.h>
#include <regex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace BASE_NAMESPACE {

static std::wregex chineseToken(
  L"[\u4e00-\u9fa5]|[a-z|A-Z]+|\\d|[-\\(\\)\\[\\]\\{\\}\\.'\":?!,;/]|[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]");
static std::wregex isEnglish(L"[a-zA-Z][a-zA-Z'\\-]*");
static std::wregex englishToken(L"[\"']|[a-z|A-Z]+|\\d");

static std::wstring tokenPatten =
  LR"([\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]|[\u4e00-\u9fa5]|[A-Za-z]+|[\.\?!,;:'"\(\)\[\]\{\}<>/\-\\$@`~#%^\*_\+=\|])";


Tokenizer::Tokenizer(bool padding) : vocabLoaded(false) {
  needPadding = padding;
}
Tokenizer::~Tokenizer() {}

int Tokenizer::loadVocab(const std::string &vocabPath) {
  LOG(INFO) << "Tokenizer loadVocab:" << vocabPath;
  if (!is_exist(vocabPath.c_str())) {
    LOG(ERROR) << vocabPath << " not exist,please check!!!";
    return -1;
  }
  auto buff = file_to_buff(vocabPath.c_str());
  int res = loadVocab(buff.data(), buff.size());
  if (res != 0)
    LOG(ERROR) << "Tokenizer loadVocab error";
  LOG(INFO) << "Tokenizer loadVocab success";
  return res;
}

int Tokenizer::loadVocab(const char *buffer, const size_t &size) {
  std::istringstream ss(std::string((char *)buffer, size));
  LOG(INFO) << "Tokenizer loadVocab from buffer";
  int id = 0;
  std::string word;
  while (ss >> word) {
    token_id_dict[word] = id;
    id_token_dict[id] = word;
    id++;
  }
  //保证有[UNK]，在encode中，不在词表中的字使用[UNK]
  if (token_id_dict.find(std::string("[UNK]")) == token_id_dict.end()
      && token_id_dict.find(std::string("[CLS]")) == token_id_dict.end()
      && token_id_dict.find(std::string("[SEP]")) == token_id_dict.end()) {
    LOG(ERROR) << "vocab don't have [UNK] or [CLS] or [SEP]";
    return -2;
  }
  vocabLoaded = true;
  LOG(INFO) << "Tokenizer loadVocab from buffer success";
  return 0;
}

std::vector<std::vector<std::string>> Tokenizer::tokenize(const std::string &text, int max_len) {
  VLOG(3) << "Tokenizer tokenize";
  std::vector<std::vector<std::string>> result;
  if (!vocabLoaded) {
    LOG(ERROR) << "vocab not loaded,please call loadVocab before";
    return result;
  }
  if (text.size() == 0) {
    LOG(ERROR) << "tokenize get empty text";
    return result;
  }
  int part = 1;  //长字符串每max_len被切分为一段，part表示段的个数
  //匹配中文/中文标点/英文/英文标点

  std::vector<std::vector<std::wstring>> matchs;
  bool res = REGEX::searchAll(to_wstring(text), tokenPatten, &matchs);
  std::vector<std::string> tokenized;
  for (const auto &m : matchs) {
    //处理英文，英文一个单词可能被拆分为多个token
    if (REGEX::match(m[0], to_wstring(REGEX::ENWord))) {
      std::string estr = to_string(m[0]);
      for (auto sub : wordPieceTokenize(estr)) {
        tokenized.push_back(sub);
      }
    } else {
      tokenized.push_back(to_string(m[0]));
    }
  }
  if (tokenized.size() > max_len)
    part = tokenized.size() / max_len + 1;
  for (int i = 0; i < part; i++) {
    int beg = i * max_len;
    int end = (i + 1) * max_len > tokenized.size() ? tokenized.size() : (i + 1) * max_len;
    std::vector<std::string> seq;
    // 添加起始符号和结束符号
    seq.emplace_back(std::string("[CLS]"));
    for (int j = beg; j < end; j++) {
      seq.push_back(tokenized[j]);
    }
    if (needPadding && (end != (i + 1) * max_len)) {
      for (int j = end; j < (i + 1) * max_len; j++)
        seq.push_back("[PAD]");
    }
    seq.emplace_back("[SEP]");
    result.push_back(seq);
  }
  return result;
}

std::vector<std::string> Tokenizer::wordPieceTokenize(const std::string &word) {
  // 英文单词拆分成subword
  std::string lower = word;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  std::vector<std::string> result;
  size_t start = 0, end = 0;
  while (start < lower.length()) {
    std::string sub = "";
    end = lower.length();
    while (end > start) {
      sub = lower.substr(start, end - start);
      if (start > 0) {
        sub = "##" + sub;
      }
      if (token_id_dict.find(sub) != token_id_dict.end())
        break;
      end -= 1;
    }
    if (start == end) {
      result.push_back(word);
      return result;
    } else {
      result.push_back(sub);
      start = end;
    }
  }
  return result;
}

std::vector<std::vector<int>> Tokenizer::encode(const std::string &text) {
  VLOG(3) << "Tokenizer encode text";
  if (text.size() == 0)
    LOG(ERROR) << "encode get empty text";
  std::vector<std::vector<std::string>> res_t = tokenize(text);

  std::vector<std::vector<int>> result;
  for (int i = 0; i < res_t.size(); i++) {
    std::vector<int> sub;
    for (int j = 0; j < res_t[i].size(); j++) {
      if (token_id_dict.find(res_t[i][j]) != token_id_dict.end()) {
        sub.push_back(token_id_dict[res_t[i][j]]);
      } else {
        sub.push_back(token_id_dict[std::string("[UNK]")]);
      }
    }
    result.emplace_back(sub);
  }
  return result;
}

std::vector<std::vector<int>> Tokenizer::encode(const std::vector<std::vector<std::string>> &tokens) {
  VLOG(3) << "Tokenizer encode tokens";
  std::vector<std::vector<int>> result;
  for (int i = 0; i < tokens.size(); i++) {
    std::vector<int> sub;
    for (int j = 0; j < tokens[i].size(); j++) {
      if (token_id_dict.find(tokens[i][j]) != token_id_dict.end()) {
        sub.push_back(token_id_dict[tokens[i][j]]);
      } else {
        sub.push_back(token_id_dict[std::string("[UNK]")]);
      }
    }
    result.emplace_back(sub);
  }
  return result;
}

std::vector<std::string> Tokenizer::decode(const std::vector<int> &preds) {
  VLOG(3) << "Tokenizer decode std::vector int";
  std::vector<std::string> result;
  for (const auto &p : preds) {
    if (id_token_dict.count(p))
      result.push_back(id_token_dict[p]);
    else
      result.push_back(std::string("[UNK]"));
  }
  return result;
}

std::string Tokenizer::decode(int pred) {
  if (id_token_dict.count(pred))
    return id_token_dict[pred];
  else
    return std::string("[UNK]");
}

int Tokenizer::getLabelLen(){
  return id_token_dict.size();
}

};  // namespace BASE_NAMESPACE
