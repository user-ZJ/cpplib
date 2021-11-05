/*
 * @Author: zack 
 * @Date: 2021-11-01 10:56:37 
 * @Last Modified by: zack
 * @Last Modified time: 2021-11-01 11:35:31
 */
#include "Tokenizer.h"
#include "string-util.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <libgen.h>
#include <regex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace BASE_NAMESPACE {

//中文标点
//。 ？ ！ ， 、 ； ： “ ” ‘ ' （ ） 《 》 〈 〉 【 】 『 』 「 」 ﹃ ﹄ 〔 〕 … — ～ ﹏ ￥
//[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]
static std::wregex chineseToken(L"[\u4e00-\u9fa5]|[a-z|A-Z]+|\\d|-|[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]");
static std::wregex isEnglish(L"[a-zA-Z][a-zA-Z'\\-]*");
static std::wregex englishToken(L"[\"']|[a-z|A-Z]+|\\d");


ZHTokenizer::ZHTokenizer(bool padding) :
  vocabLoaded(false) {
  needPadding = padding;
}
ZHTokenizer::~ZHTokenizer() {}

int ZHTokenizer::loadVocab(std::string vocabPath) {
  LOG(INFO)<<"ZHTokenizer loadVocab:"<<vocabPath.c_str();
  struct stat fs;
  if (stat(vocabPath.c_str(), &fs) != 0) {
    LOG(ERROR)<<"model file error;"<<vocabPath.c_str()<<" not exist,please check!!!";
    return 1;
  }
  if (vocabPath.find("vocab_zh.txt") == std::string::npos) {
    LOG(ERROR)<<"ZHTokenizer vocab name must be vocab_zh.txt,please check if the wrong file is loaded";
    return 1;
  }
  std::ifstream fin(vocabPath);
  int id = 0;
  std::string word;
  while (fin >> word) {
    vocab_dict[to_wstring(word)] = id;
    id++;
  }
  id_to_label = {
    {0, L""},
    {1, L"，"},
    {2, L"。"},
    {3, L"？"},
    {4, L"！"},
    {5, L""}  //"[PAD]"
  };
  //保证有[UNK]，在encode中，不在词表中的字使用[UNK]
  if (vocab_dict.find(std::wstring(L"[UNK]")) == vocab_dict.end() && vocab_dict.find(std::wstring(L"[CLS]")) == vocab_dict.end() && vocab_dict.find(std::wstring(L"[SEP]")) == vocab_dict.end()) {
    LOG(ERROR)<<vocabPath.c_str()<<" don't have [UNK] or [CLS] or [SEP]";
    return 2;
  }
  vocabLoaded = true;
  LOG(INFO)<<"ZHTokenizer loadVocab success";
  return 0;
}

int ZHTokenizer::loadVocab(const char *buffer, size_t size) {
  std::istringstream ss(std::string((char *)buffer, size));
  LOG(INFO)<<"ZHTokenizer loadVocab from buffer";
  int id = 0;
  std::string word;
  while (ss >> word) {
    vocab_dict[to_wstring(word)] = id;
    id++;
  }
  id_to_label = {
    {0, L""},
    {1, L"，"},
    {2, L"。"},
    {3, L"？"},
    {4, L"！"},
    {5, L""}  //"[PAD]"
  };
  //保证有[UNK]，在encode中，不在词表中的字使用[UNK]
  if (vocab_dict.find(std::wstring(L"[UNK]")) == vocab_dict.end() && vocab_dict.find(std::wstring(L"[CLS]")) == vocab_dict.end() && vocab_dict.find(std::wstring(L"[SEP]")) == vocab_dict.end()) {
    LOG(ERROR)<<"vocab don't have [UNK] or [CLS] or [SEP]\n";
    return 2;
  }
  vocabLoaded = true;
  LOG(INFO)<<"ZHTokenizer loadVocab success";
  return 0;
}

std::vector<std::vector<std::wstring>> ZHTokenizer::tokenize(const std::wstring &text) {
  VLOG(4)<<"ZHTokenizer tokenize";
  if (!vocabLoaded) {
    LOG(ERROR)<<"vocab not loaded,please call loadVocab before";
    return {};
  }
  if (text.size() == 0)
    LOG(ERROR)<<"tokenize get empty text\n";
  std::vector<std::vector<std::wstring>> result;
  int part = 1;  //长字符串每MAX_TEXT_LEN被切分为一段，part表示段的个数
  //匹配中文/中文标点/英文/-
  std::wsregex_token_iterator it{text.begin(), text.end(), chineseToken, 0};
  std::vector<std::wstring> splited{it, {}};
  std::vector<std::wstring> tokenized;
  //处理英文，英文一个单词可能被拆分为多个token
  for (auto sp : splited) {
    if (std::regex_match(sp, isEnglish)) {
      //如果是英文，转换为小写
      std::transform(sp.begin(), sp.end(), sp.begin(), ::tolower);
      for (auto sub : wordPieceTokenize(sp)) {
        tokenized.push_back(sub);
      }
    } else {
      tokenized.push_back(sp);
    }
  }
  if (tokenized.size() > MAX_TEXT_LEN)
    part = tokenized.size() / MAX_TEXT_LEN + 1;
  for (int i = 0; i < part; i++) {
    int beg = i * MAX_TEXT_LEN;
    int end = (i + 1) * MAX_TEXT_LEN > tokenized.size() ? tokenized.size() : (i + 1) * MAX_TEXT_LEN;
    std::vector<std::wstring> seq;
    // 添加起始符号和结束符号
    seq.emplace_back(std::wstring(L"[CLS]"));
    for (int j = beg; j < end; j++) {
      seq.push_back(tokenized[j]);
    }
    seq.emplace_back(std::wstring(L"[SEP]"));
    result.push_back(seq);
  }
  return result;
}

std::vector<std::wstring> ZHTokenizer::wordPieceTokenize(const std::wstring &word) {
  //word内分成subword
  std::vector<std::wstring> result;
  size_t start = 0, end = 0;
  while (start < word.length()) {
    std::wstring sub = L"";
    end = word.length();
    while (end > start) {
      sub = word.substr(start, end - start);
      if (start > 0) {
        sub = L"##" + sub;
      }
      if (vocab_dict.find(sub) != vocab_dict.end())
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

std::vector<std::vector<float>> ZHTokenizer::encode(const std::wstring &text) {
  VLOG(4)<<"ZHTokenizer encode text";
  if (text.size() == 0)
    LOG(ERROR)<<"encode get empty text\n";
  std::vector<std::vector<std::wstring>> res_t = tokenize(text);
  if (needPadding) {
    std::vector<std::vector<float>> result(res_t.size(), std::vector<float>(MAX_SEQ_LEN, 0.0f));
    for (int i = 0; i < res_t.size(); i++) {
      for (int j = 0; j < res_t[i].size(); j++) {
        if (vocab_dict.find(res_t[i][j]) != vocab_dict.end()) {
          result[i][j] = float(vocab_dict[res_t[i][j]]);
        } else {
          result[i][j] = float(vocab_dict[std::wstring(L"[UNK]")]);
        }
      }
    }
    return result;
  } else {
    std::vector<std::vector<float>> result;
    for (int i = 0; i < res_t.size(); i++) {
      std::vector<float> sub;
      for (int j = 0; j < res_t[i].size(); j++) {
        if (vocab_dict.find(res_t[i][j]) != vocab_dict.end()) {
          sub.push_back(float(vocab_dict[res_t[i][j]]));
        } else {
          sub.push_back(float(vocab_dict[std::wstring(L"[UNK]")]));
        }
      }
      result.emplace_back(sub);
    }
    return result;
  }
}

std::vector<std::vector<float>> ZHTokenizer::encode(const std::vector<std::vector<std::wstring>> &tokens) {
  VLOG(4)<<"ZHTokenizer encode tokens";
  if (needPadding) {
    //padding result to MAX_SEQ_LEN
    std::vector<std::vector<float>> result(tokens.size(), std::vector<float>(MAX_SEQ_LEN, 0.0f));
    for (int i = 0; i < tokens.size(); i++) {
      for (int j = 0; j < tokens[i].size(); j++) {
        if (vocab_dict.find(tokens[i][j]) != vocab_dict.end()) {
          result[i][j] = float(vocab_dict[tokens[i][j]]);
        } else {
          result[i][j] = float(vocab_dict[std::wstring(L"[UNK]")]);
        }
      }
    }
    return result;
  } else {
    std::vector<std::vector<float>> result;
    for (int i = 0; i < tokens.size(); i++) {
      std::vector<float> sub;
      for (int j = 0; j < tokens[i].size(); j++) {
        if (vocab_dict.find(tokens[i][j]) != vocab_dict.end()) {
          sub.push_back(float(vocab_dict[tokens[i][j]]));
        } else {
          sub.push_back(vocab_dict[std::wstring(L"[UNK]")]);
        }
      }
      result.emplace_back(sub);
    }
    return result;
  }
}

std::vector<std::wstring> ZHTokenizer::decode(const std::vector<int> &preds) {
  VLOG(4)<<"ZHTokenizer decode vector int";
  std::vector<std::wstring> result;
  for (auto p : preds) {
    result.push_back(id_to_label[p]);
  }
  return result;
}

std::wstring ZHTokenizer::decode(int pred) {
  return id_to_label[pred];
}

// english tokenizer
ENTokenizer::ENTokenizer(bool padding) :
  vocabLoaded(false) {
  needPadding = padding;
}
ENTokenizer::~ENTokenizer() {}

int ENTokenizer::loadVocab(std::string vocabPath) {
  LOG(INFO)<<"ENTokenizer load vocab:"<<vocabPath.c_str();
  struct stat fs;
  if (stat(vocabPath.c_str(), &fs) != 0) {
    LOG(ERROR)<<"model file error;"<<vocabPath.c_str()<<" not exist,please check!!!";
    return 1;
  }
  if (vocabPath.find("vocab_en.txt") == std::string::npos) {
    LOG(ERROR)<<"ZHTokenizer vocab name must be vocab_zh.txt,please check if the wrong file is loaded";
    return 1;
  }
  std::ifstream fin(vocabPath);
  int id = 0;
  std::string word;
  while (fin >> word) {
    vocab_dict[to_wstring(word)] = id;
    id++;
  }
  id_to_label = {
    {0, L""},
    {1, L","},
    {2, L"."},
    {3, L"?"},
    {4, L"!"},
    {5, L""}  //"[PAD]"
  };
  //保证有[UNK]，在encode中，不在词表中的字使用[UNK]
  if (vocab_dict.find(std::wstring(L"[UNK]")) == vocab_dict.end() && vocab_dict.find(std::wstring(L"[CLS]")) == vocab_dict.end() && vocab_dict.find(std::wstring(L"[SEP]")) == vocab_dict.end()) {
    LOG(ERROR)<<vocabPath<<" don't have [UNK] or [CLS] or [SEP]\n";
    return 2;
  }
  vocabLoaded = true;
  LOG(INFO)<<"ENTokenizer load vocab success";
  return 0;
}

int ENTokenizer::loadVocab(const char *buffer, size_t size) {
  std::istringstream ss(std::string((char *)buffer, size));
  LOG(INFO)<<"ENTokenizer loadVocab from buffer";
  int id = 0;
  std::string word;
  while (ss >> word) {
    vocab_dict[to_wstring(word)] = id;
    id++;
  }
  id_to_label = {
    {0, L""},
    {1, L","},
    {2, L"."},
    {3, L"?"},
    {4, L"!"},
    {5, L""}  //"[PAD]"
  };
  //保证有[UNK]，在encode中，不在词表中的字使用[UNK]
  if (vocab_dict.find(std::wstring(L"[UNK]")) == vocab_dict.end() && vocab_dict.find(std::wstring(L"[CLS]")) == vocab_dict.end() && vocab_dict.find(std::wstring(L"[SEP]")) == vocab_dict.end()) {
    LOG(INFO)<<"vocab don't have [UNK] or [CLS] or [SEP]\n";
    return 2;
  }
  vocabLoaded = true;
  LOG(INFO)<<"ENTokenizer load vocab success";
  return 0;
}

std::vector<std::vector<std::wstring>> ENTokenizer::tokenize(const std::wstring &text) {
  VLOG(4)<<"ENTokenizer tokenize";
  if (!vocabLoaded) {
    LOG(ERROR)<<"vocab not loaded,please call loadVocab before";
    return {};
  }
  if (text.size() == 0)
    LOG(ERROR)<<"tokenize get empty text\n";
  //将英文转换为小写
  std::wstring text_low = text;
  std::transform(text_low.begin(), text_low.end(), text_low.begin(), ::tolower);
  std::vector<std::vector<std::wstring>> result;
  int part = 1;  //长字符串每MAX_TEXT_LEN被切分为一段，part表示段的个数
  std::wsregex_token_iterator it{text_low.begin(), text_low.end(), englishToken, 0};
  std::vector<std::wstring> tokenized{it, {}};
  if (tokenized.size() > MAX_TEXT_LEN)
    part = tokenized.size() / MAX_TEXT_LEN + 1;
  for (int i = 0; i < part; i++) {
    int beg = i * MAX_TEXT_LEN;
    int end = (i + 1) * MAX_TEXT_LEN > tokenized.size() ? tokenized.size() : (i + 1) * MAX_TEXT_LEN;
    std::vector<std::wstring> seq;
    // 添加起始符号和结束符号
    seq.emplace_back(std::wstring(L"[CLS]"));
    for (int j = beg; j < end; j++) {
      for (auto sub : wordPieceTokenize(tokenized[j])) {
        seq.push_back(sub);
      }
    }
    seq.emplace_back(std::wstring(L"[SEP]"));
    result.push_back(seq);
  }
  VLOG(4)<<"ENTokenizer tokenize end";
  return result;
}

std::vector<std::wstring> ENTokenizer::wordPieceTokenize(const std::wstring &word) {
  //word内分成subword
  std::vector<std::wstring> result;
  size_t start = 0, end = 0;
  while (start < word.length()) {
    std::wstring sub = L"";
    end = word.length();
    while (end > start) {
      sub = word.substr(start, end - start);
      if (start > 0) {
        sub = L"##" + sub;
      }
      if (vocab_dict.find(sub) != vocab_dict.end())
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

std::vector<std::vector<float>> ENTokenizer::encode(const std::wstring &text) {
  VLOG(4)<<"ENTokenizer encode:"<<to_string(text);
  if (text.size() == 0)
    LOG(ERROR)<<"encode get empty text\n";
  std::vector<std::vector<std::wstring>> res_t = tokenize(text);
  if (needPadding) {
    std::vector<std::vector<float>> result(res_t.size(), std::vector<float>(MAX_SEQ_LEN, 0.0f));
    for (int i = 0; i < res_t.size(); i++) {
      for (int j = 0; j < res_t[i].size(); j++) {
        if (vocab_dict.find(res_t[i][j]) != vocab_dict.end()) {
          result[i][j] = float(vocab_dict[res_t[i][j]]);
        } else {
          result[i][j] = float(vocab_dict[std::wstring(L"[UNK]")]);
        }
      }
    }
    return result;
  } else {
    std::vector<std::vector<float>> result;
    for (int i = 0; i < res_t.size(); i++) {
      std::vector<float> sub;
      for (int j = 0; j < res_t[i].size(); j++) {
        if (vocab_dict.find(res_t[i][j]) != vocab_dict.end()) {
          sub.push_back(float(vocab_dict[res_t[i][j]]));
        } else {
          sub.push_back(float(vocab_dict[std::wstring(L"[UNK]")]));
        }
      }
      result.emplace_back(sub);
    }
    return result;
  }
}

std::vector<std::vector<float>> ENTokenizer::encode(const std::vector<std::vector<std::wstring>> &tokens) {
  VLOG(4)<<"ENTokenizer encode tokens";
  if (needPadding) {
    //padding result to MAX_SEQ_LEN
    std::vector<std::vector<float>> result(tokens.size(), std::vector<float>(MAX_SEQ_LEN, 0.0f));
    for (int i = 0; i < tokens.size(); i++) {
      for (int j = 0; j < tokens[i].size(); j++) {
        if (vocab_dict.find(tokens[i][j]) != vocab_dict.end()) {
          result[i][j] = float(vocab_dict[tokens[i][j]]);
        } else {
          result[i][j] = float(vocab_dict[std::wstring(L"[UNK]")]);
        }
      }
    }
    return result;
  } else {
    std::vector<std::vector<float>> result;
    for (int i = 0; i < tokens.size(); i++) {
      std::vector<float> sub;
      for (int j = 0; j < tokens[i].size(); j++) {
        if (vocab_dict.find(tokens[i][j]) != vocab_dict.end()) {
          sub.push_back(float(vocab_dict[tokens[i][j]]));
        } else {
          sub.push_back(vocab_dict[std::wstring(L"[UNK]")]);
        }
      }
      result.emplace_back(sub);
    }
    return result;
  }
}

std::vector<std::wstring> ENTokenizer::decode(const std::vector<int> &preds) {
  std::vector<std::wstring> result;
  for (auto p : preds) {
    result.push_back(id_to_label[p]);
  }
  return result;
}

std::wstring ENTokenizer::decode(int pred) {
  return id_to_label[pred];
}

}; 
