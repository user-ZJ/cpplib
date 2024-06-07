#pragma once
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

struct ConversationMessage {
  std::string role;
  std::string content;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(role));
    ar(CEREAL_NVP(content));
  }
  ConversationMessage() : role(""), content("") {}
};

struct TokenForm {
  std::string id;
  std::string model;
  bool add_special_tokens;
  std::vector<ConversationMessage> messages;
  template <class Archive> void serialize(Archive &ar) {
    ar(cereal::make_optional_nvp("id", id));
    ar(CEREAL_NVP(model));
    ar(cereal::make_optional_nvp("add_special_tokens", add_special_tokens));
    ar(CEREAL_NVP(messages));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct TokenResponse {
  std::string id;
  int64_t created = 0;
  int code = 0;
  std::string message;
  std::string model;
  std::string prompt;
  std::vector<int> token;
  int token_len;
  int timecost;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(id));
    ar(CEREAL_NVP(created));
    ar(CEREAL_NVP(code));
    ar(CEREAL_NVP(message));
    ar(CEREAL_NVP(model));
    ar(CEREAL_NVP(prompt));
    ar(CEREAL_NVP(token));
    ar(CEREAL_NVP(token_len));
    ar(CEREAL_NVP(timecost));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct RoleProfile{
  std::string user_name;
  std::string user_info;
  std::string bot_name;
  std::string bot_info;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(user_name));
    ar(CEREAL_NVP(user_info));
    ar(CEREAL_NVP(bot_name));
    ar(CEREAL_NVP(bot_info));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct ChatForm {
  std::string id;
  std::string model;
  std::vector<ConversationMessage> messages;
  bool stream;             // optional
  bool do_sample;          // optional
  bool ignore_eos;         // optional
  int n;                   // optional
  int num_beams;           // optional
  int top_k;               // optional
  int max_tokens;          // optional
  int min_new_tokens;      // optional
  int max_new_tokens;      // optional
  int max_input_tokens;    // optional
  float top_p;             // optional
  float temperature;       // optional
  float presence_penalty;  // optional
  float frequency_penalty; // optional
  RoleProfile role_profile;
  template <class Archive> void serialize(Archive &ar) {
    ar(cereal::make_optional_nvp("id", id));
    ar(CEREAL_NVP(model));
    ar(CEREAL_NVP(messages));
    ar(cereal::make_optional_nvp("stream", stream));
    ar(cereal::make_optional_nvp("do_sample", do_sample));
    ar(cereal::make_optional_nvp("ignore_eos", ignore_eos));
    ar(cereal::make_optional_nvp("n", n));
    ar(cereal::make_optional_nvp("num_beams", num_beams));
    ar(cereal::make_optional_nvp("top_k", top_k));
    ar(cereal::make_optional_nvp("max_tokens", max_tokens));
    ar(cereal::make_optional_nvp("min_new_tokens", min_new_tokens));
    ar(cereal::make_optional_nvp("max_new_tokens", max_new_tokens));
    ar(cereal::make_optional_nvp("max_input_tokens", max_input_tokens));
    ar(cereal::make_optional_nvp("top_p", top_p));
    ar(cereal::make_optional_nvp("temperature", temperature));
    ar(cereal::make_optional_nvp("presence_penalty", presence_penalty));
    ar(cereal::make_optional_nvp("frequency_penalty", frequency_penalty));
    ar(cereal::make_optional_nvp("role_profile", role_profile));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
  std::string serialize_gpt() {
    std::stringstream ss;
    try {
      cereal::JSONOutputArchive ar(ss);
      ar(CEREAL_NVP(model));
      ar(CEREAL_NVP(messages));
      ar(cereal::make_optional_nvp("stream", stream));
      ar(cereal::make_optional_nvp("n", n));
      ar(cereal::make_optional_nvp("max_tokens", max_tokens));
      ar(cereal::make_optional_nvp("top_p", top_p));
      ar(cereal::make_optional_nvp("temperature", temperature));
      ar(cereal::make_optional_nvp("presence_penalty", presence_penalty));
      ar(cereal::make_optional_nvp("frequency_penalty", frequency_penalty));
    } catch (std::exception &e) {
      LOG(ERROR) << e.what();
    }
    return ss.str();
  }
  ChatForm()
      : stream(false), do_sample(true), ignore_eos(false), n(1), num_beams(1),
        top_k(40), max_tokens(1024), min_new_tokens(1), max_new_tokens(1024),
        max_input_tokens(2048), presence_penalty(0), frequency_penalty(0),
        temperature(0.1), top_p(0.75) {}
};

struct Choice {
  int index;
  ConversationMessage message; // oneof
  ConversationMessage delta;   // oneof
  std::string finish_reason;
  template <class Archive> void save(Archive &ar) const {
    ar(CEREAL_NVP(index));
    if (!message.role.empty())
      ar(cereal::make_optional_nvp("message", message));
    else
      ar(cereal::make_optional_nvp("delta", delta));
    ar(CEREAL_NVP(finish_reason));
  }
  template <class Archive> void load(Archive &ar) {
    ar(CEREAL_NVP(index));
    ar(cereal::make_optional_nvp("message", message));
    ar(cereal::make_optional_nvp("delta", delta));
    ar(CEREAL_NVP(finish_reason));
  }
};

struct Usage {
  int prompt_tokens;
  int completion_tokens;
  int total_tokens;
  template <class Archive> void serialize(Archive &ar) {
    ar(cereal::make_optional_nvp("prompt_tokens", prompt_tokens));
    ar(cereal::make_optional_nvp("completion_tokens", completion_tokens));
    ar(cereal::make_optional_nvp("total_tokens", total_tokens));
  }
  Usage() : prompt_tokens(0), completion_tokens(0), total_tokens(0) {}
};

struct ChatResponse {
  std::string id;
  std::string object;
  int code = 0;        // optional
  std::string message; // optional
  int64_t created = 0;
  std::string model;
  std::vector<Choice> choices;
  Usage usage;
  int timecost;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(id));
    ar(CEREAL_NVP(object));
    ar(cereal::make_optional_nvp("code", code));
    ar(cereal::make_optional_nvp("message", message));
    ar(CEREAL_NVP(created));
    ar(CEREAL_NVP(model));
    ar(CEREAL_NVP(choices));
    ar(CEREAL_NVP(usage));
    ar(cereal::make_optional_nvp("timecost", timecost));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct DMChatForm {
  std::string task_id;
  std::string bot_name;
  std::string bot_version;
  std::string query;
  bool do_sample;                                // optional
  int num_beams;                                 // optional
  int top_k;                                     // optional
  int max_total_tokens;                          // optional
  int min_new_tokens;                            // optional
  int max_new_tokens;                            // optional
  int max_input_tokens;                          // optional
  float top_p;                                   // optional
  float temperature;                             // optional
  float repetition_penalty;                      // optional
  float frequency_penalty;                       // optional
  std::vector<std::vector<std::string>> history; // optional
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(task_id));
    ar(CEREAL_NVP(bot_name));
    ar(cereal::make_optional_nvp("bot_version", bot_version));
    ar(CEREAL_NVP(query));
    ar(cereal::make_optional_nvp("do_sample", do_sample));
    ar(cereal::make_optional_nvp("num_beams", num_beams));
    ar(cereal::make_optional_nvp("top_k", top_k));
    ar(cereal::make_optional_nvp("max_total_tokens", max_total_tokens));
    ar(cereal::make_optional_nvp("min_new_tokens", min_new_tokens));
    ar(cereal::make_optional_nvp("max_new_tokens", max_new_tokens));
    ar(cereal::make_optional_nvp("max_input_tokens", max_input_tokens));
    ar(cereal::make_optional_nvp("top_p", top_p));
    ar(cereal::make_optional_nvp("temperature", temperature));
    ar(cereal::make_optional_nvp("repetition_penalty", repetition_penalty));
    ar(cereal::make_optional_nvp("frequency_penalty", frequency_penalty));
    ar(cereal::make_optional_nvp("history", history));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
  DMChatForm()
      : do_sample(true), num_beams(1), top_k(40), max_total_tokens(2048),
        min_new_tokens(1), max_new_tokens(1024), max_input_tokens(2048),
        repetition_penalty(0), frequency_penalty(0), temperature(0.1),
        top_p(0.75) {}
};

struct DMChatResponse {
  std::string task_id;
  std::string response;
  int code = 0;              // optional
  std::string message;       // optional
  std::string response_type; // optional
  std::string finish_reason; // optional
  Usage usage;               // optional
  int timecost = 0;          // optional
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(task_id));
    ar(CEREAL_NVP(response));
    ar(cereal::make_optional_nvp("code", code));
    ar(cereal::make_optional_nvp("message", message));
    ar(cereal::make_optional_nvp("response_type", response_type));
    ar(cereal::make_optional_nvp("finish_reason", finish_reason));
    ar(cereal::make_optional_nvp("usage", usage));
    ar(cereal::make_optional_nvp("timecost", timecost));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct VLLMForm {
  bool use_beam_search;
  bool stream;
  int max_tokens;
  int n;
  int top_k;
  int best_of;
  float top_p;
  float temperature;
  float presence_penalty;
  float frequency_penalty;
  std::string task_id;
  std::string prompt;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(use_beam_search));
    ar(CEREAL_NVP(stream));
    ar(CEREAL_NVP(max_tokens));
    ar(CEREAL_NVP(n));
    ar(CEREAL_NVP(top_k));
    ar(CEREAL_NVP(best_of));
    ar(CEREAL_NVP(temperature));
    ar(CEREAL_NVP(presence_penalty));
    ar(CEREAL_NVP(frequency_penalty));
    ar(CEREAL_NVP(task_id));
    ar(CEREAL_NVP(prompt));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct VLLMChunkResponse {
  std::vector<std::vector<std::string>> text;
  template <class Archive> void serialize(Archive &ar) { ar(CEREAL_NVP(text)); }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct VLLMResponse {
  std::string task_id;       // optional
  std::string text;          // optional
  std::string finish_reason; // optional
  int completion_token_len;  // optional
  int prompt_token_len;      // optional
                             // stream
  std::string response;      // optional
  std::string response_type; // optional
  int token_len;             // optional
  template <class Archive> void serialize(Archive &ar) {
    ar(cereal::make_optional_nvp("task_id", task_id));
    ar(cereal::make_optional_nvp("text", text));
    ar(cereal::make_optional_nvp("finish_reason", finish_reason));
    ar(cereal::make_optional_nvp("completion_token_len", completion_token_len));
    ar(cereal::make_optional_nvp("prompt_token_len", prompt_token_len));
    ar(cereal::make_optional_nvp("response", response));
    ar(cereal::make_optional_nvp("response_type", response_type));
    ar(cereal::make_optional_nvp("token_len", token_len));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct CharGLMTokenForm {
  std::string apiKey;
  std::string encrypted;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(apiKey));
    ar(CEREAL_NVP(encrypted));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct CharGLMTokenResponse {
  std::string data;
  template <class Archive> void serialize(Archive &ar) { ar(CEREAL_NVP(data)); }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct CharGLMMeta{
  std::string user_info;
  std::string bot_info;
  std::string bot_name;
  std::string user_name;
  template <class Archive> void serialize(Archive &ar) { 
    ar(CEREAL_NVP(user_info));
    ar(CEREAL_NVP(bot_info));
    ar(CEREAL_NVP(bot_name));
    ar(CEREAL_NVP(user_name));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct CharGLMForm {
  std::string request_id;
  CharGLMMeta meta;
  std::vector<ConversationMessage> prompt;
  bool incremental=true;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(request_id));
    ar(CEREAL_NVP(meta));
    ar(CEREAL_NVP(prompt));
    ar(CEREAL_NVP(incremental));
  }
  MESSAGE_DESERIALIZE;
  std::string serialize() {
    std::stringstream ss;
    try {
      cereal::JSONOutputArchive archive(ss,
                                        cereal::JSONOutputArchive::Options(1));
      serialize(archive);
    } catch (std::exception &e) {
      LOG(ERROR) << e.what();
    }
    return ss.str();
  }
};

struct CharGLMResponseData {
  std::string request_id;
  std::string task_id;
  std::string task_status;
  std::vector<ConversationMessage> choices;
  Usage usage;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(request_id));
    ar(CEREAL_NVP(task_id));
    ar(CEREAL_NVP(task_status));
    ar(cereal::make_optional_nvp("choices", choices));
    ar(CEREAL_NVP(usage));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct CharGLMResponse {
  int code;
  std::string msg;
  bool success;
  CharGLMResponseData data;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(code));
    ar(CEREAL_NVP(msg));
    ar(CEREAL_NVP(success));
    ar(cereal::make_optional_nvp("data", data));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct MinimaxMessage {
  std::string sender_type;
  std::string sender_name;
  std::string text;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(sender_type));
    ar(CEREAL_NVP(sender_name));
    ar(CEREAL_NVP(text));
  }
};

struct MinimaxReplyConstraints {
  std::string sender_type;
  std::string sender_name;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(sender_type));
    ar(CEREAL_NVP(sender_name));
  }
};

struct MinimaxBotSetting {
  std::string bot_name;
  std::string content;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(bot_name));
    ar(CEREAL_NVP(content));
  }
};

struct MinimaxForm {
  std::string model;
  bool stream;
  int tokens_to_generate;
  float top_p;
  float temperature;
  std::vector<MinimaxMessage> messages;
  std::vector<MinimaxBotSetting> bot_setting;
  MinimaxReplyConstraints reply_constraints;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(model));
    ar(CEREAL_NVP(stream));
    ar(CEREAL_NVP(tokens_to_generate));
    ar(CEREAL_NVP(top_p));
    ar(CEREAL_NVP(temperature));
    ar(CEREAL_NVP(messages));
    ar(CEREAL_NVP(bot_setting));
    ar(CEREAL_NVP(reply_constraints));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct MinimaxChoice{
  std::vector<MinimaxMessage> messages;
  std::string finish_reason;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(messages));
    ar(cereal::make_optional_nvp("finish_reason", finish_reason));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct MinimaxResponse {
  unsigned int created;
  std::string reply;
  std::vector<MinimaxChoice> choices;
  Usage usage;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(created));
    ar(CEREAL_NVP(reply));
    ar(cereal::make_optional_nvp("choices", choices));
    ar(cereal::make_optional_nvp("usage", usage));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};


struct CharacterProfile{
  std::string user_name;
  std::string user_info;
  std::string character_name;
  std::string character_info;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(user_name));
    ar(CEREAL_NVP(user_info));
    ar(CEREAL_NVP(character_name));
    ar(CEREAL_NVP(character_info));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct BaichuanForm {
  std::string model;
  std::vector<ConversationMessage> messages;
  bool stream=false;             // optional
  int top_k=40;               // optional
  int max_tokens=512;          // optional
  float top_p=0.75;             // optional
  float temperature=0.1;       // optional
  CharacterProfile character_profile;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(model));
    ar(CEREAL_NVP(messages));
    ar(cereal::make_optional_nvp("stream", stream));
    ar(cereal::make_optional_nvp("top_k", top_k));
    ar(cereal::make_optional_nvp("max_tokens", max_tokens));
    ar(cereal::make_optional_nvp("top_p", top_p));
    ar(cereal::make_optional_nvp("temperature", temperature));
    ar(cereal::make_optional_nvp("character_profile", character_profile));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

using BaichuanResponse = ChatResponse;


struct ModelPermission {
  std::string id;
  std::string object = "model_permission";
  unsigned int created;
  bool allow_create_engine;
  bool allow_sampling;
  bool allow_logprobs;
  bool allow_search_indices;
  bool allow_view;
  bool allow_fine_tuning;
  std::string organization;
  std::string group;
  bool is_blocking;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(id));
    ar(CEREAL_NVP(object));
    ar(CEREAL_NVP(created));
    ar(CEREAL_NVP(allow_create_engine));
    ar(CEREAL_NVP(allow_sampling));
    ar(CEREAL_NVP(allow_logprobs));
    ar(CEREAL_NVP(allow_search_indices));
    ar(CEREAL_NVP(allow_view));
    ar(CEREAL_NVP(allow_fine_tuning));
    ar(CEREAL_NVP(organization));
    ar(CEREAL_NVP(group));
    ar(CEREAL_NVP(is_blocking));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct ModelParam {
  std::string name;
  int token;
  int max_token;
  int min_token;
  float diversity;
  float max_diversity;
  float min_diversity;
  float sample_range;
  float max_sample_range;
  float min_sample_range;
  float repetitive_control;
  float max_repetitive_control;
  float min_repetitive_control;
  float lexical_control;
  float max_lexical_control;
  float min_lexical_control;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(name));
    ar(CEREAL_NVP(token));
    ar(CEREAL_NVP(max_token));
    ar(CEREAL_NVP(min_token));
    ar(CEREAL_NVP(diversity));
    ar(CEREAL_NVP(max_diversity));
    ar(CEREAL_NVP(min_diversity));
    ar(CEREAL_NVP(sample_range));
    ar(CEREAL_NVP(max_sample_range));
    ar(CEREAL_NVP(min_sample_range));
    ar(CEREAL_NVP(repetitive_control));
    ar(CEREAL_NVP(max_repetitive_control));
    ar(CEREAL_NVP(min_repetitive_control));
    ar(CEREAL_NVP(lexical_control));
    ar(CEREAL_NVP(max_lexical_control));
    ar(CEREAL_NVP(min_lexical_control));
  }
};

struct ModelInfo {
  std::string id;
  uint32_t hash;
  std::string object = "model";
  uint32_t created;
  std::string owned_by = "DMAI";
  std::string root;
  std::string parent;
  std::vector<ModelParam> paramList;
  std::vector<ModelPermission> permission;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(id));
    ar(CEREAL_NVP(object));
    ar(CEREAL_NVP(hash));
    ar(CEREAL_NVP(created));
    ar(CEREAL_NVP(owned_by));
    ar(CEREAL_NVP(paramList));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};

struct ModelsResponse {
  int code = 0;
  std::string message; //提示信息
  std::string msg;     //提示信息2
  std::string object = "list";
  std::vector<ModelInfo> data;
  template <class Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(code));
    ar(CEREAL_NVP(message));
    ar(CEREAL_NVP(msg));
    ar(CEREAL_NVP(object));
    ar(CEREAL_NVP(data));
  }
  MESSAGE_DESERIALIZE;
  MESSAGE_SERIALIZE;
};