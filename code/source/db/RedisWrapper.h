/*
 * @Author: zack 
 * @Date: 2021-12-06 09:56:14 
 * @Last Modified by: zack
 * @Last Modified time: 2021-12-07 16:17:04
 */
#pragma once

#include "Poco/Exception.h"
#include "Poco/Redis/Client.h"
#include "Poco/Redis/Command.h"
#include "Poco/Redis/Redis.h"
#include <string>

namespace BASE_NAMESPACE {

class RedisWrapper {
public:
  explicit RedisWrapper(const std::string &url);
  explicit RedisWrapper(std::string host, int port);

  ~RedisWrapper();
  void set(const std::string &key, const std::string &value,int timeout=0);
  int append(const std::string &key, const std::string &value);
  std::string get(const std::string &key);
  bool exists(const std::string &key);
  // list²Ù×÷
  int lpush(const std::string &key, const std::string &value);
  std::string rpop(const std::string &key);
  int llen(const std::string &key);
  // set²Ù×÷
  int sadd(const std::string &key, const std::string &value);
  int sadd(const std::string &key, const std::vector<std::string> &values);
  int srem(const std::string &key, const std::string &value);
  int srem(const std::string &key, const std::vector<std::string> &values);

  RedisWrapper(const RedisWrapper&) = delete;
  RedisWrapper& operator=(const RedisWrapper&) = delete;
  bool IsConnected(){ return _connected; }

private:
  std::string _url;
  bool _connected;
  Poco::Redis::Client _redis;
};

inline bool testRedisConnection(const std::string &url){
  RedisWrapper re(url);
  return re.IsConnected();
}

}; // namespace BASE_NAMESPACE