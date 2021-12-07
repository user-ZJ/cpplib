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
  explicit RedisWrapper(std::string host, int port = 6379);

  ~RedisWrapper();

  int set(const std::string &key, const std::string &value,int timeout=0);

  int append(const std::string &key, const std::string &value);

  int get(const std::string &key,std::string *value);

  bool exists(const std::string &key);

  RedisWrapper(const RedisWrapper&) = delete;
  RedisWrapper& operator=(const RedisWrapper&) = delete;

private:
  std::string _host;
  int _port;
  bool _connected;
  Poco::Redis::Client _redis;
};

}; // namespace BASE_NAMESPACE