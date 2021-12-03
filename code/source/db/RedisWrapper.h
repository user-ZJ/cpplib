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