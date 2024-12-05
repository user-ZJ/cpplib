/*
 * @Author: zack 
 * @Date: 2021-12-06 09:56:19 
 * @Last Modified by: zack
 * @Last Modified time: 2021-12-07 17:54:44
 */
#include "RedisWrapper.h"
#include "utils/logging.h"

namespace BASE_NAMESPACE {

RedisWrapper::RedisWrapper(const std::string &url)
    : _url(url), _connected(false) {
  if (!_connected) {
    try {
      // Poco::Timespan t(10, 0); // Connect within 10 seconds
      _redis.connect(_url /*,t*/);
      _connected = true;
      VLOG(2) << "Connected to [" << _url << ']';
    } catch (Poco::Exception &e) {
      LOG(ERROR) << "Couldn't connect to [" << _url << ']' << e.message()
                 << ". ";
    }
  }
}

RedisWrapper::RedisWrapper(std::string host, int port) : _connected(false) {
  if (!_connected) {
    try {
      _url = host + ":" + std::to_string(port);
      // Poco::Timespan t(10, 0); // Connect within 10 seconds
      _redis.connect(_url /*,t*/);
      _connected = true;
      VLOG(2) << "Connected to [" << _url << ']';
    } catch (Poco::Exception &e) {
      LOG(ERROR) << "Couldn't connect to [" << _url << ']' << e.message()
                 << ". ";
    }
  }
}

RedisWrapper::~RedisWrapper() {
  if (_connected) {
    _redis.disconnect();
    _connected = false;
    VLOG(2) << "Disconnected from [" << _url << ']';
  }
}

void RedisWrapper::set(const std::string &key, const std::string &value,
                       int timeout) {
  Poco::Timespan t(timeout); // delete after 10 seconds
  Poco::Redis::Command command = Poco::Redis::Command::set(key, value);
  ;
  if (timeout > 0)
    command = Poco::Redis::Command::set(key, value, true, t, true);
  try {
    std::string result = _redis.execute<std::string>(command);
    if (result.compare("OK") != 0) {
      LOG(ERROR) << "set " << key << " failed";
    }
  } catch (Poco::Redis::RedisException &e) {
    LOG(ERROR) << e.message();
  } catch (Poco::BadCastException &e) {
    LOG(ERROR) << e.message();
  }
}

int RedisWrapper::append(const std::string &key, const std::string &value) {
  Poco::Redis::Command appendCommand = Poco::Redis::Command::append(key, value);
  try {
    Poco::Int64 result = _redis.execute<Poco::Int64>(appendCommand);
    return result;
  } catch (Poco::Redis::RedisException &e) {
    LOG(ERROR) << e.message();
    return -1;
  } catch (Poco::BadCastException &e) {
    LOG(ERROR) << e.message();
    return -1;
  }
}

std::string RedisWrapper::get(const std::string &key) {
  std::string str = "";
  Poco::Redis::Command getCommand = Poco::Redis::Command::get(key);
  try {
    Poco::Redis::BulkString result =
        _redis.execute<Poco::Redis::BulkString>(getCommand);
    str = result.value();
  } catch (Poco::Redis::RedisException &e) {
    LOG(ERROR) << e.message();
  } catch (Poco::BadCastException &e) {
    LOG(ERROR) << e.message();
  } catch (Poco::Exception &e) {
    LOG(ERROR) << e.what();
  }
  return str;
}

bool RedisWrapper::exists(const std::string &key) {
  Poco::Redis::Command existsCommand = Poco::Redis::Command::exists(key);
  try {
    Poco::Int64 result = _redis.execute<Poco::Int64>(existsCommand);
    return result > 0;
  } catch (Poco::Redis::RedisException &e) {
    LOG(ERROR) << e.message();
    return false;
  } catch (Poco::Exception &e) {
    LOG(ERROR) << e.what();
    return false;
  }
}

int RedisWrapper::lpush(const std::string &key, const std::string &value) {
  Poco::Redis::Command command = Poco::Redis::Command::lpush(key, value);
  try {
    Poco::Int64 result = _redis.execute<Poco::Int64>(command);
    return result;
  } catch (Poco::Redis::RedisException &e) {
    LOG(ERROR) << e.message();
  } catch (Poco::Exception &e) {
    LOG(ERROR) << e.what();
  }
  return -1;
}

std::string RedisWrapper::rpop(const std::string &key) {
  std::string str;
  Poco::Redis::Command command = Poco::Redis::Command::rpop(key);
  try {
    Poco::Redis::BulkString result =
        _redis.execute<Poco::Redis::BulkString>(command);
    str = result.value();
  } catch (Poco::Redis::RedisException &e) {
    LOG(ERROR) << e.message();
  } catch (Poco::Exception &e) {
    LOG(ERROR) << e.what();
  }
  return str;
}

int RedisWrapper::llen(const std::string &key) {
  Poco::Redis::Command command = Poco::Redis::Command::llen(key);
  try {
    Poco::Int64 n = _redis.execute<Poco::Int64>(command);
    return n;
  } catch (Poco::Redis::RedisException &e) {
    LOG(ERROR) << e.message();
  } catch (Poco::Exception &e) {
    LOG(ERROR) << e.what();
  }
  return -1;
}

int RedisWrapper::sadd(const std::string &key, const std::string &value) {
  Poco::Redis::Command command = Poco::Redis::Command::sadd(key, value);
  try {
    Poco::Int64 result = _redis.execute<Poco::Int64>(command);
    return result;
  } catch (Poco::Redis::RedisException &e) {
    LOG(ERROR) << e.message();
  } catch (Poco::Exception &e) {
    LOG(ERROR) << e.what();
  }
  return -1;
}
int RedisWrapper::sadd(const std::string &key,
                       const std::vector<std::string> &values) {
  Poco::Redis::Command command = Poco::Redis::Command::sadd(key, values);
  try {
    Poco::Int64 result = _redis.execute<Poco::Int64>(command);
    return result;
  } catch (Poco::Redis::RedisException &e) {
    LOG(ERROR) << e.message();
  } catch (Poco::Exception &e) {
    LOG(ERROR) << e.what();
  }
  return -1;
}
int RedisWrapper::srem(const std::string &key, const std::string &value) {
  Poco::Redis::Command command = Poco::Redis::Command::srem(key, value);
  try {
    Poco::Int64 result = _redis.execute<Poco::Int64>(command);
    return result;
  } catch (Poco::Redis::RedisException &e) {
    LOG(ERROR) << e.message();
  } catch (Poco::Exception &e) {
    LOG(ERROR) << e.what();
  }
  return -1;
}
int RedisWrapper::srem(const std::string &key,
                       const std::vector<std::string> &values) {
  Poco::Redis::Command command = Poco::Redis::Command::srem(key, values);
  try {
    Poco::Int64 result = _redis.execute<Poco::Int64>(command);
    return result;
  } catch (Poco::Redis::RedisException &e) {
    LOG(ERROR) << e.message();
  } catch (Poco::Exception &e) {
    LOG(ERROR) << e.what();
  }
  return -1;
}

}; // namespace DMAI