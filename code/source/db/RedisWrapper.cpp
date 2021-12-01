#include "RedisWrapper.h"

namespace BASE_NAMESPACE {

RedisWrapper::RedisWrapper(std::string host, int port)
    : _host(host), _port(port), _connected(false) {
  if (!_connected) {
    try {
      // Poco::Timespan t(10, 0); // Connect within 10 seconds
      _redis.connect(_host, _port/*,t*/);
      _connected = true;
      std::cout << "Connected to [" << _host << ':' << _port << ']'
                << std::endl;
    } catch (Poco::Exception &e) {
      std::cout << "Couldn't connect to [" << _host << ':' << _port << ']'
                << e.message() << ". " << std::endl;
    }
  }
}

RedisWrapper::~RedisWrapper() {
  if (_connected) {
    _redis.disconnect();
    _connected = false;
    std::cout << "Disconnected from [" << _host << ':' << _port << ']'
              << std::endl;
  }
}

int RedisWrapper::set(const std::string &key, const std::string &value,int timeout) {
  // Poco::Timespan t(10); // delete after 10 seconds
  // Poco::Redis::Command setCommand = Poco::Redis::Command::set(key, value,true,t,true);
  Poco::Redis::Command setCommand("SET");
  setCommand<<key<<value;
  if(timeout>0)
    setCommand<<"EX"<<std::to_string(timeout);
  try {
    std::string result = _redis.execute<std::string>(setCommand);
    if (result.compare("OK") == 0) {
      return 0;
    }
    return -1;
  } catch (Poco::Redis::RedisException &e) {
    std::cout << e.message() << std::endl;
    return -1;
  } catch (Poco::BadCastException &e) {
    std::cout << e.message() << std::endl;
    return -1;
  }
}

int RedisWrapper::get(const std::string &key,std::string *value) {
  Poco::Redis::Command getCommand = Poco::Redis::Command::get(key);
  try {
    Poco::Redis::BulkString result = _redis.execute<Poco::Redis::BulkString>(getCommand);
    *value = result.value();
    return 0;
  } catch (Poco::Redis::RedisException &e) {
    std::cout << e.message() << std::endl;
    *value = e.message();
    return -1;
  } catch (Poco::BadCastException &e) {
    std::cout << e.message() << std::endl;
    *value = e.message();
    return -1;
  }catch (Poco::Exception &e) {
    std::cout << e.what() << std::endl;
    *value = e.what();
    return -1;
  }
}

bool RedisWrapper::exists(const std::string &key){
  Poco::Redis::Command existsCommand = Poco::Redis::Command::exists(key);
  try {
    Poco::Int64 result = _redis.execute<Poco::Int64>(existsCommand);
    return result>0;
  } catch (Poco::Redis::RedisException &e) {
    std::cout << e.message() << std::endl;
    return false;
  }catch (Poco::Exception &e) {
    std::cout << e.what() << std::endl;
    return false;
  }
}

}; // namespace BASE_NAMESPACE