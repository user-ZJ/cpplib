/*
 * @Author: zack 
 * @Date: 2021-12-06 09:56:19 
 * @Last Modified by: zack
 * @Last Modified time: 2021-12-07 17:54:44
 */
#include "RedisWrapper.h"
#include "utils/logging.h"

namespace BASE_NAMESPACE {

RedisWrapper::RedisWrapper(std::string host, int port)
    : _host(host), _port(port), _connected(false) {
  if (!_connected) {
    try {
      // Poco::Timespan t(10, 0); // Connect within 10 seconds
      _redis.connect(_host, _port/*,t*/);
      _connected = true;
      VLOG(2) << "Connected to [" << _host << ':' << _port << ']';
    } catch (Poco::Exception &e) {
      LOG(ERROR) << "Couldn't connect to [" << _host << ':' << _port << ']'
                << e.message() << ". ";
    }
  }
}

RedisWrapper::~RedisWrapper() {
  if (_connected) {
    _redis.disconnect();
    _connected = false;
    VLOG(2) << "Disconnected from [" << _host << ':' << _port << ']';
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
      VLOG(2)<<"set "<<key<<" as "<<value<<" success";
      return 0;
    }
    return -1;
  } catch (Poco::Redis::RedisException &e) {
    LOG(ERROR) << e.message();
    return -1;
  } catch (Poco::BadCastException &e) {
    LOG(ERROR) << e.message();
    return -1;
  }
}

int RedisWrapper::append(const std::string &key, const std::string &value){
  Poco::Redis::Command appendCommand = Poco::Redis::Command::append(key, value);
	try
	{
		Poco::Int64 result = _redis.execute<Poco::Int64>(appendCommand);
    return result;
	}
	catch (Poco::Redis::RedisException& e)
	{
		LOG(ERROR) << e.message();
    return -1;
	}
	catch (Poco::BadCastException& e)
	{
		LOG(ERROR) << e.message();
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
    LOG(ERROR) << e.message();
    *value = e.message();
    return -1;
  } catch (Poco::BadCastException &e) {
    LOG(ERROR) << e.message();
    *value = e.message();
    return -1;
  }catch (Poco::Exception &e) {
    LOG(ERROR) << e.what() ;
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
    LOG(ERROR) << e.message() ;
    return false;
  }catch (Poco::Exception &e) {
    LOG(ERROR) << e.what();
    return false;
  }
}

}; // namespace BASE_NAMESPACE