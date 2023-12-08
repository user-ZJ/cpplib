/*
 * @Author: zack 
 * @Date: 2021-12-21 16:32:36 
 * @Last Modified by: zack
 * @Last Modified time: 2022-09-21 14:39:23
 */
#ifndef BASE_UUID_UTIL_H_
#define BASE_UUID_UTIL_H_
#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
#include <string>
#include "utils/hex-util.h"

namespace BASE_NAMESPACE
{

inline std::string gen_uuid() {
  std::string uuid_result;
  // 如果同一个时间使用 uuid_generate_time ,输出的基本是一致
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  uuid_result = boost::uuids::to_string(uuid);
  // std::remove( uuid_result.begin(), uuid_result.end(), '-');
  return uuid_result;
}

inline std::string gen_uuid_hex() {
  std::string uuid_result;
  // 如果同一个时间使用 uuid_generate_time ,输出的基本是一致
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  std::vector<unsigned char> buff(sizeof(uuid));
  memcpy(buff.data(),&uuid,buff.size());
  uuid_result = HexBinaryEncoder(buff);
  return uuid_result;
}


};

#endif