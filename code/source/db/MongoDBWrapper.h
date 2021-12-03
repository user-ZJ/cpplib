#pragma once
#include <string>
#include <memory>
#include "Poco/DateTime.h"
#include "Poco/ObjectPool.h"
#include "Poco/MongoDB/InsertRequest.h"
#include "Poco/MongoDB/QueryRequest.h"
#include "Poco/MongoDB/DeleteRequest.h"
#include "Poco/MongoDB/GetMoreRequest.h"
#include "Poco/MongoDB/PoolableConnectionFactory.h"
#include "Poco/MongoDB/Database.h"
#include "Poco/MongoDB/Cursor.h"
#include "Poco/MongoDB/ObjectId.h"
#include "Poco/MongoDB/Binary.h"
#include "Poco/Net/NetException.h"
#include "Poco/UUIDGenerator.h"

namespace BASE_NAMESPACE {

class MongoDBWrapper {
public:
  MongoDBWrapper(const std::string &uri="mongodb://admin:admin@10.12.50.209:27017/admin");
  ~MongoDBWrapper();
  void InsertRequest();
  void QueryRequest();
  void DeleteRequest();

private:
  std::string _uri;
  bool _connected;
  Poco::MongoDB::Connection _connection;
  Poco::MongoDB::Connection::SocketFactory _sf;
};

}; // namespace BASE_NAMESPACE