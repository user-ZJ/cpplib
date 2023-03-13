// sudo apt install libmysqlclient-dev
#include "Poco/Data/DataException.h"
#include "Poco/Data/LOB.h"
#include "Poco/Data/MySQL/Connector.h"
#include "Poco/Data/MySQL/MySQLException.h"
#include "Poco/Data/MySQL/Utility.h"
#include "Poco/Data/SessionPool.h"
#include "Poco/Data/SessionPoolContainer.h"
#include "Poco/Data/StatementImpl.h"
#include "Poco/Exception.h"
#include "Poco/Format.h"
#include "Poco/NamedTuple.h"
#include "Poco/Nullable.h"
#include "Poco/String.h"
#include "Poco/Tuple.h"
#include "VPFeatHandler.h"
#include <iostream>

using namespace Poco::Data;
using namespace Poco::Data::Keywords;
using Poco::format;
using Poco::Int32;
using Poco::NamedTuple;
using Poco::NotFoundException;
using Poco::Nullable;
using Poco::Tuple;
using Poco::Data::MySQL::ConnectionException;
using Poco::Data::MySQL::StatementException;
using Poco::Data::MySQL::Utility;
using Poco::Data::SessionPool;
using Poco::Data::SessionPoolContainer;
using Poco::Data::SessionPoolExhaustedException;
using Poco::Data::SessionPoolExistsException;

namespace BASE_NAMESPACE {

class MYSQLWrapper {
 public:
  static MYSQLWrapper &instance();
  void dbInfo(Session &&session);
  int connect(const std::string &host, int port, const std::string &database, const std::string &user,
              const std::string &password);
  std::set<std::string> getTables();
  void dropTable(const std::string &tableName);
  void createVPTable(const std::string &tableName);
  int insertFeat(const std::string &tableName, const VPFeat &vpfeat);
  int updateFeat(const std::string &tableName, const VPFeat &vpfeat);
  VPFeat queryFeat(const std::string &tableName, const std::string &spk_id);
  void delFeat(const std::string &tableName, const std::string &spk_id);

 private:
  MYSQLWrapper();
  ~MYSQLWrapper();
  std::unique_ptr<SessionPool> pool;
};

}  // namespace BASE_NAMESPACE