#include <iostream>
#include <string>
#include <cassert>
#include <vector>
#include <limits>
#include "utils/file-util.h"
#include "utils/logging.h"
#include "utils/flags.h"
#include "utils/string-util.h"
#include "db/MYSQLWrapper.h"


using namespace Poco::Data;
using namespace Poco::Data::Keywords;
using Poco::Data::MySQL::ConnectionException;
using Poco::Data::MySQL::Utility;
using Poco::Data::MySQL::StatementException;
using Poco::format;
using Poco::NotFoundException;
using Poco::Int32;
using Poco::Nullable;
using Poco::Tuple;
using Poco::NamedTuple;



int testconnect(const std::string &host, int port,
                          const std::string &database, const std::string &user,
                          const std::string &password) {

  std::string dbConnString =
      "host=" + host + ";port=" + std::to_string(port) + ";user=" + user +
      ";password=" + password + ";db=" + database +
      ";compress=true;auto-reconnect=true;secure-auth=true;protocol=tcp";
//   dbConnString = "host=10.12.51.68;port=3306;user=root;password=root;compress=true;auto-reconnect=true;protocol=tcp";
  LOG(INFO) << "dbConnString:" << dbConnString;
  try {
    std::string key = "mysql";
    auto  session=new Session(key, dbConnString);
  } catch (ConnectionFailedException &ex) {
    LOG(ERROR) << ex.displayText();
  } catch (NotFoundException &ex) {
    LOG(ERROR) << ex.displayText();
  }
  return 0;
}


using namespace BASE_NAMESPACE;


int main(int argc, char *argv[]) {
  FLAGS_log_dir = ".";
  google::EnableLogCleaner(30);  // keep your logs for 30 days
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);  //初始化 glog
  google::SetStderrLogging(google::GLOG_ERROR);
  
  MYSQLWrapper::instance().connect("10.12.51.68",3306,"voice_print","root","root");
  
  return 0;
}
