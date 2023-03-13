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






using namespace BASE_NAMESPACE;


int main(int argc, char *argv[]) {
  FLAGS_log_dir = ".";
  google::EnableLogCleaner(30);  // keep your logs for 30 days
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);  //初始化 glog
  google::SetStderrLogging(google::GLOG_ERROR);
  
  MYSQLWrapper::instance();
  
  return 0;
}
