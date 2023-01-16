#include "MYSQLWrapper.h"
#include "utils/logging.h"

namespace BASE_NAMESPACE {

MYSQLWrapper &MYSQLWrapper::instance() {
  static MYSQLWrapper ins;
  return ins;
}
MYSQLWrapper::MYSQLWrapper() {
  MySQL::Connector::registerConnector();
  std::string host = "10.12.51.68";
  std::string port = "3306";
  std::string user = "root";
  std::string password = "root";
  std::string database = "voice_print";
  std::string dbConnString = "host=" + host + ";port=" + port + ";user=" + user + ";password=" + password
                             + ";db=" + database + ";compress=true;auto-reconnect=true;secure-auth=true;protocol=tcp";
  pool = std::make_unique<SessionPool>("mysql", dbConnString, 2/*最少连接数*/, 20/*最大连接数*/, 60/*超时时间(s)*/);
}

MYSQLWrapper::~MYSQLWrapper() {
  MySQL::Connector::unregisterConnector();
}

void MYSQLWrapper::dbInfo(Session session) {
  LOG(INFO) << "Server Info: " << Utility::serverInfo(session);
  LOG(INFO) << "Server Version: " << Utility::serverVersion(session);
  LOG(INFO) << "Host Info: " << Utility::hostInfo(session);
}

int MYSQLWrapper::connect(const std::string &host, int port, const std::string &database, const std::string &user,
                          const std::string &password) {
  LOG(INFO) << "connect mysql." << host << ":" << port << " db:" << database;
  std::string dbConnString = "host=" + host + ";port=" + std::to_string(port) + ";user=" + user
                             + ";password=" + password + ";db=" + database
                             + ";compress=true;auto-reconnect=true;secure-auth=true;protocol=tcp";
  try {
    // Session session("mysql", dbConnString);
    dbInfo(pool->get());
  }
  catch (ConnectionFailedException &ex) {
    LOG(ERROR) << ex.displayText();
  }
  return 0;
}

}  // namespace BASE_NAMESPACE
