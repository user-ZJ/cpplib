#include "MYSQLWrapper.h"
#include "VPFeatHandler.h"
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
  pool = std::make_unique<SessionPool>("mysql", dbConnString, 2, 20, 60);
  dbInfo(pool->get());
}

MYSQLWrapper::~MYSQLWrapper() {
  MySQL::Connector::unregisterConnector();
}

void MYSQLWrapper::dbInfo(Session &&session) {
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
  LOG(INFO) << dbConnString;
  try {
    // Session session("mysql", dbConnString);
    dbInfo(pool->get());
  }
  catch (ConnectionFailedException &ex) {
    LOG(ERROR) << ex.displayText();
  }
  return 0;
}

std::set<std::string> MYSQLWrapper::getTables() {
  std::set<std::string> tables;
  std::string table_name = "test";
  try {
    pool->get() << "SHOW TABLES", into(tables), now;
  }
  catch (ConnectionException &ce) {
    LOG(ERROR) << ce.displayText();
  }
  catch (StatementException &se) {
    LOG(ERROR) << se.displayText();
  }
  return tables;
}

void MYSQLWrapper::dropTable(const std::string &tableName) {
  try {
    pool->get() << format("DROP TABLE IF EXISTS %s", tableName), now;
  }
  catch (ConnectionException &ce) {
    LOG(ERROR) << ce.displayText();
  }
  catch (StatementException &se) {
    LOG(ERROR) << se.displayText();
  }
}

void MYSQLWrapper::createVPTable(const std::string &tableName) {
  dropTable(tableName);
  std::string create_cmd = "CREATE TABLE " + tableName
                           + " (id BIGINT PRIMARY KEY AUTO_INCREMENT,speak_id VARCHAR(32) NOT "
                             "NULL,dmid "
                             "CHAR(32),feature BLOB,create_time TIMESTAMP DEFAULT "
                             "CURRENT_TIMESTAMP,update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON "
                             "UPDATE CURRENT_TIMESTAMP,soft_del TINYINT,INDEX idx_speak_id "
                             "(speak_id));";
  try {
    pool->get() << create_cmd, now;
  }
  catch (ConnectionException &ce) {
    LOG(ERROR) << ce.displayText();
  }
  catch (StatementException &se) {
    LOG(ERROR) << se.displayText();
  }
}

int MYSQLWrapper::insertFeat(const std::string &tableName, const VPFeat &vpfeat) {
  LOG(INFO) << "insert feat to mysql";
  VPFeat feat = queryFeat(tableName, vpfeat.speaker_id);
  if (feat.id != 0) {
    LOG(ERROR) << vpfeat.speaker_id << "was in table:" << tableName;
    return -1;
  }
  std::string cmd = "INSERT INTO " + tableName + " VALUES (?,?,?,?,?,?,?)";
  try {
    VPFeat ft = vpfeat;
    pool->get() << cmd, use(ft), now;
  }
  catch (ConnectionException &ce) {
    LOG(ERROR) << ce.displayText();
    return -1;
  }
  catch (StatementException &se) {
    LOG(ERROR) << se.displayText();
    return -1;
  }
  LOG(INFO) << "insert feat to mysql success";
  return 0;
}

int MYSQLWrapper::updateFeat(const std::string &tableName, const VPFeat &vpfeat) {
  LOG(INFO) << "update feat to mysql";
  VPFeat feat = queryFeat(tableName, vpfeat.speaker_id);
  if (feat.id != 0) {
    auto feature = vpfeat.feature;
    std::string cmd = "UPDATE " + tableName + " SET feature=? WHERE speak_id = " + vpfeat.speaker_id + "'";
    try {
      pool->get() << cmd, use(feature), now;
    }
    catch (ConnectionException &ce) {
      LOG(ERROR) << ce.displayText();
      return -1;
    }
    catch (StatementException &se) {
      LOG(ERROR) << se.displayText();
      return -1;
    }
  }
  LOG(INFO) << "update feat to mysql success";
  return 0;
}

VPFeat MYSQLWrapper::queryFeat(const std::string &tableName, const std::string &spk_id) {
  LOG(INFO) << "query feat from mysql";
  VPFeat feat;
  std::string cmd = "SELECT * FROM " + tableName + " WHERE speak_id = '" + spk_id + "'";
  try {
    pool->get() << cmd, into(feat), now;
  }
  catch (ConnectionException &ce) {
    LOG(ERROR) << ce.displayText();
    return feat;
  }
  catch (StatementException &se) {
    // LOG(WARNING) << se.displayText();
    LOG(WARNING) << "no spk_id:" << spk_id << " in table:" << tableName;
    return feat;
  }
  LOG(INFO) << "query feat from mysql success";
  return feat;
}

void MYSQLWrapper::delFeat(const std::string &tableName, const std::string &spk_id) {
  LOG(INFO) << "delete feat from mysql";
  VPFeat feat = queryFeat(tableName, spk_id);
  if (feat.id != 0) {
    feat.soft_del = 1;
    std::string cmd = "UPDATE " + tableName + " SET soft_del=1 WHERE speak_id = '" + spk_id + "'";
    try {
      pool->get() << cmd, now;
    }
    catch (ConnectionException &ce) {
      LOG(ERROR) << ce.displayText();
    }
    catch (StatementException &se) {
      LOG(ERROR) << se.displayText();
    }
  }
  LOG(INFO) << "delete feat from mysql success";
}

}  // namespace BASE_NAMESPACE