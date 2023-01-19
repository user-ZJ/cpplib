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
  pool = std::make_unique<SessionPool>("mysql", dbConnString, 2 /*最少连接数*/, 20 /*最大连接数*/, 60 /*超时时间(s)*/);
}

MYSQLWrapper::~MYSQLWrapper() {
  MySQL::Connector::unregisterConnector();
}

void MYSQLWrapper::dbInfo(Session session) {
  LOG(INFO) << "Server Info: " << Utility::serverInfo(session);
  LOG(INFO) << "Server Version: " << Utility::serverVersion(session);
  LOG(INFO) << "Host Info: " << Utility::hostInfo(session);
}

int MYSQLWrapper::connect() {
  LOG(INFO) << "connect mysql.";
  try {
    // Session session("mysql", dbConnString);
    dbInfo(pool->get());
  }
  catch (ConnectionFailedException &ex) {
    LOG(ERROR) << ex.displayText();
  }
  return 0;
}

void MYSQLWrapper::test() {
  try {
    auto session = pool->get();
    // drop sample table, if it exists
    session << "DROP TABLE IF EXISTS Person", now;

    // (re)create table
    session << "CREATE TABLE voice_print.Person (Name VARCHAR(30),Address VARCHAR(45),Age INT(3));", now;

      // insert some rows
      Person person = {"Bart Simpson", "Springfield", 12};

    Statement insert(session);
    insert << "INSERT INTO voice_print.Person VALUES(?, ?, ?)", use(person.name), use(person.address), use(person.age);

    insert.execute();

    person.name = "Lisa Simpson";
    person.address = "Springfield";
    person.age = 10;

    insert.execute();

    // a simple query
    Statement select(session);
    select << "SELECT Name, Address, Age FROM Person", into(person.name), into(person.address), into(person.age),
      range(0, 1);  //  iterate over result set one row at a time

    while (!select.done()) {
      select.execute();
      std::cout << person.name << " " << person.address << " " << person.age << std::endl;
    }
  }
  catch (Poco::Exception &ex) {
    LOG(ERROR) << ex.displayText();
  }
}

}  // namespace BASE_NAMESPACE
