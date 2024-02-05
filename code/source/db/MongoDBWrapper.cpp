/*
 * @Author: zack 
 * @Date: 2021-12-07 15:13:14 
 * @Last Modified by: zack
 * @Last Modified time: 2021-12-08 10:16:17
 */
/*
 * @Author: zack 
 * @Date: 2021-12-06 09:56:04 
 * @Last Modified by: zack
 * @Last Modified time: 2021-12-07 15:13:14
 */
#include "MongoDBWrapper.h"
#include "utils/logging.h"

namespace BASE_NAMESPACE {

MongoDBWrapper::MongoDBWrapper(const std::string &uri)
    : _uri(uri), _connected(false) {
  if (!_connected) {
    try {
      //   std::string uri = "mongodb://admin:admin@10.12.50.209:27017/admin";
      std::cout<<"uri:"<<uri<<std::endl;
      _connection.connect(uri, _sf);
      _connected = true;
      VLOG(2) << "Connected to [" << _uri << ']';
    } catch (Poco::NoPermissionException &e) {
      LOG(ERROR) << "Have no access to " << e.message() << ". ";
    } catch (Poco::Net::ConnectionRefusedException &e) {
      LOG(ERROR) << "Couldn't connect to " << e.message() << ". ";
    } catch (Poco::Exception &e) {
      LOG(ERROR) << e.what();
    }
  }
}

MongoDBWrapper::~MongoDBWrapper() { _connection.disconnect(); }

void MongoDBWrapper::InsertRequest() {
  try {
    Poco::MongoDB::Database db("team");
    Poco::SharedPtr<Poco::MongoDB::InsertRequest> insertPlayerRequest =
        db.createInsertRequest("players");

    Poco::DateTime birthdate;
    birthdate.assign(1969, 3, 9);
    Poco::LocalDateTime now;
    insertPlayerRequest->addNewDocument()
        .add("lastname", "Valdes")
        .add("firstname", "Victor")
        .add("birthdate", birthdate.timestamp())
        .add("start", 1993)
        .add("active", false)
        .add("lastupdated", now.timestamp())
        .add("unknown", Poco::MongoDB::NullValue());

    _connection.sendRequest(*insertPlayerRequest);
    std::string lastError = db.getLastError(_connection);
    if (lastError.empty()) {
      LOG(INFO) << "insert success";
    } else {
      LOG(ERROR) << "insert error:" << lastError;
    }
  } catch (Poco::Exception &e) {
    LOG(ERROR) << e.what();
  }
}


void MongoDBWrapper::UpdateRequest(){
  Poco::MongoDB::Database db("team");
	Poco::SharedPtr<Poco::MongoDB::UpdateRequest> request = db.createUpdateRequest("players");
	request->selector().add("firstname", "Victor"); //WHERE firstname = 'Victor'

	request->update().addNewDocument("$inc").add("start", 1);  //set start = start+1

	_connection.sendRequest(*request);

	Poco::MongoDB::Document::Ptr lastError = db.getLastErrorDoc(_connection);
	LOG(INFO) << "LastError: " << lastError->toString(2);
}

void MongoDBWrapper::QueryRequest() {
  Poco::MongoDB::QueryRequest request("team.players");
  request.selector().add("lastname", std::string("Valdes"));
  request.setNumberToReturn(1);

  Poco::MongoDB::ResponseMessage response;

  _connection.sendRequest(request, response);

  if (response.documents().size() > 0) {
    Poco::MongoDB::Document::Ptr doc = response.documents()[0];

    try {
      std::string lastname = doc->get<std::string>("lastname");
      CHECK(lastname.compare("Valdes") == 0);
      std::string firstname = doc->get<std::string>("firstname");
      CHECK(firstname.compare("Victor") == 0);
      Poco::Timestamp birthDateTimestamp =
          doc->get<Poco::Timestamp>("birthdate");
      Poco::DateTime birthDate(birthDateTimestamp);
      CHECK(birthDate.year() == 1969 && birthDate.month() == 3 &&
            birthDate.day() == 9);
      Poco::Timestamp lastupdatedTimestamp =
          doc->get<Poco::Timestamp>("lastupdated");
      CHECK(doc->isType<Poco::MongoDB::NullValue>("unknown"));
      bool active = doc->get<bool>("active");
      CHECK(!active);

      std::string id = doc->get("_id")->toString();
      LOG(INFO) << lastname << " " << firstname << " " << active << " " << id;
    } catch (Poco::NotFoundException &nfe) {
      LOG(ERROR) << nfe.message() << " not found.";
    } catch (Poco::Exception &e) {
      LOG(ERROR) << e.what();
    }
  } else {
    LOG(ERROR) << "No document returned";
  }
}

void MongoDBWrapper::DeleteRequest() {
  try {
    Poco::MongoDB::Database db("team");
    Poco::SharedPtr<Poco::MongoDB::DeleteRequest> request =
        db.createDeleteRequest("players");
    request->selector().add("firstname", "Victor");

    _connection.sendRequest(*request);

    std::string lastError = db.getLastError(_connection);
    if (lastError.empty()) {
      LOG(INFO) << "delete success";
    } else {
      LOG(ERROR) << "delete error:" << lastError;
    }
  } catch (Poco::Exception &e) {
    LOG(ERROR) << e.what();
  }
}

}; // namespace BASE_NAMESPACE