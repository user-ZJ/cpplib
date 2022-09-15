
#include "HttpJsonHandler.h"
#include "utils/logging.h"
#include "utils/string-util.h"

namespace BASE_NAMESPACE {

void HTTPJsonHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response) {
  try {
    LOG(INFO) << "URI: " << request.getURI() << "  Method:" << request.getMethod();
    std::string jsonstr;
    StreamCopier::copyToString(request.stream(), jsonstr);
    LOG(INFO)<<"json str:"<<jsonstr;
    Parser parser;
    auto result = parser.parse(jsonstr);
    auto pObject = result.extract<Object::Ptr>();
    std::string username = pObject->getValue<std::string>("username");
    std::string password = pObject->getValue<std::string>("password");
    
    LOG(INFO)<<"parsed json:";
    LOG(INFO) << "\tusername:" << username ;
    LOG(INFO) << "\tpassword:" << password ;

    response.setChunkedTransferEncoding(true);
    response.setContentType("application/json");

    std::ostream &ostr = response.send();

    LOG(INFO) << "Request";
    LOG(INFO) << "Method: " << request.getMethod() ;
    LOG(INFO) << "URI: " << request.getURI();
    NameValueCollection::ConstIterator it = request.begin();
    NameValueCollection::ConstIterator end = request.end();
    LOG(INFO)<<"request header:";
    for (; it != end; ++it) {
      LOG(INFO) <<"\t"<< it->first << ": " << it->second;
    }

    Poco::JSON::Object obj;
    obj.set("code", "0000");
    obj.set("message", "success");
    obj.stringify(ostr);
  } catch (const std::exception &e) { LOG(ERROR) << "http exception:" << e.what(); }
}

}  // namespace BASE_NAMESPACE