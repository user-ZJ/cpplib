
#include "HttpGetHandler.h"
#include "utils/logging.h"
#include "utils/string-util.h"


namespace BASE_NAMESPACE {

void HTTPGetHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response) {
  try {
    LOG(INFO) << "URI: " << request.getURI() << "  Method:" << request.getMethod();
    Poco::URI uri(request.getURI());
    auto query = uri.getQueryParameters();
    for(const auto &kv:query){
      LOG(INFO)<<"key:"<<std::get<0>(kv)<<" value:"<<std::get<1>(kv);
    }
  } catch (const std::exception &e) { LOG(ERROR) << "http exception:" << e.what(); }
}



}