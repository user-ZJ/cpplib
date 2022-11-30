
#include "HttpGetHandler.h"
#include "utils/logging.h"
#include "utils/string-util.h"

namespace BASE_NAMESPACE {

void HTTPGetHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response) {
  try {
    LOG(INFO) << "URI: " << request.getURI() << "  Method:" << request.getMethod();
    Poco::URI uri(request.getURI());
    auto query = uri.getQueryParameters();
    for (const auto &kv : query) {
      LOG(INFO) << "key:" << std::get<0>(kv) << " value:" << std::get<1>(kv);
    }
    response.setChunkedTransferEncoding(true);
    response.setContentType("application/json");

    std::ostream &ostr = response.send();
    Poco::JSON::Object obj;
    obj.set("code", "0000");
    obj.set("message", "success");
    obj.stringify(ostr);
  }
  catch (const std::exception &e) {
    LOG(ERROR) << "http exception:" << e.what();
  }
}

}  // namespace BASE_NAMESPACE