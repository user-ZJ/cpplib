
#include "HttpGetHandler.h"
#include "HttpUtil.h"
#include "utils/logging.h"

namespace BASE_NAMESPACE {

void HTTPGetHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response) {
  HandleSafely(request, response, [&] {
    Poco::URI uri(request.getURI());
    if (VLOG_IS_ON(1)) {
      for (const auto &kv : uri.getQueryParameters()) {
        VLOG(1) << "key:" << kv.first << " value:" << kv.second;
      }
    }
    SendResult(response, "0000", "success");
  });
}

}  // namespace BASE_NAMESPACE
