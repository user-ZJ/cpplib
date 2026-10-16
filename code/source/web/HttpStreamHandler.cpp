
#include "HttpStreamHandler.h"
#include "HttpUtil.h"
#include "utils/logging.h"

namespace BASE_NAMESPACE {

void HttpStreamHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response) {
  HandleSafely(request, response, [&] {
    RequireMethod(request, response, Poco::Net::HTTPRequest::HTTP_POST);
    std::string binary_str = ReadBody(request);
    VLOG(1) << "octet stream received " << binary_str.size() << " bytes";

    Poco::JSON::Object obj;
    obj.set("code", "0000");
    obj.set("message", "success");
    obj.set("size", binary_str.size());
    SendJson(response, obj);
  });
}

}  // namespace BASE_NAMESPACE
