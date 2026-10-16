
#include "DefaultHandler.h"
#include "HttpUtil.h"

namespace BASE_NAMESPACE {

void DefaultHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response) {
  HandleSafely(request, response, [&] {
    // 请求体未读取，不能复用连接
    response.setKeepAlive(false);
    SendResult(response, "9999", "default response,please check url", HTTPResponse::HTTP_NOT_FOUND);
  });
}

}  // namespace BASE_NAMESPACE
