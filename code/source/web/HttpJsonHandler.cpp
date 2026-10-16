
#include "HttpJsonHandler.h"
#include "HttpUtil.h"
#include "utils/logging.h"

namespace BASE_NAMESPACE {

static void LogHeaders(const HTTPServerRequest &request) {
  if (!VLOG_IS_ON(1)) return;
  for (const auto &kv : request) {
    VLOG(1) << "\t" << kv.first << ": " << kv.second;
  }
}

void HTTPJsonHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response) {
  HandleSafely(request, response, [&] {
    RequireMethod(request, response, Poco::Net::HTTPRequest::HTTP_POST);
    LogHeaders(request);
    auto pObject = ParseJsonObject(ReadBody(request));
    std::string username = RequireString(pObject, "username");
    RequireString(pObject, "password");
    VLOG(1) << "username:" << username;

    // 设置返回头，必须在发送响应前
    response.set("status", "2000");
    SendResult(response, "0000", "success");
  });
}

void AsyncHTTPJsonHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response) {
  HandleSafely(request, response, [&] {
    RequireMethod(request, response, Poco::Net::HTTPRequest::HTTP_POST);
    LogHeaders(request);
    auto pObject = ParseJsonObject(ReadBody(request));
    std::string username = RequireString(pObject, "username");
    RequireString(pObject, "password");
    VLOG(1) << "username:" << username;

    // 只负责受理并返回 202；耗时任务需投递到独立的工作线程，不能阻塞 HTTP 线程池
    SendResult(response, "0000", "accepted", HTTPResponse::HTTP_ACCEPTED);
  });
}

}  // namespace BASE_NAMESPACE
