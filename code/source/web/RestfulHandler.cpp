
#include "RestfulHandler.h"
#include "HttpUtil.h"
#include "utils/logging.h"

namespace BASE_NAMESPACE {

void RestfulHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response) {
  HandleSafely(request, response, [&] {
    const std::string &method = request.getMethod();
    if (method == Poco::Net::HTTPRequest::HTTP_GET) {
      GET(request, response);
    } else if (method == Poco::Net::HTTPRequest::HTTP_POST) {
      POST(request, response);
    } else if (method == Poco::Net::HTTPRequest::HTTP_PUT) {
      PUT(request, response);
    } else if (method == Poco::Net::HTTPRequest::HTTP_PATCH) {
      PATCH(request, response);
    } else if (method == Poco::Net::HTTPRequest::HTTP_DELETE) {
      DELETE(request, response);
    } else if (method == Poco::Net::HTTPRequest::HTTP_HEAD) {
      HEAD(request, response);
    } else if (method == Poco::Net::HTTPRequest::HTTP_OPTIONS) {
      OPTIONS(request, response);
    } else {
      NotSupported(request);
    }
  });
}

void RestfulHandler::NotSupported(const HTTPServerRequest &request) {
  throw HttpError(HTTPResponse::HTTP_METHOD_NOT_ALLOWED, "not support method " + request.getMethod());
}

void RestfulHandler::GET(HTTPServerRequest &request, HTTPServerResponse &response) {
  NotSupported(request);
}

void RestfulHandler::POST(HTTPServerRequest &request, HTTPServerResponse &response) {
  NotSupported(request);
}

void RestfulHandler::PUT(HTTPServerRequest &request, HTTPServerResponse &response) {
  NotSupported(request);
}

void RestfulHandler::PATCH(HTTPServerRequest &request, HTTPServerResponse &response) {
  NotSupported(request);
}

void RestfulHandler::DELETE(HTTPServerRequest &request, HTTPServerResponse &response) {
  NotSupported(request);
}

void RestfulHandler::HEAD(HTTPServerRequest &request, HTTPServerResponse &response) {
  // HEAD 响应不能带 body
  response.setStatusAndReason(HTTPResponse::HTTP_METHOD_NOT_ALLOWED);
  response.setContentLength(0);
  response.send();
}

void RestfulHandler::OPTIONS(HTTPServerRequest &request, HTTPServerResponse &response) {
  response.setStatusAndReason(HTTPResponse::HTTP_NO_CONTENT);
  response.set("Allow", "GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS");
  response.setContentLength(0);
  response.send();
}

}  // namespace BASE_NAMESPACE
