
#include "RestfulHandler.h"
#include "Poco/Net/WebSocket.h"
#include "utils/logging.h"
#include "utils/timer.h"
#include "utils/uuid-wrapper.h"

using Poco::Net::WebSocket;
using Poco::Net::WebSocketException;

namespace DMAI {

void RestfulHandler::handleRequest(HTTPServerRequest &request,
                                   HTTPServerResponse &response) {
  try {
    LOG(INFO) << "RestfulHandler::handleRequest start";

    if (request.getMethod() == Poco::Net::HTTPRequest::HTTP_GET) {
      GET(request, response);
    } else if (request.getMethod() == Poco::Net::HTTPRequest::HTTP_POST) {
      POST(request, response);
    } else if (request.getMethod() == Poco::Net::HTTPRequest::HTTP_POST) {
      POST(request, response);
    } else if (request.getMethod() == Poco::Net::HTTPRequest::HTTP_PUT) {
      PUT(request, response);
    } else if (request.getMethod() == Poco::Net::HTTPRequest::HTTP_PATCH) {
      PATCH(request, response);
    } else if (request.getMethod() == Poco::Net::HTTPRequest::HTTP_DELETE) {
      DELETE(request, response);
    } else if (request.getMethod() == Poco::Net::HTTPRequest::HTTP_HEAD) {
      HEAD(request, response);
    } else if (request.getMethod() == Poco::Net::HTTPRequest::HTTP_OPTIONS) {
      OPTIONS(request, response);
    }

  } catch (std::exception &e) {
    LOG(ERROR) << e.what();
  }

  LOG(INFO) << "RestfulHandler::handleRequest end";
}

void RestfulHandler::GET(HTTPServerRequest &request,
                         HTTPServerResponse &response) {
  response.setChunkedTransferEncoding(true);
  response.setContentType("application/json");
  std::ostream &ostr = response.send();
  ostr << R"({"code":1,"msg":"not support method GET"})";
  LOG(ERROR)<<R"({"code":1,"msg":"not support method GET"})";
  return;
}
void RestfulHandler::POST(HTTPServerRequest &request,
                          HTTPServerResponse &response) {
  response.setChunkedTransferEncoding(true);
  response.setContentType("application/json");
  std::ostream &ostr = response.send();
  ostr << R"({"code":1,"msg":"not support method POST"})";
  LOG(ERROR)<<R"({"code":1,"msg":"not support method POST"})";
  return;
}
void RestfulHandler::PUT(HTTPServerRequest &request,
                         HTTPServerResponse &response) {
  response.setChunkedTransferEncoding(true);
  response.setContentType("application/json");
  std::ostream &ostr = response.send();
  ostr << R"({"code":1,"msg":"not support method PUT"})";
  LOG(ERROR)<<R"({"code":1,"msg":"not support method PUT"})";
  return;
}
void RestfulHandler::PATCH(HTTPServerRequest &request,
                           HTTPServerResponse &response) {
  response.setChunkedTransferEncoding(true);
  response.setContentType("application/json");
  std::ostream &ostr = response.send();
  ostr << R"({"code":1,"msg":"not support method PATCH"})";
  LOG(ERROR)<<R"({"code":1,"msg":"not support method PATCH"})";
  return;
}
void RestfulHandler::DELETE(HTTPServerRequest &request,
                            HTTPServerResponse &response) {
  response.setChunkedTransferEncoding(true);
  response.setContentType("application/json");
  std::ostream &ostr = response.send();
  ostr << R"({"code":1,"msg":"not support method DELETE"})";
  LOG(ERROR)<<R"({"code":1,"msg":"not support method DELETE"})";
  return;
}
void RestfulHandler::HEAD(HTTPServerRequest &request,
                          HTTPServerResponse &response) {
  response.setContentType("application/json");
  std::ostream &ostr = response.send();
  ostr << R"({"code":1,"msg":"not support method HEAD"})";
  LOG(ERROR)<<R"({"code":1,"msg":"not support method HEAD"})";
  return;
}
void RestfulHandler::OPTIONS(HTTPServerRequest &request,
                             HTTPServerResponse &response) {
  response.setContentType("application/json");
  std::ostream &ostr = response.send();
  ostr << R"({"code":1,"msg":"not support method OPTIONS"})";
  LOG(ERROR)<<R"({"code":1,"msg":"not support method OPTIONS"})";
  return;
}

} // namespace DMAI