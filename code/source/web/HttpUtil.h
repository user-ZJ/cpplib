#pragma once
#include "Poco/Exception.h"
#include "Poco/JSON/Object.h"
#include "Poco/JSON/Parser.h"
#include "Poco/Net/HTTPRequest.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "Poco/Net/HTTPServerResponse.h"
#include "utils/logging.h"
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>

namespace BASE_NAMESPACE {

// 请求体（含 multipart 中单个文件）大小上限
constexpr std::size_t kMaxBodySize = 16 * 1024 * 1024;

// 在 handler 中抛出，由 HandleSafely 转成对应状态码的 JSON 响应
class HttpError : public std::runtime_error {
public:
  HttpError(Poco::Net::HTTPResponse::HTTPStatus status, const std::string &message)
      : std::runtime_error(message), status_(status) {}
  Poco::Net::HTTPResponse::HTTPStatus status() const { return status_; }

private:
  Poco::Net::HTTPResponse::HTTPStatus status_;
};

inline void SendJson(Poco::Net::HTTPServerResponse &response, const Poco::JSON::Object &obj,
                     Poco::Net::HTTPResponse::HTTPStatus status = Poco::Net::HTTPResponse::HTTP_OK) {
  std::ostringstream oss;
  obj.stringify(oss);
  const std::string body = oss.str();
  response.setStatusAndReason(status);
  response.setContentType("application/json");
  response.sendBuffer(body.data(), body.size());
}

inline void SendResult(Poco::Net::HTTPServerResponse &response, const std::string &code, const std::string &message,
                       Poco::Net::HTTPResponse::HTTPStatus status = Poco::Net::HTTPResponse::HTTP_OK) {
  Poco::JSON::Object obj;
  obj.set("code", code);
  obj.set("message", message);
  SendJson(response, obj, status);
}

inline void RequireMethod(const Poco::Net::HTTPServerRequest &request, Poco::Net::HTTPServerResponse &response,
                          const std::string &method) {
  if (request.getMethod() != method) {
    response.set("Allow", method);
    throw HttpError(Poco::Net::HTTPResponse::HTTP_METHOD_NOT_ALLOWED, "method not allowed");
  }
}

// 先按 Content-Length 拒绝，再在读取过程中限制大小（应对 chunked 请求）
inline void CheckContentLength(const Poco::Net::HTTPServerRequest &request, std::size_t limit = kMaxBodySize) {
  if (request.hasContentLength() && request.getContentLength64() > static_cast<Poco::Int64>(limit)) {
    throw HttpError(Poco::Net::HTTPResponse::HTTP_REQUEST_ENTITY_TOO_LARGE, "request body too large");
  }
}

inline void ReadLimited(std::istream &in, std::string &out, std::size_t limit = kMaxBodySize) {
  out.clear();
  char buf[8192];
  while (in.read(buf, sizeof(buf)) || in.gcount() > 0) {
    out.append(buf, static_cast<std::size_t>(in.gcount()));
    if (out.size() > limit) {
      throw HttpError(Poco::Net::HTTPResponse::HTTP_REQUEST_ENTITY_TOO_LARGE, "request body too large");
    }
  }
}

inline std::string ReadBody(Poco::Net::HTTPServerRequest &request, std::size_t limit = kMaxBodySize) {
  CheckContentLength(request, limit);
  std::string body;
  if (request.hasContentLength()) body.reserve(static_cast<std::size_t>(request.getContentLength64()));
  ReadLimited(request.stream(), body, limit);
  return body;
}

inline Poco::JSON::Object::Ptr ParseJsonObject(const std::string &body) {
  Poco::Dynamic::Var result;
  try {
    Poco::JSON::Parser parser;
    result = parser.parse(body);
  }
  catch (const Poco::Exception &e) {
    throw HttpError(Poco::Net::HTTPResponse::HTTP_BAD_REQUEST, "invalid json: " + e.message());
  }
  if (result.type() != typeid(Poco::JSON::Object::Ptr)) {
    throw HttpError(Poco::Net::HTTPResponse::HTTP_BAD_REQUEST, "json body must be an object");
  }
  return result.extract<Poco::JSON::Object::Ptr>();
}

inline std::string RequireString(const Poco::JSON::Object::Ptr &obj, const std::string &key) {
  if (!obj->has(key) || obj->isNull(key)) {
    throw HttpError(Poco::Net::HTTPResponse::HTTP_BAD_REQUEST, "missing field: " + key);
  }
  return obj->getValue<std::string>(key);
}

// handler 抛出的异常若未被处理且响应尚未发送，Poco 不会回任何内容，客户端只能等到连接超时
template <typename F>
void HandleSafely(Poco::Net::HTTPServerRequest &request, Poco::Net::HTTPServerResponse &response, F &&handle) {
  auto fail = [&](Poco::Net::HTTPResponse::HTTPStatus status, const std::string &message) {
    if (response.sent()) return;
    // 请求体可能没读完，不能复用连接
    response.setKeepAlive(false);
    try {
      SendResult(response, std::to_string(static_cast<int>(status)), message, status);
    }
    catch (const std::exception &e) {
      LOG(ERROR) << "send error response failed: " << e.what();
    }
  };
  try {
    handle();
  }
  catch (const HttpError &e) {
    LOG(WARNING) << request.getMethod() << " " << request.getURI() << " -> " << static_cast<int>(e.status()) << " "
                 << e.what();
    fail(e.status(), e.what());
  }
  catch (const Poco::Exception &e) {
    LOG(ERROR) << request.getMethod() << " " << request.getURI() << " exception: " << e.displayText();
    fail(Poco::Net::HTTPResponse::HTTP_INTERNAL_SERVER_ERROR, "internal error");
  }
  catch (const std::exception &e) {
    LOG(ERROR) << request.getMethod() << " " << request.getURI() << " exception: " << e.what();
    fail(Poco::Net::HTTPResponse::HTTP_INTERNAL_SERVER_ERROR, "internal error");
  }
}

}  // namespace BASE_NAMESPACE
