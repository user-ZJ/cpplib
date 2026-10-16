//
// HTTPFormServer.cpp
//
// This sample demonstrates the HTTPServer and HTMLForm classes.
//
// Copyright (c) 2006, Applied Informatics Software Engineering GmbH.
// and Contributors.
//
// SPDX-License-Identifier:	BSL-1.0
//

#include "Poco/Exception.h"
#include "Poco/JSON/Object.h"
#include "Poco/JSON/Parser.h"
#include "Poco/Net/FilePartSource.h"
#include "Poco/Net/HTMLForm.h"
#include "Poco/Net/HTTPClientSession.h"
#include "Poco/Net/HTTPRequest.h"
#include "Poco/Net/HTTPResponse.h"
#include "Poco/StreamCopier.h"
#include "Poco/URI.h"
#include "utils/file-util.h"
#include "utils/logging.h"
#include <functional>
#include <sstream>
#include <string>
#include <vector>

using Poco::Net::HTTPClientSession;
using Poco::Net::HTTPRequest;
using Poco::Net::HTTPResponse;
using Poco::Net::HTTPMessage;
using Poco::Net::FilePartSource;
using Poco::StreamCopier;
using Poco::URI;
using Poco::Exception;

using namespace BASE_NAMESPACE;

static const std::string kServer = "http://localhost:9900";

// 发送请求并返回响应体；write_body 为空表示没有请求体
static std::string Request(const std::string &url, HTTPRequest &request, HTTPResponse &response,
                           const std::function<void(std::ostream &)> &write_body = nullptr) {
  URI uri(url);
  // https则使用HTTPSClientSession，并链接NetSSL库
  HTTPClientSession session(uri.getHost(), uri.getPort());
  std::string path(uri.getPathAndQuery());
  if (path.empty()) path = "/";
  request.setURI(path);
  request.set("Test-Header", "Test-Header");

  std::ostream &requestStream = session.sendRequest(request);
  if (write_body) write_body(requestStream);
  LOG(INFO) << request.getMethod() << " " << url;

  std::istream &responseStream = session.receiveResponse(response);
  std::string body;
  StreamCopier::copyToString(responseStream, body);
  LOG(INFO) << response.getStatus() << " " << response.getReason() << " " << body;
  return body;
}

static void PrintResult(const HTTPResponse &response, const std::string &body) {
  if (response.getStatus() / 100 != 2) {
    LOG(ERROR) << "request failed: " << response.getStatus() << " " << body;
    return;
  }
  if (body.empty()) return;
  Poco::JSON::Parser parser;
  auto obj = parser.parse(body).extract<Poco::JSON::Object::Ptr>();
  LOG(INFO) << "code:" << obj->optValue<std::string>("code", "") << " message:"
            << obj->optValue<std::string>("message", "");
}

static void PostBody(const std::string &url, const std::string &content_type, const std::string &body) {
  HTTPRequest request(HTTPRequest::HTTP_POST, "/", HTTPMessage::HTTP_1_1);
  request.setContentType(content_type);
  request.setContentLength(body.size());
  HTTPResponse response;
  std::string result = Request(url, request, response, [&](std::ostream &os) { os << body; });
  PrintResult(response, result);
}

static std::string LoginJson() {
  Poco::JSON::Object obj;
  obj.set("username", "user1@yourdomain.com");
  obj.set("password", "mypword");
  std::stringstream sbody;
  obj.stringify(sbody);
  return sbody.str();
}

void get() {
  try {
    URI uri(kServer + "/get");
    uri.addQueryParameter("token", "xxxxx");
    HTTPRequest request(HTTPRequest::HTTP_GET, "/", HTTPMessage::HTTP_1_1);
    HTTPResponse response;
    std::string result = Request(uri.toString(), request, response);
    PrintResult(response, result);
  }
  catch (Exception &ex) {
    LOG(ERROR) << ex.displayText();
  }
}

void form() {
  try {
    const std::string filename = "test.txt";
    std::string teststr = "测试文件文本";
    std::vector<char> buf(teststr.begin(), teststr.end());
    BASE_NAMESPACE::writeBinaryFile(filename.c_str(), buf);

    Poco::Net::HTMLForm form;
    form.setEncoding(Poco::Net::HTMLForm::ENCODING_MULTIPART);  //没有文件时设置为ENCODING_URL
    form.add("grant_type", "password");
    form.add("client_id", "client token");
    form.add("client_secret", "client secret");
    form.set("username", "user@example.com");
    form.set("password", "secret");
    form.addPart("file", new FilePartSource(filename));

    HTTPRequest request(HTTPRequest::HTTP_POST, "/", HTTPMessage::HTTP_1_1);
    form.prepareSubmit(request);
    HTTPResponse response;
    std::string result = Request(kServer + "/form", request, response, [&](std::ostream &os) { form.write(os); });
    PrintResult(response, result);
  }
  catch (Exception &ex) {
    LOG(ERROR) << ex.displayText();
  }
}

void json() {
  try {
    PostBody(kServer + "/json", "application/json", LoginJson());
  }
  catch (Exception &ex) {
    LOG(ERROR) << ex.displayText();
  }
}

void async_json() {
  try {
    PostBody(kServer + "/async_json", "application/json", LoginJson());
  }
  catch (Exception &ex) {
    LOG(ERROR) << ex.displayText();
  }
}

void octet_stream() {
  try {
    PostBody(kServer + "/octet_stream", "application/octet-stream", file_to_str("test.txt"));
  }
  catch (Exception &ex) {
    LOG(ERROR) << ex.displayText();
  }
}

int main() {
  get();
  form();
  json();
  async_json();
  octet_stream();
}
