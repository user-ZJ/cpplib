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
#include "Poco/Net/HTMLForm.h"
#include "Poco/Net/HTTPClientSession.h"
#include "Poco/Net/HTTPCredentials.h"
#include "Poco/Net/HTTPRequest.h"
#include "Poco/Net/HTTPResponse.h"
#include "Poco/NullStream.h"
#include "Poco/Path.h"
#include "Poco/StreamCopier.h"
#include "Poco/URI.h"
#include "utils/logging.h"
#include <iostream>
#include <map>
#include <sstream>

using Poco::Net::HTTPClientSession;
using Poco::Net::HTTPRequest;
using Poco::Net::HTTPResponse;
using Poco::Net::HTTPMessage;
using Poco::StreamCopier;
using Poco::Path;
using Poco::URI;
using Poco::Exception;

void get() {
  try {
    std::string url = "http://localhost:9900/get";
    std::string token = "xxxxx";
    URI uri(url);
    uri.addQueryParameter("token", token);
    HTTPClientSession session(uri.getHost(), uri.getPort());

    std::string path(uri.getPathAndQuery());
    if (path.empty()) path = "/";

    HTTPRequest request(HTTPRequest::HTTP_GET, path, HTTPMessage::HTTP_1_1);
    std::map<std::string, std::string> headers;
    headers["Test-Header"] = "Test-Header";
    // Set headers here
    for (std::map<std::string, std::string>::iterator it = headers.begin(); it != headers.end(); it++) {
      request.set(it->first, it->second);
    }

    std::ostream &requestStream = session.sendRequest(request);
    //获取请求文本
    std::stringstream iss;
    request.write(iss);
    LOG(INFO) << iss.str();

    // get response
    HTTPResponse response;
    std::istream &responseStream = session.receiveResponse(response);
    LOG(INFO) << response.getStatus() << " " << response.getReason();
    std::stringstream oss;
    StreamCopier::copyStream(responseStream, oss);
    LOG(INFO) << oss.str();
  } catch (Exception &ex) { LOG(ERROR) << ex.displayText(); }
}

void form() {
  try {
    std::string url = "http://localhost:9900/form";
    URI uri(url);
    HTTPClientSession session(uri.getHost(), uri.getPort());

    std::string path(uri.getPathAndQuery());
    if (path.empty()) path = "/";

    HTTPRequest request(HTTPRequest::HTTP_GET, path, HTTPMessage::HTTP_1_1);

    Poco::Net::HTMLForm form;
    form.add("grant_type", "password");
    form.add("client_id", "client token");
    form.add("client_secret", "client secret");
    form.add("username", "user@example.com");
    form.add("password", "secret");
    form.prepareSubmit(request);

    std::ostream &requestStream = session.sendRequest(request);
    form.write(requestStream); //发送form表单
    //获取请求文本
    std::stringstream iss;
    request.write(iss);
    LOG(INFO) << iss.str();

    // get response
    HTTPResponse response;
    std::istream &responseStream = session.receiveResponse(response);
    LOG(INFO) << response.getStatus() << " " << response.getReason();
    std::stringstream oss;
    StreamCopier::copyStream(responseStream, oss);
    LOG(INFO) << oss.str();
    Poco::JSON::Parser parser;
    Poco::JSON::Object::Ptr authObj = parser.parse(oss).extract<Poco::JSON::Object::Ptr>();
  } catch (Exception &ex) { LOG(ERROR) << ex.displayText(); }
}

void json() {
  try {
    std::string url = "http://localhost:9900/json";
    URI uri(url);
    HTTPClientSession session(uri.getHost(), uri.getPort());

    std::string path(uri.getPathAndQuery());
    if (path.empty()) path = "/";

    HTTPRequest request(HTTPRequest::HTTP_GET, path, HTTPMessage::HTTP_1_1);
    
    std::map<std::string, std::string> headers;
    headers["Test-Header"] = "Test-Header";
    // req.setContentType("application/x-www-form-urlencoded");
    request.setContentType("application/json");
    // Set headers here
    for (std::map<std::string, std::string>::iterator it = headers.begin(); it != headers.end(); it++) {
      request.set(it->first, it->second);
    }
    // set body
    Poco::JSON::Object obj;
    obj.set("username", "user1@yourdomain.com");
    obj.set("password", "mypword");
    std::stringstream sbody;
    obj.stringify(sbody);
    LOG(INFO) << "send body:" << sbody.str();

    // Set the request body
    request.setContentLength(sbody.str().size());
    // sends request, returns open stream
    std::ostream &requestStream = session.sendRequest(request);
    // os << body; // sends the body
    obj.stringify(requestStream);

    //获取请求文本
    std::stringstream iss;
    request.write(iss);
    LOG(INFO) << iss.str();

    // get response
    HTTPResponse response;
    std::istream &responseStream = session.receiveResponse(response);
    LOG(INFO) << response.getStatus() << " " << response.getReason();
    std::stringstream oss;
    StreamCopier::copyStream(responseStream, oss);
    LOG(INFO) << oss.str();
    Poco::JSON::Parser parser;
    Poco::JSON::Object::Ptr authObj = parser.parse(oss).extract<Poco::JSON::Object::Ptr>();
  } catch (Exception &ex) { LOG(ERROR) << ex.displayText(); }
}

int main(int argc, char **argv) {
  get();
  form();
  json();
}
