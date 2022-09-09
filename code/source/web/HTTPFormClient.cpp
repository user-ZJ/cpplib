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

#include "Poco/Net/HTTPClientSession.h"
#include "Poco/Net/HTTPRequest.h"
#include "Poco/Net/HTTPResponse.h"
#include "Poco/Net/HTTPCredentials.h"
#include "Poco/StreamCopier.h"
#include "Poco/NullStream.h"
#include "Poco/Path.h"
#include "Poco/URI.h"
#include "Poco/Exception.h"
#include "Poco/JSON/Object.h"
#include <iostream>
#include <sstream>
#include <map>

using Poco::Net::HTTPClientSession;
using Poco::Net::HTTPRequest;
using Poco::Net::HTTPResponse;
using Poco::Net::HTTPMessage;
using Poco::StreamCopier;
using Poco::Path;
using Poco::URI;
using Poco::Exception;

int main(int argc, char **argv) {
  try {
    std::string url = "http://localhost:9980/form";
    std::string body = "username=user1@yourdomain.com&password=mypword";
    // std::string body = R"({"password":"mypword","username":"user1@yourdomain.com"})";
    
    std::map<std::string, std::string> headers;
    headers["Test-Header"] = "Test-Header";
    URI uri(url);
    HTTPClientSession session(uri.getHost(), uri.getPort());

    std::string path(uri.getPathAndQuery());
    if (path.empty())
      path = "/";

    HTTPRequest req(HTTPRequest::HTTP_POST, path, HTTPMessage::HTTP_1_1);
    req.setContentType("application/x-www-form-urlencoded");

    // Set headers here
    for (std::map<std::string, std::string>::iterator it = headers.begin();
         it != headers.end(); it++) {
      req.set(it->first, it->second);
    }

    // Set the request body
    req.setContentLength(body.size());
    // sends request, returns open stream
    std::ostream &os = session.sendRequest(req);
    os << body; // sends the body
    req.write(std::cout); // print out request

    // get response
    HTTPResponse res;
    std::cout << res.getStatus() << " " << res.getReason() << std::endl;

    std::istream &is = session.receiveResponse(res);
    std::stringstream ss;
    StreamCopier::copyStream(is, ss);
    std::cout << ss.str() << std::endl;
  } catch (Exception &ex) {
    std::cerr << ex.displayText() << std::endl;
  }
}
