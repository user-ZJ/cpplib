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
#include "Poco/Net/HTTPClientSession.h"
#include "Poco/Net/HTTPRequest.h"
#include "Poco/Net/HTTPResponse.h"
#include "Poco/Net/WebSocket.h"
#include "Poco/NullStream.h"
#include "Poco/Path.h"
#include "Poco/StreamCopier.h"
#include "Poco/URI.h"
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
using Poco::Net::WebSocket;

int main(int argc, char **argv) {
  try {
    std::string url = "http://localhost:9980/ws";
    // std::string body = "username=user1@yourdomain.com&password=mypword";
    std::string body = "{\"password\":\"mypword\",\"username\":\"user1@yourdomain.com\"}";
    Poco::JSON::Object obj;
    obj.set("username", "user1@yourdomain.com");
    obj.set("password", "mypword");
    std::stringstream sbody;
    obj.stringify(sbody);
    std::cout << "send body:" << sbody.str() << std::endl;
    std::string text = sbody.str();

    URI uri(url);
    HTTPClientSession session(uri.getHost(), uri.getPort());

    std::string path(uri.getPathAndQuery());
    if (path.empty()) path = "/";

    HTTPRequest req(HTTPRequest::HTTP_GET, path, HTTPMessage::HTTP_1_1);
    HTTPResponse res;
    char buffer[1024];
    int flags;
    int n;
    WebSocket *ws = new WebSocket(session, req, res);
    n = ws->sendFrame(text.c_str(), text.length(), WebSocket::FRAME_TEXT);
    std::cout<<Poco::format("Frame send (length=%d).\n", n);
    n = ws->receiveFrame(buffer,sizeof(buffer),flags);
    std::cout<<Poco::format("Frame received (length=%d, flags=0x%x).\n", n, unsigned(flags));
    ws->close();
    delete ws;
  } catch (Exception &ex) { std::cerr << ex.displayText() << std::endl; }
}
