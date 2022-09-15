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
#include "Poco/Net/NetException.h"
#include "Poco/Net/WebSocket.h"
#include "Poco/NullStream.h"
#include "Poco/Path.h"
#include "Poco/StreamCopier.h"
#include "Poco/URI.h"
#include "utils/logging.h"
#include <iostream>
#include <map>
#include <sstream>
#include <thread>

using Poco::Net::HTTPClientSession;
using Poco::Net::HTTPRequest;
using Poco::Net::HTTPResponse;
using Poco::Net::HTTPMessage;
using Poco::StreamCopier;
using Poco::Path;
using Poco::URI;
using Poco::Exception;
using Poco::Net::WebSocket;
using Poco::Net::WebSocketException;

using namespace BASE_NAMESPACE;

static bool IsText(const int &flags) {
  return flags == WebSocket::FRAME_TEXT;
}

static bool IsBinary(const int &flags) {
  return flags == WebSocket::FRAME_BINARY;
}

static bool IsClose(const int &flags) {
  return (flags & WebSocket::FRAME_OP_BITMASK) == WebSocket::FRAME_OP_CLOSE;
}

class WebSocketClient {
 public:
  explicit WebSocketClient(std::string url) {
    try {
      URI uri(url);
      session = std::make_shared<HTTPClientSession>(uri.getHost(), uri.getPort());

      std::string path(uri.getPathAndQuery());
      if (path.empty()) path = "/";

      request = std::make_shared<HTTPRequest>(HTTPRequest::HTTP_GET, path, HTTPMessage::HTTP_1_1);
      response = std::make_shared<HTTPResponse>();
      ws.reset(new WebSocket(*session, *request, *response));
      connect = std::thread(&WebSocketClient::GetResponse, this);
    }
    catch (WebSocketException &exc) {
      LOG(ERROR) << "OnError: ERRORCODE:" << exc.code() << " MESSAGE:" << exc.message();
    }
    catch (std::exception &e) {
      LOG(ERROR) << "ws exception:" << e.what();
    }
  }

  ~WebSocketClient() {
    ws->shutdown();
    connect.join();
    ws->close();
  }

  void GetResponse() {
    try {
      Poco::Buffer<char> buffer(1024);
      int flags;
      int n;
      while (1) {
        buffer.resize(0);
        n = ws->receiveFrame(buffer, flags);
        LOG(INFO) << Poco::format("recive data (length=%d, flags=0x%x).", n, unsigned(flags));
        if (n == 0 || IsClose(flags)) {
          OnClose();
          break;
        } else if (IsText(flags)) {
          std::string text(buffer.begin(), buffer.size());
          OnText(text);
        } else if (IsBinary(flags)) {
          OnBinary(buffer);
        } else {
          LOG(WARNING) << Poco::format("Unespect flags (length=%d, flags=0x%x).", n, unsigned(flags));
        }
      }
    }
    catch (WebSocketException &exc) {
      LOG(ERROR) << "OnError: ERRORCODE:" << exc.code() << " MESSAGE:" << exc.message();
    }
    catch (std::exception &e) {
      LOG(ERROR) << "ws exception:" << e.what();
    }
  }

  void OnClose() {
    LOG(INFO) << "OnClose";
  }
  void OnText(const std::string &text) {
    LOG(INFO) << "OnText" << text;
  }
  void OnBinary(const Poco::Buffer<char> &buffer) {
    LOG(INFO) << "OnBinary: receiver" << buffer.sizeBytes() << "bytes";
  }

  int SendText(const std::string &text) {
    LOG(INFO) << "Send text";
    return ws->sendFrame(text.c_str(), text.length(), WebSocket::FRAME_TEXT);
  }

  int SendBinary(const std::vector<char> &buffer) {
    LOG(INFO) << "Send binary";
    int n = ws->sendFrame(buffer.data(), buffer.size(), WebSocket::FRAME_BINARY);
    return n;
  }

 private:
  std::shared_ptr<HTTPRequest> request;
  std::shared_ptr<HTTPResponse> response;
  std::shared_ptr<HTTPClientSession> session;
  std::shared_ptr<WebSocket> ws;
  std::thread connect;
};

int main(int argc, char **argv) {
  try {
    std::string url = "http://localhost:9900/ws";
    // std::string body = "username=user1@yourdomain.com&password=mypword";
    std::string body = R"({"password":"mypword","username":"user1@yourdomain.com"})";
    Poco::JSON::Object obj;
    obj.set("username", "user1@yourdomain.com");
    obj.set("password", "mypword");
    std::stringstream sbody;
    obj.stringify(sbody);
    std::cout << "send body:" << sbody.str() << std::endl;
    std::string text = sbody.str();
    WebSocketClient wsc(url);
    wsc.SendText(text);
  }
  catch (Exception &ex) {
    std::cerr << ex.displayText() << std::endl;
  }
}
