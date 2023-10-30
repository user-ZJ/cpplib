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

class WSListener {
public:
  void OnClose() {
    LOG(INFO) << "OnClose";
    if (tgt_)
      tgt_->shutdown();
  }
  void OnText(const std::string &text) {
    LOG(INFO) << "OnText" << text;
    if (tgt_)
      tgt_->sendFrame(text.c_str(), text.length(), WebSocket::FRAME_TEXT);
  }
  void OnBinary(const Poco::Buffer<char> &buffer) {
    LOG(INFO) << "OnBinary: receiver" << buffer.sizeBytes() << "bytes";
    if (tgt_)
      tgt_->sendFrame(buffer.begin(), buffer.sizeBytes(),
                      WebSocket::FRAME_BINARY);
  }

  void OnError() { LOG(INFO) << "OnError"; }
  void OnPing() {
    LOG(INFO) << "OnPing";
    if (tgt_) {
      tgt_->sendFrame(nullptr, 0, WebSocket::FRAME_OP_PING);
    } else {
      src_->sendFrame(nullptr, 0, Poco::Net::WebSocket::FRAME_OP_PONG);
    }
  }
  void OnPong() {
    LOG(INFO) << "OnPong";
    if (tgt_)
      tgt_->sendFrame(nullptr, 0, WebSocket::FRAME_OP_PONG);
  }

  void operator()(std::shared_ptr<WebSocket> ws,
                  std::shared_ptr<WebSocket> other = nullptr) {
    tgt_ = other;
    src_ = ws;
    auto IsBinary = [](const int &flags) -> bool {
      return flags == Poco::Net::WebSocket::FRAME_BINARY;
    };

    auto IsClose = [](const int &flags) -> bool {
      return (flags & Poco::Net::WebSocket::FRAME_OP_BITMASK) ==
             Poco::Net::WebSocket::FRAME_OP_CLOSE;
    };

    auto IsPing = [](const int &flags) -> bool {
      return ((flags & Poco::Net::WebSocket::FRAME_OP_BITMASK) ==
              Poco::Net::WebSocket::FRAME_OP_PING);
    };

    auto IsPong = [](const int &flags) -> bool {
      return ((flags & Poco::Net::WebSocket::FRAME_OP_BITMASK) ==
              Poco::Net::WebSocket::FRAME_OP_PONG);
    };
    auto IsText = [](const int &flags) -> bool {
      return flags == Poco::Net::WebSocket::FRAME_TEXT;
    };
    try {
      Poco::Buffer<char> buffer(1024);
      int flags;
      int n;
      while (1) {
        buffer.resize(0);
        n = ws->receiveFrame(buffer, flags);
        LOG(INFO) << Poco::format("recive data (length=%d, flags=0x%x).", n,
                                  unsigned(flags));
        if (IsPing(flags)) {
          OnPing();
        } else if (IsPong(flags)) {
          OnPong();
        }
        if (n == 0 || IsClose(flags)) {
          OnClose();
          break;
        } else if (IsText(flags)) {
          std::string text(buffer.begin(), buffer.size());
          OnText(text);
        } else if (IsBinary(flags)) {
          OnBinary(buffer);
        } else {
          LOG(WARNING) << Poco::format(
              "Unespect flags (length=%d, flags=0x%x).", n, unsigned(flags));
        }
      }
    } catch (WebSocketException &exc) {
      OnError();
      LOG(ERROR) << "OnError: ERRORCODE:" << exc.code()
                 << " MESSAGE:" << exc.message();
    } catch (std::exception &e) {
      OnError();
      LOG(ERROR) << "ws exception:" << e.what();
    }
  }

private:
  std::shared_ptr<WebSocket> tgt_;
  std::shared_ptr<WebSocket> src_;
};


class WebSocketClient {
 public:
  explicit WebSocketClient(std::string url, WSListener &listener,std::shared_ptr<Poco::Net::WebSocket> tgt = nullptr) {
    try {
      URI uri(url);
      session = std::make_shared<HTTPClientSession>(uri.getHost(), uri.getPort());

      std::string path(uri.getPathAndQuery());
      if (path.empty()) path = "/";

      request = std::make_shared<HTTPRequest>(HTTPRequest::HTTP_GET, path, HTTPMessage::HTTP_1_1);
      response = std::make_shared<HTTPResponse>();
      ws.reset(new WebSocket(*session, *request, *response));
      message_handler = std::thread(std::ref(listener), ws,tgt);
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
    if (message_handler.joinable()) message_handler.join();
    ws->close();
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

  void Join() {
    if (message_handler.joinable()) message_handler.join();
  }

 private:
  std::shared_ptr<HTTPRequest> request;
  std::shared_ptr<HTTPResponse> response;
  std::shared_ptr<HTTPClientSession> session;
  std::shared_ptr<WebSocket> ws;
  std::thread message_handler;
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
    WSListener listener;
    WebSocketClient wsc(url,listener);
    wsc.SendText(text);
  }
  catch (Exception &ex) {
    std::cerr << ex.displayText() << std::endl;
  }
}
