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
#include "Poco/URI.h"
#include "utils/logging.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

using Poco::Net::HTTPClientSession;
using Poco::Net::HTTPRequest;
using Poco::Net::HTTPResponse;
using Poco::Net::HTTPMessage;
using Poco::URI;
using Poco::Exception;
using Poco::Net::WebSocket;
using Poco::Net::WebSocketException;

using namespace BASE_NAMESPACE;

// 回调都在 WebSocketClient 的接收线程中执行
class WSListener {
public:
  virtual ~WSListener() = default;
  virtual void OnText(const std::string &text) { LOG(INFO) << "OnText: " << text; }
  virtual void OnBinary(const Poco::Buffer<char> &buffer) {
    LOG(INFO) << "OnBinary: " << buffer.size() << " bytes";
  }
  virtual void OnClose() { LOG(INFO) << "OnClose"; }
  virtual void OnError(const std::string &message) { LOG(ERROR) << "OnError: " << message; }
};

class WebSocketClient {
public:
  // 连接或握手失败时抛出 Poco::Exception
  WebSocketClient(const std::string &url, WSListener &listener) : listener_(listener) {
    URI uri(url);
    // wss则使用HTTPSClientSession，并链接NetSSL库
    session_ = std::make_shared<HTTPClientSession>(uri.getHost(), uri.getPort());

    std::string path(uri.getPathAndQuery());
    if (path.empty()) path = "/";

    HTTPRequest request(HTTPRequest::HTTP_GET, path, HTTPMessage::HTTP_1_1);
    HTTPResponse response;
    ws_ = std::make_shared<WebSocket>(*session_, request, response);
    ws_->setMaxPayloadSize(kMaxMessageSize);
    // 防止帧传到一半卡住或对端不读数据时线程永久阻塞
    ws_->setReceiveTimeout(kIoTimeout);
    ws_->setSendTimeout(kIoTimeout);
    receiver_ = std::thread(&WebSocketClient::ReceiveLoop, this);
  }

  WebSocketClient(const WebSocketClient &) = delete;
  WebSocketClient &operator=(const WebSocketClient &) = delete;

  ~WebSocketClient() {
    Close();
    Join();
    try {
      ws_->close();
    } catch (...) {
    }
  }

  // 可在任意线程调用；关闭握手开始后返回 -1
  int SendText(const std::string &text) {
    return SendData(text.data(), static_cast<int>(text.size()), WebSocket::FRAME_TEXT);
  }

  int SendBinary(const std::vector<char> &buffer) {
    return SendData(buffer.data(), static_cast<int>(buffer.size()), WebSocket::FRAME_BINARY);
  }

  // 发起关闭握手，接收线程收到对端的 Close 或等待超时后退出
  void Close(Poco::UInt16 code = WebSocket::WS_NORMAL_CLOSE, const std::string &reason = "") {
    if (finished_ || close_sent_.exchange(true)) return;
    try {
      std::lock_guard<std::mutex> lock(send_mutex_);
      ws_->shutdown(code, reason);
    } catch (Poco::Exception &e) {
      LOG(WARNING) << "send close failed: " << e.displayText();
    }
  }

  // 阻塞直到连接关闭
  void Join() {
    if (receiver_.joinable()) receiver_.join();
  }

private:
  static constexpr int kMaxMessageSize = 16 * 1024 * 1024;
  static constexpr std::chrono::seconds kCloseTimeout{5};
  static const Poco::Timespan kIoTimeout;
  static const Poco::Timespan kPollInterval;

  int SendData(const void *data, int length, int flags) {
    if (close_sent_ || finished_) return -1;
    return SendFrame(data, length, flags);
  }

  int SendFrame(const void *data, int length, int flags) {
    std::lock_guard<std::mutex> lock(send_mutex_);
    return ws_->sendFrame(data, length, flags);
  }

  void ProtocolError(Poco::UInt16 code, const std::string &reason) {
    listener_.OnError(reason);
    Close(code, reason);
  }

  void ReceiveLoop() {
    try {
      ReceiveFrames();
    } catch (Poco::Exception &e) {
      listener_.OnError(e.displayText());
    } catch (std::exception &e) {
      listener_.OnError(e.what());
    }
    finished_ = true;
    listener_.OnClose();
  }

  void ReceiveFrames() {
    using Clock = std::chrono::steady_clock;
    Poco::Buffer<char> frame(1024);
    Poco::Buffer<char> message(0);
    int message_opcode = 0;  // 正在重组的分片消息的 opcode，0 表示没有
    Clock::time_point close_deadline = Clock::time_point::max();

    while (true) {
      if (close_sent_ && close_deadline == Clock::time_point::max()) {
        close_deadline = Clock::now() + kCloseTimeout;
      }
      if (Clock::now() >= close_deadline) {
        LOG(WARNING) << "close handshake timeout";
        return;
      }
      // 短周期轮询，以便 Close() 后能及时检查超时
      if (ws_->available() <= 0 && !ws_->poll(kPollInterval, Poco::Net::Socket::SELECT_READ)) continue;

      int flags = 0;
      frame.resize(0);
      int n = ws_->receiveFrame(frame, flags);
      VLOG(1) << Poco::format("receive frame (length=%d, flags=0x%x).", n, unsigned(flags));

      if (n == 0 && flags == 0) {
        LOG(INFO) << "peer closed connection";
        return;
      }

      const int opcode = flags & WebSocket::FRAME_OP_BITMASK;
      const bool fin = (flags & WebSocket::FRAME_FLAG_FIN) != 0;
      switch (opcode) {
      case WebSocket::FRAME_OP_PING:
        SendFrame(frame.begin(), n, WebSocket::FRAME_FLAG_FIN | WebSocket::FRAME_OP_PONG);
        break;
      case WebSocket::FRAME_OP_PONG:
        break;
      case WebSocket::FRAME_OP_CLOSE:
        // 对端主动关闭时原样回送状态码；若是我们先发的 Close，这就是对端的确认
        if (!close_sent_.exchange(true)) {
          SendFrame(frame.begin(), n, WebSocket::FRAME_FLAG_FIN | WebSocket::FRAME_OP_CLOSE);
        }
        return;
      case WebSocket::FRAME_OP_TEXT:
      case WebSocket::FRAME_OP_BINARY:
        if (message_opcode != 0) {
          ProtocolError(WebSocket::WS_PROTOCOL_ERROR, "new message before previous one finished");
          return;
        }
        if (fin) {
          Dispatch(opcode, frame);
        } else {
          message_opcode = opcode;
          message.resize(0);
          message.append(frame.begin(), n);
        }
        break;
      case WebSocket::FRAME_OP_CONT:
        if (message_opcode == 0) {
          ProtocolError(WebSocket::WS_PROTOCOL_ERROR, "unexpected continuation frame");
          return;
        }
        if (message.size() + n > static_cast<std::size_t>(kMaxMessageSize)) {
          ProtocolError(WebSocket::WS_PAYLOAD_TOO_BIG, "message too big");
          return;
        }
        message.append(frame.begin(), n);
        if (fin) {
          Dispatch(message_opcode, message);
          message_opcode = 0;
          message.resize(0);
        }
        break;
      default:
        ProtocolError(WebSocket::WS_PROTOCOL_ERROR,
                      Poco::format("unexpected flags (length=%d, flags=0x%x)", n, unsigned(flags)));
        return;
      }
    }
  }

  void Dispatch(int opcode, const Poco::Buffer<char> &payload) {
    if (opcode == WebSocket::FRAME_OP_TEXT) {
      listener_.OnText(std::string(payload.begin(), payload.size()));
    } else {
      listener_.OnBinary(payload);
    }
  }

  WSListener &listener_;
  std::shared_ptr<HTTPClientSession> session_;
  std::shared_ptr<WebSocket> ws_;
  std::mutex send_mutex_;
  std::atomic<bool> close_sent_{false};
  std::atomic<bool> finished_{false};
  std::thread receiver_;
};

const Poco::Timespan WebSocketClient::kIoTimeout(30, 0);
const Poco::Timespan WebSocketClient::kPollInterval(1, 0);

int main() {
  try {
    std::string url = "http://localhost:9900/ws";
    Poco::JSON::Object obj;
    obj.set("username", "user1@yourdomain.com");
    obj.set("password", "mypword");
    std::stringstream sbody;
    obj.stringify(sbody);
    std::cout << "send body:" << sbody.str() << std::endl;
    std::string text = sbody.str();
    WSListener listener;
    WebSocketClient wsc(url, listener);
    wsc.SendText(text);
  }
  catch (Exception &ex) {
    std::cerr << ex.displayText() << std::endl;
  }
}
