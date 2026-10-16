
#include "WebsocketHandler.h"
#include "utils/logging.h"
#include "utils/string-util.h"

namespace BASE_NAMESPACE {

// 单帧与分片重组后的整条消息都不能超过该大小，防止恶意帧头导致内存耗尽
static constexpr int kMaxMessageSize = 16 * 1024 * 1024;
// 连接空闲超过该时间发送 ping，再过一个周期仍无任何数据则断开
static const Poco::Timespan kPingInterval(30, 0);

void WebSocketHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response) {
  try {
    ws = std::make_shared<WebSocket>(request, response);
  }
  catch (WebSocketException &exc) {
    LOG(ERROR) << "ERRORCODE:" << exc.code() << " MESSAGE:" << exc.message();
    switch (exc.code()) {
    case WebSocket::WS_ERR_HANDSHAKE_UNSUPPORTED_VERSION:
      response.set("Sec-WebSocket-Version", WebSocket::WEBSOCKET_VERSION);
      // fallthrough
    case WebSocket::WS_ERR_NO_HANDSHAKE:
    case WebSocket::WS_ERR_HANDSHAKE_NO_VERSION:
    case WebSocket::WS_ERR_HANDSHAKE_NO_KEY:
      response.setStatusAndReason(HTTPResponse::HTTP_BAD_REQUEST);
      response.setContentLength(0);
      response.send();
      break;
    }
    return;
  }

  const std::string peer = request.clientAddress().toString();
  LOG(INFO) << "WebSocket connection established: " << peer;
  try {
    Serve();
  }
  catch (Poco::Exception &e) {
    LOG(ERROR) << "ws exception(" << peer << "): " << e.displayText();
  }
  catch (std::exception &e) {
    LOG(ERROR) << "ws exception(" << peer << "): " << e.what();
  }
  OnClose();
  LOG(INFO) << "WebSocket connection closed: " << peer;
}

void WebSocketHandler::Serve() {
  ws->setMaxPayloadSize(kMaxMessageSize);
  // 握手后 socket 仍沿用 HTTPServerParams 的超时，这里换成心跳周期，
  // 保证帧传到一半卡住或对端不读数据时不会无限阻塞
  ws->setReceiveTimeout(kPingInterval);
  ws->setSendTimeout(kPingInterval);

  Poco::Buffer<char> frame(1024);
  Poco::Buffer<char> message(0);
  int message_opcode = 0;  // 正在重组的分片消息的 opcode，0 表示没有
  bool awaiting_pong = false;

  while (true) {
    if (ws->available() <= 0 && !ws->poll(kPingInterval, Poco::Net::Socket::SELECT_READ)) {
      if (awaiting_pong) {
        LOG(WARNING) << "ping timeout, closing";
        ws->shutdown(WebSocket::WS_ENDPOINT_GOING_AWAY, "ping timeout");
        return;
      }
      SendFrame(nullptr, 0, WebSocket::FRAME_FLAG_FIN | WebSocket::FRAME_OP_PING);
      awaiting_pong = true;
      continue;
    }

    int flags = 0;
    frame.resize(0);
    int n = ws->receiveFrame(frame, flags);
    awaiting_pong = false;
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
      // 原样回送对端的状态码，完成关闭握手
      SendFrame(frame.begin(), n, WebSocket::FRAME_FLAG_FIN | WebSocket::FRAME_OP_CLOSE);
      return;
    case WebSocket::FRAME_OP_TEXT:
    case WebSocket::FRAME_OP_BINARY:
      if (message_opcode != 0) {
        ws->shutdown(WebSocket::WS_PROTOCOL_ERROR, "new message before previous one finished");
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
        ws->shutdown(WebSocket::WS_PROTOCOL_ERROR, "unexpected continuation frame");
        return;
      }
      if (message.size() + n > static_cast<std::size_t>(kMaxMessageSize)) {
        ws->shutdown(WebSocket::WS_PAYLOAD_TOO_BIG, "message too big");
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
      LOG(WARNING) << Poco::format("unexpected flags (length=%d, flags=0x%x).", n, unsigned(flags));
      ws->shutdown(WebSocket::WS_PROTOCOL_ERROR, "unknown opcode");
      return;
    }
  }
}

void WebSocketHandler::Dispatch(int opcode, const Poco::Buffer<char> &payload) {
  if (opcode == WebSocket::FRAME_OP_TEXT) {
    OnText(std::string(payload.begin(), payload.size()));
  } else {
    OnBinary(payload);
  }
}

int WebSocketHandler::SendFrame(const void *data, int length, int flags) {
  int n = ws->sendFrame(data, length, flags);
  if (n < 0) {
    LOG(WARNING) << Poco::format("send frame failed (length=%d, flags=0x%x, ret=%d).", length, unsigned(flags), n);
  }
  return n;
}

void WebSocketHandler::OnText(const std::string &text) {
  VLOG(1) << "OnText: " << text.size() << " bytes";
  SendFrame(text.data(), static_cast<int>(text.size()), WebSocket::FRAME_TEXT);
}

void WebSocketHandler::OnBinary(const Poco::Buffer<char> &buffer) {
  VLOG(1) << "OnBinary: " << buffer.size() << " bytes";
  SendFrame(buffer.begin(), static_cast<int>(buffer.size()), WebSocket::FRAME_BINARY);
}

void WebSocketHandler::OnClose() {
  VLOG(1) << "OnClose";
}

}  // namespace BASE_NAMESPACE
