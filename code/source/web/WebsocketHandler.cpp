
#include "WebsocketHandler.h"
#include "utils/logging.h"
#include "utils/string-util.h"

namespace BASE_NAMESPACE {

static bool IsText(const int &flags) {
  return flags == WebSocket::FRAME_TEXT;
}

static bool IsBinary(const int &flags) {
  return flags == WebSocket::FRAME_BINARY;
}

static bool IsClose(const int &flags) {
  return (flags & WebSocket::FRAME_OP_BITMASK) == WebSocket::FRAME_OP_CLOSE;
}

void WebSocketHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response) {
  try {
    ws = std::make_shared<WebSocket>(request, response);
    LOG(INFO) << "WebSocket connection established.";
    int flags;
    Poco::Buffer<char> buffer(1024);
    int n;
    while (1) {
      buffer.resize(0);
      n = ws->receiveFrame(buffer, flags);
      LOG(INFO) << Poco::format("recive data (length=%d, flags=0x%x).", n, unsigned(flags));
      // flag 0x00为close  flag 0x88为shutdown
      if (n == 0 || IsClose(flags)) {  
        LOG(INFO) << "on_close";
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
    LOG(INFO) << "WebSocket connection closed.";
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
  }
  catch (std::exception &e) {
    LOG(ERROR) << "ws exception:" << e.what();
  }
}

void WebSocketHandler::OnText(const std::string &text) {
  LOG(INFO) << "OnText" << text;
  int n = ws->sendFrame(text.c_str(), text.length(), WebSocket::FRAME_TEXT);
}

void WebSocketHandler::OnBinary(const Poco::Buffer<char> &buffer) {
  LOG(INFO) << "OnBinary";
  int n = ws->sendFrame(buffer.begin(), buffer.sizeBytes(), WebSocket::FRAME_BINARY);
}

void WebSocketHandler::OnClose() {
  LOG(INFO) << "OnClose";
}

}  // namespace BASE_NAMESPACE