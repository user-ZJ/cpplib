#include "WebServer.h"
#include "DefaultHandler.h"
#include "HttpFormHandler.h"
#include "HttpGetHandler.h"
#include "HttpJsonHandler.h"
#include "HttpStreamHandler.h"
#include "WebsocketHandler.h"
#include "utils/logging.h"

namespace BASE_NAMESPACE {

int WebServer::start() {
  try {
    LOG(INFO) << "websocket listen at " << _port << " port, max threads " << _max_threads;
    HTTPServerParams *pParams = new HTTPServerParams;
    pParams->setMaxThreads(_max_threads);
    pParams->setMaxQueued(_max_threads);

    // set-up a server socket
    // ServerSocket svs(_port);
    Poco::Net::IPAddress wildcardAddr;
    Poco::Net::SocketAddress address(wildcardAddr, _port);
    ServerSocket svs;
    svs.bind(address, true, false);  // 地址可重用，端口不可重用
    svs.listen(64);
    // 默认线程池上限只有 16 且全进程共享，需单独创建；必须在 srv 之前构造、之后析构
    ThreadPool pool(2, _max_threads);
    HTTPServer srv(new RequestHandlerFactory, pool, svs, pParams);
    // start the HTTPServer
    srv.start();
    // wait for CTRL-C or kill
    waitForTerminationRequest();
    // 关闭现有连接的 socket，否则阻塞在 receiveFrame 的 WebSocket 会让进程无法退出
    srv.stopAll(true);

    return Application::EXIT_OK;
  }
  catch (Poco::Exception &e) {
    LOG(ERROR) << "server start error! " << e.displayText();
    return -1;
  }
}

HTTPRequestHandler *RequestHandlerFactory::createRequestHandler(const HTTPServerRequest &request) {
  LOG(INFO) << "Request from " << request.clientAddress().toString() << ": " << request.getMethod() << " "
            << request.getURI() << " " << request.getVersion();
  const std::string path = Poco::URI(request.getURI()).getPath();

  if (Poco::icompare(request.get("Upgrade", ""), "websocket") == 0)
    return new WebSocketHandler;
  // 先按路径分发，方法不匹配时由各 handler 返回 405
  if (path == "/form")
    return new HTTPFormHandler;
  if (path == "/json")
    return new HTTPJsonHandler;
  if (path == "/async_json")
    return new AsyncHTTPJsonHandler;
  if (path == "/octet_stream")
    return new HttpStreamHandler;
  if (request.getMethod() == Poco::Net::HTTPRequest::HTTP_GET)
    return new HTTPGetHandler;
  return new DefaultHandler;
}
};  // namespace BASE_NAMESPACE
