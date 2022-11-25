#include "WebServer.h"
#include "DefaultHandler.h"
#include "HttpFormHandler.h"
#include "HttpGetHandler.h"
#include "HttpJsonHandler.h"
#include "WebsocketHandler.h"
#include "utils/logging.h"

#include <thread>

namespace BASE_NAMESPACE {

int WebServer::start() {
  try {
    LOG(INFO) << "websocet listen at " << _port << " port";
    HTTPServerParams *pParams = new HTTPServerParams;
    pParams->setMaxQueued(100);
    pParams->setMaxThreads(std::thread::hardware_concurrency());

    // set-up a server socket
    // ServerSocket svs(_port);
    Poco::Net::IPAddress wildcardAddr;
    Poco::Net::SocketAddress address(wildcardAddr, _port);
    ServerSocket svs;
    svs.bind(address, true, false);  // 地址可重用，端口不可重用
    svs.listen(64);
    // set-up a HTTPServer instance
    HTTPServer srv(new RequestHandlerFactory, svs, pParams);
    // start the HTTPServer
    srv.start();
    // wait for CTRL-C or kill
    waitForTerminationRequest();
    // Stop the HTTPServer
    srv.stop();

    return Application::EXIT_OK;
  }
  catch (Poco::Exception &e) {
    LOG(ERROR) << "server start error! " << e.displayText();
    return -1;
  }
}

HTTPRequestHandler *RequestHandlerFactory::createRequestHandler(const HTTPServerRequest &request) {
  std::string s;
  s = "Request from " + request.clientAddress().toString() + ": " + request.getMethod() + " " + request.getURI() + " "
      + request.getVersion();
  LOG(INFO) << s;

  if (request.find("Upgrade") != request.end() && Poco::icompare(request["Upgrade"], "websocket") == 0)
    return new WebSocketHandler;
  else if (request.getMethod() == "GET") {
    return new HTTPGetHandler;
  } else if (request.getURI() == "/form") {
    return new HTTPFormHandler;
  } else if (request.getURI() == "/json") {
    return new HTTPJsonHandler;
  } else
    return new DefaultHandler;
}
};  // namespace BASE_NAMESPACE
