#pragma once
#include "Poco/Net/HTTPRequestHandler.h"
#include "Poco/Net/HTTPRequestHandlerFactory.h"
#include "Poco/Net/HTTPServer.h"
#include "Poco/Net/HTTPServerParams.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "Poco/Net/HTTPServerResponse.h"
#include "Poco/Net/NetException.h"
#include "Poco/Net/ServerSocket.h"
#include "Poco/StreamCopier.h"
#include "Poco/Util/ServerApplication.h"

using Poco::StreamCopier;
using Poco::Net::HTTPRequestHandler;
using Poco::Net::HTTPRequestHandlerFactory;
using Poco::Net::HTTPResponse;
using Poco::Net::HTTPServer;
using Poco::Net::HTTPServerParams;
using Poco::Net::HTTPServerRequest;
using Poco::Net::HTTPServerResponse;
using Poco::Net::ServerSocket;

namespace DMAI {

class RestfulHandler : public HTTPRequestHandler
/// Return a HTML document with some JavaScript creating
/// a WebSocket connection.
{
public:
  void handleRequest(HTTPServerRequest &request, HTTPServerResponse &response);
  virtual void GET(HTTPServerRequest &request,HTTPServerResponse &response);
  virtual void POST(HTTPServerRequest &request,HTTPServerResponse &response);
  virtual void PUT(HTTPServerRequest &request,HTTPServerResponse &response);
  virtual void PATCH(HTTPServerRequest &request,HTTPServerResponse &response);
  virtual void DELETE(HTTPServerRequest &request,HTTPServerResponse &response);
  virtual void HEAD(HTTPServerRequest &request,HTTPServerResponse &response);
  virtual void OPTIONS(HTTPServerRequest &request,HTTPServerResponse &response);
};

}; // namespace DMAI