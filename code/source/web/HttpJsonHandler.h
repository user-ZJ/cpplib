#pragma once
#include "Poco/Net/HTTPRequestHandler.h"
#include "Poco/Net/HTTPRequestHandlerFactory.h"
#include "Poco/Net/HTTPServerParams.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "Poco/Net/HTTPServerResponse.h"
#include "Poco/Net/HTTPServerParams.h"
#include "Poco/Net/ServerSocket.h"
#include "Poco/Net/WebSocket.h"
#include "Poco/Net/NetException.h"
#include "Poco/Util/ServerApplication.h"
#include "Poco/Util/Option.h"
#include "Poco/Util/OptionSet.h"
#include "Poco/Util/HelpFormatter.h"
#include "Poco/Format.h"
#include "Poco/StreamCopier.h"
#include "Poco/JSON/Parser.h"

using Poco::Net::ServerSocket;
using Poco::Net::WebSocket;
using Poco::Net::WebSocketException;
using Poco::Net::HTTPRequestHandler;
using Poco::Net::HTTPRequestHandlerFactory;
using Poco::Net::HTTPServerRequest;
using Poco::Net::HTTPResponse;
using Poco::Net::HTTPServerResponse;
using Poco::Net::HTTPServerParams;
using Poco::Timestamp;
// using Poco::ThreadPool;
using Poco::Util::ServerApplication;
using Poco::Util::Application;
using Poco::Util::Option;
using Poco::Util::OptionSet;
using Poco::Util::HelpFormatter;
using Poco::StreamCopier;
using Poco::JSON::Parser;
using Poco::JSON::Object;
using Poco::Net::NameValueCollection;

namespace BASE_NAMESPACE{


class HTTPJsonHandler: public HTTPRequestHandler
	/// Handle a JSON POST request.
{
public:
	void handleRequest(HTTPServerRequest& request, HTTPServerResponse& response) override;

private:

};

class AsyncHTTPJsonHandler: public HTTPRequestHandler
	/// Accept a JSON POST request and reply 202 immediately.
{
public:
	void handleRequest(HTTPServerRequest& request, HTTPServerResponse& response) override;

private:

};

};