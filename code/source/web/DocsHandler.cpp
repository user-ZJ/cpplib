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

#include "DocsHandler.h"
#include "Poco/CountingStream.h"
#include "Poco/Dynamic/Struct.h"
#include "Poco/Exception.h"
#include "Poco/Net/HTMLForm.h"
#include "Poco/Net/HTTPRequestHandler.h"
#include "Poco/Net/HTTPRequestHandlerFactory.h"
#include "Poco/Net/HTTPServer.h"
#include "Poco/Net/HTTPServerParams.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "Poco/Net/HTTPServerResponse.h"
#include "Poco/Net/MessageHeader.h"
#include "Poco/Net/PartHandler.h"
#include "Poco/Net/ServerSocket.h"
#include "Poco/NullStream.h"
#include "Poco/StreamCopier.h"
#include "Poco/URI.h"
#include "Poco/Util/HelpFormatter.h"
#include "Poco/Util/Option.h"
#include "Poco/Util/OptionSet.h"
#include "Poco/Util/ServerApplication.h"
#include "status.h"
#include "utils/file-util.h"
#include "utils/logging.h"
#include <iostream>
#include <thread>

using Poco::CountingInputStream;
using Poco::NullOutputStream;
using Poco::StreamCopier;
using Poco::Net::HTMLForm;
using Poco::Net::HTTPRequestHandler;
using Poco::Net::HTTPRequestHandlerFactory;
using Poco::Net::HTTPServer;
using Poco::Net::HTTPServerParams;
using Poco::Net::HTTPServerRequest;
using Poco::Net::HTTPServerResponse;
using Poco::Net::MessageHeader;
using Poco::Net::NameValueCollection;
using Poco::Net::ServerSocket;
using Poco::Util::Application;
using Poco::Util::HelpFormatter;
using Poco::Util::Option;
using Poco::Util::OptionSet;
using Poco::Util::ServerApplication;

namespace BASE_NAMESPACE {

void DocsHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response) {
  LOG(INFO) << "docs handler";
  Poco::URI uri(request.getURI());
  response.setChunkedTransferEncoding(true);

  std::string path = "../docs/_build/html" + uri.getPath();
  if (not is_exist(path.c_str()) or uri.getPath() == "/") path = "../docs/_build/html/index.html";
  LOG(INFO) << path;
  if (path.find(".css") != std::string::npos) {
    response.setContentType("text/css");
  } else if (path.find(".js") != std::string::npos) {
    response.setContentType("application/javascript");
  } else if (path.find(".html") != std::string::npos) {
    response.setContentType("text/html");
  } else {
    response.setContentType("application/octet-stream");
  }
  auto buff = file_to_buff(path.c_str());
  response.sendBuffer(buff.data(), buff.size());
  LOG(INFO) << "docs handler end";
}

};  // namespace BASE_NAMESPACE
