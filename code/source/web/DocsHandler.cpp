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
#include "HttpUtil.h"
#include "Poco/Path.h"
#include "utils/file-util.h"
#include "utils/logging.h"
#include <iostream>
#include <vector>

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
  HandleSafely(request, response, [&] {
    static const std::string kDocRoot = "../docs/_build/html";
    Poco::URI uri(request.getURI());

    // getPath() 已做百分号解码，任何 ".." 段都可能逃出文档目录
    std::vector<std::string> segments;
    uri.getPathSegments(segments);
    for (const auto &seg : segments) {
      if (seg == ".." || seg.find('\\') != std::string::npos) {
        throw HttpError(Poco::Net::HTTPResponse::HTTP_NOT_FOUND, "not found");
      }
    }

    std::string path = kDocRoot + uri.getPath();
    if (uri.getPath() == "/" || !is_exist(path.c_str())) path = kDocRoot + "/index.html";
    VLOG(1) << "docs: " << path;

    const std::string ext = Poco::Path(path).getExtension();
    if (ext == "css") {
      response.setContentType("text/css");
    } else if (ext == "js") {
      response.setContentType("application/javascript");
    } else if (ext == "html") {
      response.setContentType("text/html");
    } else {
      response.setContentType("application/octet-stream");
    }
    auto buff = file_to_buff(path.c_str());
    response.sendBuffer(buff.data(), buff.size());
  });
}

};  // namespace BASE_NAMESPACE
