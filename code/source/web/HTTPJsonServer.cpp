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

#include "Poco/CountingStream.h"
#include "Poco/Exception.h"
#include "Poco/JSON/Parser.h"
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
#include "Poco/Util/HelpFormatter.h"
#include "Poco/Util/Option.h"
#include "Poco/Util/OptionSet.h"
#include "Poco/Util/ServerApplication.h"
#include <iostream>

using Poco::CountingInputStream;
using Poco::NullOutputStream;
using Poco::StreamCopier;
using Poco::JSON::Object;
using Poco::JSON::Parser;
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

class JsonRequestHandler : public HTTPRequestHandler
/// Return a HTML document with the current date and time.
{
public:
  JsonRequestHandler() {}

  void handleRequest(HTTPServerRequest &request, HTTPServerResponse &response) {
    Application &app = Application::instance();
    app.logger().information("Request from " +
                             request.clientAddress().toString());

    Parser parser;
    auto result = parser.parse(request.stream());
    auto pObject = result.extract<Object::Ptr>();
    std::string username = pObject->getValue<std::string>("username");
    std::string password = pObject->getValue<std::string>("password");
    std::cout << "username:" << username << std::endl;
    std::cout << "password:" << password << std::endl;

    response.setChunkedTransferEncoding(true);
    response.setContentType("application/json");

    std::ostream &ostr = response.send();

    std::cout << "Request\n";
    std::cout << "Method: " << request.getMethod() << "\n";
    std::cout << "URI: " << request.getURI() << "\n";
    NameValueCollection::ConstIterator it = request.begin();
    NameValueCollection::ConstIterator end = request.end();
    for (; it != end; ++it) {
      std::cout << it->first << ": " << it->second << "\n";
    }

    Poco::JSON::Object obj;
    obj.set("code", "0000");
    obj.set("message", "success");
    obj.stringify(ostr);
    
  }
};

class JsonRequestHandlerFactory : public HTTPRequestHandlerFactory {
public:
  JsonRequestHandlerFactory() {}

  HTTPRequestHandler *createRequestHandler(const HTTPServerRequest &request) {
    // if (request.getURI() == "/")
    // //   return new RootHandler();
    // else
    return new JsonRequestHandler();
  }
};

class HTTPFormServer : public Poco::Util::ServerApplication
/// The main application class.
///
/// This class handles command-line arguments and
/// configuration files.
/// Start the HTTPFormServer executable with the help
/// option (/help on Windows, --help on Unix) for
/// the available command line options.
///
/// To use the sample configuration file (HTTPFormServer.properties),
/// copy the file to the directory where the HTTPFormServer executable
/// resides. If you start the debug version of the HTTPFormServer
/// (HTTPFormServerd[.exe]), you must also create a copy of the configuration
/// file named HTTPFormServerd.properties. In the configuration file, you
/// can specify the port on which the server is listening (default
/// 9980) and the format of the date/Form string sent back to the client.
///
/// To test the FormServer you can use any web browser (http://localhost:9980/).
{
public:
  HTTPFormServer() : _helpRequested(false) {}

  ~HTTPFormServer() {}

protected:
  void initialize(Application &self) {
    loadConfiguration(); // load default configuration files, if present
    ServerApplication::initialize(self);
  }

  void uninitialize() { ServerApplication::uninitialize(); }

  void defineOptions(OptionSet &options) {
    ServerApplication::defineOptions(options);

    options.addOption(
        Option("help", "h",
               "display help information on command line arguments")
            .required(false)
            .repeatable(false));
  }

  void handleOption(const std::string &name, const std::string &value) {
    ServerApplication::handleOption(name, value);

    if (name == "help")
      _helpRequested = true;
  }

  void displayHelp() {
    HelpFormatter helpFormatter(options());
    helpFormatter.setCommand(commandName());
    helpFormatter.setUsage("OPTIONS");
    helpFormatter.setHeader(
        "A web server that shows how to work with HTML forms.");
    helpFormatter.format(std::cout);
  }

  int main(const std::vector<std::string> &args) {
    if (_helpRequested) {
      displayHelp();
    } else {
      unsigned short port =
          (unsigned short)config().getInt("HTTPFormServer.port", 9980);
      HTTPServerParams *pParams = new HTTPServerParams;
      pParams->setMaxQueued(100);
      pParams->setMaxThreads(8);

      // set-up a server socket
      ServerSocket svs(port);
      // set-up a HTTPServer instance
      HTTPServer srv(new JsonRequestHandlerFactory, svs, pParams);
      // start the HTTPServer
      srv.start();
      // wait for CTRL-C or kill
      waitForTerminationRequest();
      // Stop the HTTPServer
      srv.stop();
    }
    return Application::EXIT_OK;
  }

private:
  bool _helpRequested;
};

int main(int argc, char **argv) {
  HTTPFormServer app;
  return app.run(argc, argv);
}
