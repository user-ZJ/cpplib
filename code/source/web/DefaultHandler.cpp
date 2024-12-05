
#include "DefaultHandler.h"
#include "utils/logging.h"
#include "utils/string-util.h"




namespace BASE_NAMESPACE {


void DefaultHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response) {
  response.setChunkedTransferEncoding(true);
  response.setContentType("application/json");
  std::ostream &ostr = response.send();
  ostr << R"({"code":9999,"msg":"default response,please check url"})";
}



} 