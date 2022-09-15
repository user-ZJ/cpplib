
#include "HttpFormHandler.h"
#include "utils/logging.h"
#include "utils/string-util.h"

namespace BASE_NAMESPACE {

void HTTPFormHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response) {
  try {
    LOG(INFO) << "URI: " << request.getURI() << "  Method:" << request.getMethod();
    MyPartHandler partHandler;
    HTMLForm form(request, request.stream(), partHandler);

    response.setChunkedTransferEncoding(true);
    response.setContentType("application/json");

    std::ostream &ostr = response.send();

    NameValueCollection::ConstIterator it = request.begin();
    NameValueCollection::ConstIterator end = request.end();
    LOG(INFO)<<"header:";
    for (; it != end; ++it) {
      LOG(INFO)<<"\t"<<it->first << ": " << it->second;
    }

    if (!form.empty()) {
      LOG(INFO)<<"Form:";
      it = form.begin();
      end = form.end();
      for (; it != end; ++it) {
        LOG(INFO)<<"\t"<<it->first << ": " << it->second;
      }
    }

    if (!partHandler.name().empty()) {
      LOG(INFO)<<"partHandler:";
      LOG(INFO) << "\tName: " << partHandler.name();
      LOG(INFO) << "\tFile Name: " << partHandler.fileName();
      LOG(INFO) << "\tType: " << partHandler.contentType() ;
      LOG(INFO) << "\tSize: " << partHandler.length();
    }
  } catch (const std::exception &e) { LOG(ERROR) << "http exception:" << e.what(); }
}

}  // namespace BASE_NAMESPACE