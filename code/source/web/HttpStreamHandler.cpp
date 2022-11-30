
#include "HttpStreamHandler.h"
#include "utils/logging.h"
#include "utils/string-util.h"

namespace BASE_NAMESPACE {

void HttpStreamHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response) {
  try {
    LOG(INFO) << "URI: " << request.getURI() << "  Method:" << request.getMethod();
    
    response.setChunkedTransferEncoding(true);
    response.setContentType("application/json");

    NameValueCollection::ConstIterator it = request.begin();
    NameValueCollection::ConstIterator end = request.end();
    LOG(INFO)<<"header:";
    for (; it != end; ++it) {
      LOG(INFO)<<"\t"<<it->first << ": " << it->second;
    }

    std::string binary_str;
    StreamCopier::copyToString(request.stream(), binary_str);

    response.setChunkedTransferEncoding(true);
    response.setContentType("application/json");

    std::ostream &ostr = response.send();
    Poco::JSON::Object obj;
    obj.set("code", "0000");
    obj.set("message", "success");
    obj.stringify(ostr);
    
  } catch (const std::exception &e) { LOG(ERROR) << "http exception:" << e.what(); }
}

}  // namespace BASE_NAMESPACE