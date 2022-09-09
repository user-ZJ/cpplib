
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
    response.setContentType("text/html");

    std::ostream &ostr = response.send();

    ostr << "<html>\n"
            "<head>\n"
            "<title>POCO Form Server Sample</title>\n"
            "</head>\n"
            "<body>\n"
            "<h1>POCO Form Server Sample</h1>\n"
            "<h2>GET Form</h2>\n"
            "<form method=\"GET\" action=\"/form\">\n"
            "<input type=\"text\" name=\"text\" size=\"31\">\n"
            "<input type=\"submit\" value=\"GET\">\n"
            "</form>\n"
            "<h2>POST Form</h2>\n"
            "<form method=\"POST\" action=\"/form\">\n"
            "<input type=\"text\" name=\"text\" size=\"31\">\n"
            "<input type=\"submit\" value=\"POST\">\n"
            "</form>\n"
            "<h2>File Upload</h2>\n"
            "<form method=\"POST\" action=\"/form\" enctype=\"multipart/form-data\">\n"
            "<input type=\"file\" name=\"file\" size=\"31\"> \n"
            "<input type=\"submit\" value=\"Upload\">\n"
            "</form>\n";

    ostr << "<h2>Request</h2><p>\n";
    ostr << "Method: " << request.getMethod() << "<br>\n";
    ostr << "URI: " << request.getURI() << "<br>\n";
    NameValueCollection::ConstIterator it = request.begin();
    NameValueCollection::ConstIterator end = request.end();
    for (; it != end; ++it) {
      ostr << it->first << ": " << it->second << "<br>\n";
    }
    ostr << "</p>";

    if (!form.empty()) {
      ostr << "<h2>Form</h2><p>\n";
      it = form.begin();
      end = form.end();
      for (; it != end; ++it) {
        ostr << it->first << ": " << it->second << "<br>\n";
      }
      ostr << "</p>";
    }

    if (!partHandler.name().empty()) {
      ostr << "<h2>Upload</h2><p>\n";
      ostr << "Name: " << partHandler.name() << "<br>\n";
      ostr << "File Name: " << partHandler.fileName() << "<br>\n";
      ostr << "Type: " << partHandler.contentType() << "<br>\n";
      ostr << "Size: " << partHandler.length() << "<br>\n";
      ostr << "</p>";
    }
    ostr << "</body>\n";
  } catch (const std::exception &e) { LOG(ERROR) << "http exception:" << e.what(); }
}

}  // namespace BASE_NAMESPACE