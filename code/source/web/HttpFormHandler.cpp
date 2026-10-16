
#include "HttpFormHandler.h"
#include "HttpUtil.h"
#include "utils/logging.h"

namespace BASE_NAMESPACE {

void MyPartHandler::handlePart(const MessageHeader &header, std::istream &stream) {
  _type = header.get("Content-Type", "(unspecified)");
  if (header.has("Content-Disposition")) {
    std::string disp;
    NameValueCollection params;
    MessageHeader::splitParameters(header["Content-Disposition"], disp, params);
    _name = params.get("name", "(unnamed)");
    _fileName = params.get("filename", "(unnamed)");
  }
  ReadLimited(stream, filebytes);
}

void HTTPFormHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response) {
  HandleSafely(request, response, [&] {
    RequireMethod(request, response, Poco::Net::HTTPRequest::HTTP_POST);
    CheckContentLength(request);
    MyPartHandler partHandler;
    HTMLForm form(request, request.stream(), partHandler);

    // 字段可能含密码，只记录字段名
    if (VLOG_IS_ON(1)) {
      for (const auto &kv : form) VLOG(1) << "form field: " << kv.first;
    }

    Poco::JSON::Object obj;
    obj.set("code", "0000");
    obj.set("message", "success");
    obj.set("fields", form.size());
    if (!partHandler.name().empty()) {
      Poco::JSON::Object file;
      file.set("name", partHandler.name());
      file.set("filename", partHandler.fileName());
      file.set("content_type", partHandler.contentType());
      file.set("size", partHandler.length());
      obj.set("file", file);
    }
    SendJson(response, obj);
  });
}

}  // namespace BASE_NAMESPACE
