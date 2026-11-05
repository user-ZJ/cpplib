set(web_source_dir "${CMAKE_SOURCE_DIR}/source/web")
add_executable(web_server_bin
    WebServerBin.cpp
    ${web_source_dir}/WebServer.cpp
    ${web_source_dir}/WebsocketHandler.cpp
    ${web_source_dir}/HttpGetHandler.cpp
    ${web_source_dir}/HttpJsonHandler.cpp
    ${web_source_dir}/HttpFormHandler.cpp
    ${web_source_dir}/HttpStreamHandler.cpp
    ${web_source_dir}/DefaultHandler.cpp
)
target_link_libraries(web_server_bin PRIVATE Poco::Net Poco::Util Poco::XML Poco::JSON Poco::Foundation pthread glog gflags)
