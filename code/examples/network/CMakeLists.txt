add_executable(http_client HTTPClient.cpp)
target_link_libraries(http_client PRIVATE Poco::Net Poco::Util Poco::XML Poco::JSON Poco::Foundation pthread glog gflags)
add_executable(websocket_client WebSocketClient.cpp)
target_link_libraries(websocket_client PRIVATE Poco::Net Poco::Util Poco::XML Poco::JSON Poco::Foundation pthread glog gflags)
