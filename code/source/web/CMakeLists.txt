cmake_minimum_required(VERSION 3.14)
project(web)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


include_directories(
		 )
link_directories(
		)


add_executable(http_client HTTPClient.cpp)
add_dependencies(http_client Poco::Net Poco::Util Poco::XML Poco::JSON Poco::Foundation)
target_link_libraries(http_client Poco::Net Poco::Util Poco::XML Poco::JSON Poco::Foundation pthread glog gflags)

add_executable(websocket_client WebSocketClient.cpp)
add_dependencies(websocket_client Poco::Net Poco::Util Poco::XML Poco::JSON Poco::Foundation)
target_link_libraries(websocket_client Poco::Net Poco::Util Poco::XML Poco::JSON Poco::Foundation pthread glog gflags)

add_executable(web_server_bin WebServerBin.cpp WebServer.cpp WebsocketHandler.cpp HttpGetHandler.cpp HttpJsonHandler.cpp HttpFormHandler.cpp HttpStreamHandler.cpp DefaultHandler.cpp)
add_dependencies(web_server_bin Poco::Net Poco::Util Poco::XML Poco::JSON Poco::Foundation)
target_link_libraries(web_server_bin Poco::Net Poco::Util Poco::XML Poco::JSON Poco::Foundation pthread glog gflags)