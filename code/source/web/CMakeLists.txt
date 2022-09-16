cmake_minimum_required(VERSION 3.14)
project(web)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


include_directories(
		 )
link_directories(
		)


add_executable(http_client HTTPClient.cpp)
target_link_libraries(http_client PocoNet PocoUtil PocoXML PocoJSON PocoFoundation pthread glog gflags)

add_executable(websocket_client WebSocketClient.cpp)
target_link_libraries(websocket_client PocoNet PocoUtil PocoXML PocoJSON PocoFoundation pthread glog gflags)

add_executable(web_server_bin WebServerBin.cpp WebServer.cpp WebsocketHandler.cpp HttpGetHandler.cpp HttpJsonHandler.cpp HttpFormHandler.cpp DefaultHandler.cpp)
target_link_libraries(web_server_bin PocoNet PocoUtil PocoXML PocoJSON PocoFoundation pthread glog gflags)