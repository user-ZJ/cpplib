cmake_minimum_required(VERSION 3.14)
project(web)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


include_directories(
		 )
link_directories(
		)


# add_executable(WebSocketServer WebSocketServer.cpp)
# target_link_libraries(WebSocketServer PocoNet PocoUtil PocoXML PocoJSON PocoFoundation pthread)

# add_executable(WebSocketClient WebSocketClient.cpp)
# target_link_libraries(WebSocketClient PocoNet PocoUtil PocoXML PocoJSON PocoFoundation pthread)

# add_executable(HTTPJsonServer HTTPJsonServer.cpp)
# target_link_libraries(HTTPJsonServer PocoNet PocoUtil PocoXML PocoJSON PocoFoundation pthread)

# add_executable(HTTPJsonClient HTTPJsonClient.cpp)
# target_link_libraries(HTTPJsonClient PocoNet PocoUtil PocoXML PocoJSON PocoFoundation pthread)

# add_executable(HTTPFormServer HTTPFormServer.cpp)
# target_link_libraries(HTTPFormServer PocoNet PocoUtil PocoXML PocoJSON PocoFoundation pthread)

# add_executable(HTTPFormClient HTTPFormClient.cpp)
# target_link_libraries(HTTPFormClient PocoNet PocoUtil PocoXML PocoJSON PocoFoundation pthread)

add_executable(http_client HTTPClient.cpp)
target_link_libraries(http_client PocoNet PocoUtil PocoXML PocoJSON PocoFoundation pthread glog gflags)

add_executable(websocket_client WebSocketClient.cpp)
target_link_libraries(websocket_client PocoNet PocoUtil PocoXML PocoJSON PocoFoundation pthread glog gflags)

add_executable(web_server_bin WebServerBin.cpp WebServer.cpp WebsocketHandler.cpp HttpGetHandler.cpp HttpJsonHandler.cpp HttpFormHandler.cpp DefaultHandler.cpp)
target_link_libraries(web_server_bin PocoNet PocoUtil PocoXML PocoJSON PocoFoundation pthread glog gflags)