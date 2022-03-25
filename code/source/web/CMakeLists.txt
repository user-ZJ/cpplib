cmake_minimum_required(VERSION 3.14)
project(web)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


include_directories(
		 )
link_directories(
		)


add_executable(http_sync_client http_sync_client.cpp)
target_link_libraries(http_sync_client pthread)

add_executable(http_sync_server http_sync_server.cpp)
target_link_libraries(http_sync_server pthread)

add_executable(websocket_sync_client websocket_sync_client.cpp)
target_link_libraries(websocket_sync_client pthread)

add_executable(websocket_sync_server websocket_sync_server.cpp)
target_link_libraries(websocket_sync_server pthread)

add_executable(HTTPJsonServer HTTPJsonServer.cpp)
target_link_libraries(HTTPJsonServer PocoNet PocoUtil PocoXML PocoJSON PocoFoundation pthread)

add_executable(HTTPJsonClient HTTPJsonClient.cpp)
target_link_libraries(HTTPJsonClient PocoNet PocoUtil PocoXML PocoJSON PocoFoundation pthread)

add_executable(HTTPFormServer HTTPFormServer.cpp)
target_link_libraries(HTTPFormServer PocoNet PocoUtil PocoXML PocoJSON PocoFoundation pthread)

add_executable(HTTPFormClient HTTPFormClient.cpp)
target_link_libraries(HTTPFormClient PocoNet PocoUtil PocoXML PocoJSON PocoFoundation pthread)

