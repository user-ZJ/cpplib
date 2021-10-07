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

# add_executable(path_util_test path_test.cpp)
# target_link_libraries(path_util_test mbase)

# add_executable(ctensor_test ctensor_test.cpp)
# target_link_libraries(ctensor_test mbase)

