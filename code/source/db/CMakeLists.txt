cmake_minimum_required(VERSION 3.14)
project(web)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


include_directories(
		 )
# link_directories(
# 	/home/dmai/code/poco-1.11.1-all/build/lib
# 		)


add_library(dbwrapper STATIC RedisWrapper.cpp MongoDBWrapper.cpp MYSQLWrapper.cpp)
target_link_libraries(dbwrapper PocoRedis PocoMongoDB PocoDataMySQL PocoData PocoNet PocoFoundation mysqlclient utils)

