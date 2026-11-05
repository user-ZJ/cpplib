cmake_minimum_required(VERSION 3.14)
project(web)



include_directories(
		 )
# link_directories(
# 	/home/dmai/code/poco-1.11.1-all/build/lib
# 		)
aux_source_directory(. SOURCE_LIST)

add_library(dbwrapper SHARED ${SOURCE_LIST})
add_dependencies(dbwrapper Poco::Redis Poco::MongoDB Poco::DataMySQL Poco::Data Poco::Net Poco::Foundation)
target_link_libraries(dbwrapper Poco::Redis Poco::MongoDB Poco::DataMySQL Poco::Data Poco::Net Poco::Foundation mysqlclient myutil)

