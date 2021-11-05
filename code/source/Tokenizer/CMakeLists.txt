cmake_minimum_required(VERSION 3.5)
project(tokenizer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(
	${CMAKE_SOURCE_DIR}
	${CMAKE_SOURCE_DIR}/tools/include
		 )

link_directories(
	${CMAKE_SOURCE_DIR}/tools/libs/tflite/x64
        )

set(HEADERS Tokenizer.h)

set(SOURCES_LIST
		Tokenizer.cpp
		)

add_library(${PROJECT_NAME} SHARED ${SOURCES_LIST})

# target_link_libraries( ${PROJECT_NAME}	
#    )

# INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
# install(FILES ${HEADERS} DESTINATION include)



# set(TOKENBIN token_test)
# add_executable(${TOKENBIN} TokenizerTest.cpp)
# target_link_libraries( ${TOKENBIN}
# 		tokenizer
#    )
# INSTALL(TARGETS ${TOKENBIN} RUNTIME DESTINATION bin)

