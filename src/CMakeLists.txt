cmake_minimum_required(VERSION 3.2)
PROJECT(project)

INCLUDE_DIRECTORIES(
	common
	)
include_directories(${OpenCV_INCLUDE_DIRS})

FIND_PACKAGE(OpenCV REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(Threads)
set (name tm_hrnet_timvx) 
IF (OpenCV_FOUND)
        ADD_EXECUTABLE (tm_hrnet_timvx "${CMAKE_CURRENT_SOURCE_DIR}/tm_hrnet_timvx.cpp"   "${CMAKE_CURRENT_SOURCE_DIR}/json_util.cpp"  "${CMAKE_CURRENT_SOURCE_DIR}/points_in_polygon.cpp"  "${CMAKE_CURRENT_SOURCE_DIR}/common/tengine_operations.c" )
	    TARGET_LINK_LIBRARIES (${name} jsoncpp_lib tengine-lite ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
		# for test
		#ADD_EXECUTABLE (test2images "${CMAKE_CURRENT_SOURCE_DIR}/common/tengine_operations.c" "${CMAKE_CURRENT_SOURCE_DIR}/json_util.cpp" )
		#TARGET_LINK_LIBRARIES (test2images tengine-lite ${OpenCV_LIBS})

	INSTALL (TARGETS ${name} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../bin)
ELSE()
    MESSAGE (WARNING "OpenCV not found, some examples won't be built")
ENDIF()
