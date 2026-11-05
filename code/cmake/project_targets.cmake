# Apply project policy only to targets below the supplied source directories.
# Imported dependency targets keep their installed compiler settings.
function(mbase_configure_targets directory)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
        get_target_property(type "${target}" TYPE)
        if(type STREQUAL "INTERFACE_LIBRARY" OR type STREQUAL "UTILITY")
            continue()
        endif()
        target_include_directories("${target}" PRIVATE "${PROJECT_SOURCE_DIR}/source")
        set_target_properties("${target}" PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED YES
            POSITION_INDEPENDENT_CODE YES)
        target_compile_definitions("${target}" PRIVATE "$<$<CONFIG:Debug>:DEBUG>")
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options("${target}" PRIVATE
                "$<$<COMPILE_LANGUAGE:C,CXX>:-Wfatal-errors>"
                "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<NOT:$<CONFIG:Debug>>>:-O3>")
        endif()
    endforeach()
    get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(child IN LISTS children)
        mbase_configure_targets("${child}")
    endforeach()
endfunction()
