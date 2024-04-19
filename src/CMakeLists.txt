

find_package(OpenCV REQUIRED)
find_package(ade REQUIRED)

add_executable(app 
    testing.cpp
)

# Additional Include Directories
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Additional Library Directories
link_directories( ${OpenCV_LIB_DIR} )

target_link_libraries(app PRIVATE
    ${OpenCV_LIBS}
    ade
)



