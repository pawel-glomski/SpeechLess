cmake_minimum_required(VERSION 3.8)
project(Editor
    VERSION 0.5.0
    LANGUAGES C)

add_executable(editor ${PROJECT_SOURCE_DIR}/src/Editor/editor.c)
target_link_libraries(editor avformat avcodec avutil)
