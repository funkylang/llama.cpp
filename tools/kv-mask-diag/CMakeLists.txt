add_executable(llama-kv-mask-diag kv-mask-diag.cpp)
target_link_libraries(llama-kv-mask-diag PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(llama-kv-mask-diag PRIVATE ${CMAKE_SOURCE_DIR}/src)
