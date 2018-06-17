cmake_minimum_required(VERSION 3.0)

enable_testing()

option(WITH_TESTING "Include unit testing" ON)

function(py_test TARGET_NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS ARGS ENVS)
  cmake_parse_arguments(py_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  add_test(NAME ${TARGET_NAME}
    COMMAND env PYTHONPATH=.:${py_test_ENVS}
    python -u ${py_test_SRCS} ${py_test_ARGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

if (WITH_TESTING)
  file(GLOB_RECURSE TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
  string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
  foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
  endforeach()
endif()
