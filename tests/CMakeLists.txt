file(GLOB all-py-tests ./*.py)
file(COPY ${all-py-tests} DESTINATION .)

add_custom_target(python_tests
    DEPENDS ${all-py-tests}
    COMMAND TESTS_DIR=${CMAKE_CURRENT_BINARY_DIR} ${PYTHON} -m pytest -v --rootdir=${CMAKE_CURRENT_BINARY_DIR} --junitxml=reports/junit/report.xml ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(python_tests_with_coverage
    DEPENDS ${all-py-tests}
    COMMAND TESTS_DIR=${CMAKE_CURRENT_BINARY_DIR} PYTHONPATH=${PROJECT_DIR}/python:$$PYTHONPATH ${PYTHON} -m coverage run -m pytest -v --rootdir=${CMAKE_CURRENT_BINARY_DIR} --junitxml=reports/junit/report.xml ${CMAKE_CURRENT_BINARY_DIR}
)

#
# Create check target
#
if("$ENV{USE_VALGRIND}" STREQUAL "1")
    add_custom_target(check DEPENDS python_tests)
    add_custom_target(coverage DEPENDS python_tests_with_coverage)
else()
    add_custom_target(check DEPENDS python_tests)
    add_custom_target(coverage DEPENDS python_tests_with_coverage)
endif()

