# Both the armpl_mp and armpl libraries define the same public symbol names.
# In order to link against the openmp  armpl symbols, instruct cmake to link against armpl_mp.
# In order to link against the default armpl symbols, instruct cmake to link against armpl.
IF(KOKKOSKERNELS_INST_EXECSPACE_OPENMP)
  SET(ARMPL_LIB armpl_mp)
ELSE()
  SET(ARMPL_LIB armpl)
ENDIF()

IF (ARMPL_LIBRARY_DIRS AND ARMPL_LIBRARIES)
  KOKKOSKERNELS_FIND_IMPORTED(ARMPL INTERFACE LIBRARIES ${ARMPL_LIBRARIES} LIBRARY_PATHS ${ARMPL_LIBRARY_DIRS})
ELSEIF (ARMPL_LIBRARIES)
  KOKKOSKERNELS_FIND_IMPORTED(ARMPL INTERFACE LIBRARIES ${ARMPL_LIBRARIES})
ELSEIF (ARMPL_LIBRARY_DIRS)
  KOKKOSKERNELS_FIND_IMPORTED(ARMPL INTERFACE LIBRARIES amath ${ARMPL_LIB} LIBRARY_PATHS ${ARMPL_LIBRARY_DIRS})
ELSEIF (DEFINED ENV{ARMPL_DIR})
  SET(ARMPL_ROOT $ENV{ARMPL_DIR})
  KOKKOSKERNELS_FIND_IMPORTED(ARMPL INTERFACE
    LIBRARIES
      amath
      ${ARMPL_LIB}
    LIBRARY_PATHS
      ${ARMPL_ROOT}/lib
    HEADERS
      armpl.h
    HEADER_PATHS
      ${ARMPL_ROOT}/include
  )
ELSE()
  FIND_PACKAGE(ARMPL REQUIRED)
  KOKKOSKERNELS_CREATE_IMPORTED_TPL(ARMPL INTERFACE LINK_LIBRARIES ${ARMPL_LIBRARIES})
ENDIF()

TRY_COMPILE(KOKKOSKERNELS_TRY_COMPILE_ARMPL
  ${KOKKOSKERNELS_TOP_BUILD_DIR}/tpl_tests
  ${KOKKOSKERNELS_TOP_SOURCE_DIR}/cmake/compile_tests/armpl.cpp
  LINK_LIBRARIES -l${ARMPL_LIB} -lgfortran -lamath -lm
  OUTPUT_VARIABLE KOKKOSKERNELS_TRY_COMPILE_ARMPL_OUT)
IF(NOT KOKKOSKERNELS_TRY_COMPILE_ARMPL)
  MESSAGE(FATAL_ERROR "KOKKOSKERNELS_TRY_COMPILE_ARMPL_OUT=${KOKKOSKERNELS_TRY_COMPILE_ARMPL_OUT}")
ELSE()
  # KokkosKernels::ARMPL is an alias to the ARMPL target.
  # Let's add in the libgfortran and libm dependencies for users here.
  GET_TARGET_PROPERTY(ARMPL_INTERFACE_LINK_LIBRARIES KokkosKernels::ARMPL INTERFACE_LINK_LIBRARIES)
  SET(ARMPL_INTERFACE_LINK_LIBRARIES "${ARMPL_INTERFACE_LINK_LIBRARIES};-lgfortran;-lm")
  SET_TARGET_PROPERTIES(ARMPL PROPERTIES INTERFACE_LINK_LIBRARIES "${ARMPL_INTERFACE_LINK_LIBRARIES}")
ENDIF()