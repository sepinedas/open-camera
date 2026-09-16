cmake_minimum_required(VERSION 3.16)
project(open_lego_camera_cpp LANGUAGES CXX)

# MediaPipe's public headers are compiled as C++20 (its node framework uses
# `consteval` and class-type non-type template parameters), so everything that
# links against them has to be built the same way.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- Dependencies -----------------------------------------------------------
# SDL2 draws to HDMI via the KMSDRM driver on a headless Pi (no X11 needed).
find_package(SDL2 REQUIRED)
# OpenCV handles V4L2/GStreamer capture, JPEG/H.264 I/O and video playback, and
# the image processing (remap/resize/blend) the filters are built from.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)

find_package(PkgConfig REQUIRED)
# SDL2_gfx (icon primitives) has no CMake config on most distros -> pkg-config.
pkg_check_modules(SDL2_GFX REQUIRED SDL2_gfx)

# --- MediaPipe --------------------------------------------------------------
# The facial filters run MediaPipe's Face Landmarker (Tasks Vision C++ API).
# Prebuilt aarch64 artifacts come from https://github.com/sepinedas/media-pipe-builder.
#
# Three ways to find it, in order, so a hand-extracted tarball works as well as
# the .deb:
#   1. -DMEDIAPIPE_ROOT=/path  -- explicit, always wins
#   2. pkg-config `mediapipe`  -- what the .deb registers
#   3. probing the usual install layouts
set(MEDIAPIPE_ROOT "" CACHE PATH
    "Root of a MediaPipe install (the dir holding include/ and lib/)")

set(MEDIAPIPE_PROBE_HEADER mediapipe/framework/formats/image.h)

if(NOT MEDIAPIPE_ROOT)
  pkg_check_modules(MEDIAPIPE QUIET mediapipe)
endif()

if(NOT MEDIAPIPE_ROOT AND NOT MEDIAPIPE_FOUND)
  # No .pc registered. Look where the package and a hand-extracted tarball
  # normally land. /opt/mediapipe/current is the symlink the .deb maintains;
  # after it, take the highest-sorting versioned directory.
  file(GLOB MEDIAPIPE_VERSIONED LIST_DIRECTORIES true "/opt/mediapipe/*")
  if(MEDIAPIPE_VERSIONED)
    list(SORT MEDIAPIPE_VERSIONED)
    list(REVERSE MEDIAPIPE_VERSIONED)
  endif()
  # /usr is deliberately not probed: nothing installs MediaPipe there, and a
  # stray header would match while the library sits in a multiarch libdir this
  # does not search, turning a clean "not found" into a confusing hard error.
  foreach(dir "/opt/mediapipe/current" ${MEDIAPIPE_VERSIONED} "/usr/local")
    if(EXISTS "${dir}/include/${MEDIAPIPE_PROBE_HEADER}")
      set(MEDIAPIPE_ROOT "${dir}")
      message(STATUS "Found MediaPipe by probing: ${MEDIAPIPE_ROOT}")
      break()
    endif()
  endforeach()
endif()

if(MEDIAPIPE_ROOT)
  # No REQUIRED here: that needs CMake 3.18 while this project declares 3.16,
  # and an explicit check gives a more useful message anyway.
  find_path(MEDIAPIPE_INCLUDE_DIR ${MEDIAPIPE_PROBE_HEADER}
            PATHS "${MEDIAPIPE_ROOT}/include" NO_DEFAULT_PATH)
  find_library(MEDIAPIPE_LIBRARY mediapipe_tasks
               PATHS "${MEDIAPIPE_ROOT}/lib" NO_DEFAULT_PATH)
  if(NOT MEDIAPIPE_INCLUDE_DIR OR NOT MEDIAPIPE_LIBRARY)
    message(FATAL_ERROR
      "${MEDIAPIPE_ROOT} does not look like a MediaPipe install. Expected:\n"
      "    ${MEDIAPIPE_ROOT}/include/${MEDIAPIPE_PROBE_HEADER}\n"
      "    ${MEDIAPIPE_ROOT}/lib/libmediapipe_tasks.so")
  endif()
  set(MEDIAPIPE_INCLUDE_DIRS "${MEDIAPIPE_INCLUDE_DIR}")
  set(MEDIAPIPE_LIBRARIES "${MEDIAPIPE_LIBRARY}")
  # A tarball install is not on the ldconfig path the way the .deb is, so bake
  # the location in rather than making the user set LD_LIBRARY_PATH.
  set(MEDIAPIPE_RPATH "${MEDIAPIPE_ROOT}/lib")
elseif(MEDIAPIPE_FOUND)
  # pkg-config hands back a bare `-lmediapipe_tasks`; keep its -L too so a
  # versioned /opt/mediapipe/<ver>/lib install links without extra flags.
  link_directories(${MEDIAPIPE_LIBRARY_DIRS})
else()
  message(FATAL_ERROR
    "MediaPipe not found.\n"
    "The facial filters need MediaPipe's Tasks Vision C++ library "
    "(libmediapipe_tasks.so + headers + face_landmarker.task).\n"
    "Install the .deb from https://github.com/sepinedas/media-pipe-builder "
    "releases:\n"
    "    sudo dpkg -i libmediapipe_*_arm64.deb\n"
    "or point this build at an extracted tarball:\n"
    "    cmake -B build -S . -DMEDIAPIPE_ROOT=/path/to/mediapipe\n"
    "Note that the package must be built from media-pipe-builder at or after "
    "the commit that added overlay/force_export.cc's image-IO exports; older "
    "releases configure fine and then fail to link "
    "mediapipe::formats::MatView.")
endif()

add_executable(open-lego-camera
  src/main.cpp
  src/config.cpp
  src/camera.cpp
  src/battery.cpp
  src/gallery.cpp
  src/icons.cpp
  src/ui.cpp
  src/filters.cpp
  src/face3d.cpp
  src/app.cpp
)

target_include_directories(open-lego-camera PRIVATE
  ${SDL2_INCLUDE_DIRS}
  ${SDL2_GFX_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${MEDIAPIPE_INCLUDE_DIRS}
)

target_link_libraries(open-lego-camera PRIVATE
  ${SDL2_LIBRARIES}
  ${SDL2_GFX_LIBRARIES}
  ${OpenCV_LIBS}
  ${MEDIAPIPE_LIBRARIES}
  pthread
)

# MediaPipe here is built CPU-only; the GPU code paths in headers such as
# image.h are compiled out by the same define the library was built with.
target_compile_definitions(open-lego-camera PRIVATE MEDIAPIPE_DISABLE_GPU=1)

if(MEDIAPIPE_RPATH)
  set_target_properties(open-lego-camera PROPERTIES
                        BUILD_RPATH "${MEDIAPIPE_RPATH}"
                        INSTALL_RPATH "${MEDIAPIPE_RPATH}")
endif()

target_compile_options(open-lego-camera PRIVATE ${SDL2_GFX_CFLAGS_OTHER} -Wall -Wextra)

# --- Raspberry Pi 5 tuning --------------------------------------------------
# The target is a Pi 5 (BCM2712, Cortex-A76 @ 2.4 GHz). Naming the core lets the
# compiler assume the ARMv8.2-A baseline -- dot product, FP16 arithmetic, LSE
# atomics -- instead of the generic ARMv8-A that plain aarch64 targets, which
# matters most in the per-pixel loops (above all the warp map).
#
# The resulting binary will NOT run on a Pi 4, Pi 3 or Zero 2 W: those are
# Cortex-A72/A53 and fault with SIGILL. Turn this off for a portable build.
option(OLC_TUNE_PI5 "Build for the Raspberry Pi 5's Cortex-A76 (not portable)" ON)

if(OLC_TUNE_PI5 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mcpu=cortex-a76" OLC_HAVE_MCPU_CORTEX_A76)
  if(OLC_HAVE_MCPU_CORTEX_A76)
    target_compile_options(open-lego-camera PRIVATE -mcpu=cortex-a76)
  else()
    message(WARNING "OLC_TUNE_PI5 is on but the compiler rejects "
                    "-mcpu=cortex-a76; building generic aarch64.")
  endif()
endif()

# -O3 over CMake's Release default of -O2: the filters are dominated by tight
# per-pixel loops that benefit from the extra vectorisation.
target_compile_options(open-lego-camera PRIVATE $<$<CONFIG:Release>:-O3>)

install(TARGETS open-lego-camera RUNTIME DESTINATION bin)
