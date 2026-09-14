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
# Prebuilt aarch64 artifacts come from https://github.com/sepinedas/media-pipe-builder,
# whose .deb installs a pkg-config file; MEDIAPIPE_ROOT overrides the lookup for
# a tarball extracted somewhere else.
set(MEDIAPIPE_ROOT "" CACHE PATH
    "Root of a MediaPipe install (the dir holding include/ and lib/)")

if(MEDIAPIPE_ROOT)
  find_path(MEDIAPIPE_INCLUDE_DIR mediapipe/framework/formats/image.h
            PATHS "${MEDIAPIPE_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
  find_library(MEDIAPIPE_LIBRARY mediapipe_tasks
               PATHS "${MEDIAPIPE_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  set(MEDIAPIPE_INCLUDE_DIRS "${MEDIAPIPE_INCLUDE_DIR}")
  set(MEDIAPIPE_LIBRARIES "${MEDIAPIPE_LIBRARY}")
else()
  pkg_check_modules(MEDIAPIPE REQUIRED mediapipe)
  # pkg-config hands back a bare `-lmediapipe_tasks`; keep its -L too so a
  # versioned /opt/mediapipe/<ver>/lib install links without extra flags.
  link_directories(${MEDIAPIPE_LIBRARY_DIRS})
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
  src/pig3d.cpp
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

target_compile_options(open-lego-camera PRIVATE ${SDL2_GFX_CFLAGS_OTHER} -Wall -Wextra)

# --- Raspberry Pi 5 tuning --------------------------------------------------
# The target is a Pi 5 (BCM2712, Cortex-A76 @ 2.4 GHz). Naming the core lets the
# compiler assume the ARMv8.2-A baseline -- dot product, FP16 arithmetic, LSE
# atomics -- instead of the generic ARMv8-A that plain aarch64 targets, which
# matters most in the per-pixel loops (the warp map, the pig rasteriser).
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
