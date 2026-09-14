cmake_minimum_required(VERSION 3.16)
project(open_lego_camera_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- Dependencies -----------------------------------------------------------
# SDL2 draws to HDMI via the KMSDRM driver on a headless Pi (no X11 needed).
find_package(SDL2 REQUIRED)
# OpenCV handles V4L2/GStreamer capture, JPEG/H.264 I/O and video playback.
# objdetect provides the Haar cascade classifier used by the facial filters.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio objdetect)

# SDL2_gfx (icon primitives) has no CMake config on most distros -> pkg-config.
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2_GFX REQUIRED SDL2_gfx)

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
)

target_link_libraries(open-lego-camera PRIVATE
  ${SDL2_LIBRARIES}
  ${SDL2_GFX_LIBRARIES}
  ${OpenCV_LIBS}
  pthread
)

target_compile_options(open-lego-camera PRIVATE ${SDL2_GFX_CFLAGS_OTHER} -Wall -Wextra)

install(TARGETS open-lego-camera RUNTIME DESTINATION bin)
