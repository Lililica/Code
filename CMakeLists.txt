cmake_minimum_required(VERSION 3.17)

# You can set the name of your project here
project(conicFiltring VERSION 0.1.0 LANGUAGES C CXX)
include(FetchContent)
FetchContent_Declare(
    sil
    GIT_REPOSITORY https://github.com/JulesFouchy/Simple-Image-Lib
    GIT_TAG 89bd175cd55fb6daa9e99b218e869c3d516043f4
)
FetchContent_MakeAvailable(sil)




function(add_exercise FOLDER)
    add_executable(${FOLDER})

    # include Eigen
    find_package(Eigen3 REQUIRED)
    if(${EIGEN3_FOUND})
        message(STATUS "lib EIGEN3 found")
        message(STATUS "  version " ${EIGEN3_VERSION_STRING})
        message(STATUS "  include " ${EIGEN3_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "lib EIGEN3 not found")
    endif()


    # Trouver OpenCV
    find_package(OpenCV REQUIRED)

    # Vérifiez si GifLib est installé
    find_package(GIF REQUIRED)
    if (NOT GIF_FOUND)
        message(FATAL_ERROR "GifLib not found!")
    endif()

    # Choose your C++ version
    target_compile_features(${FOLDER} PRIVATE cxx_std_17)

    # Prevents compiler-specific extensions to C++ because they might allow code to compile on your machine but not on other people's machine
    set_target_properties(${FOLDER} PROPERTIES
        CXX_EXTENSIONS OFF)

    # Add all the source files
    file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/${FOLDER}/* lib/*)
    target_sources(${FOLDER} PRIVATE ${SOURCES})



    # Link sil into the project
    target_link_libraries(${FOLDER} PRIVATE sil)
    target_include_directories(${FOLDER} PRIVATE "${EIGEN3_INCLUDE_DIR}")


    # Liez OpenCV à l'exécutable
    target_link_libraries(${FOLDER} PRIVATE ${OpenCV_LIBS})

    # Inclure le répertoire d'en-têtes d'OpenCV
    include_directories(${OpenCV_INCLUDE_DIRS})

    target_include_directories(${FOLDER} PRIVATE ${GIF_INCLUDE_DIR})
    target_link_libraries(${FOLDER} PRIVATE ${OpenCV_LIBS} GIF::GIF)



endfunction(add_exercise)

add_exercise(Final)
