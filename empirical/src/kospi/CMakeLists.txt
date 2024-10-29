cmake_minimum_required(VERSION 3.15)
project(JsonParser)

# Create the executable
add_executable(json_parser main.cpp)

# Link the libraries from Conan
find_package(nlohmann_json REQUIRED)
target_link_libraries(json_parser nlohmann_json::nlohmann_json)
