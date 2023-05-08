CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

SET (CMAKE_CXX_STANDARD 20)

AUX_SOURCE_DIRECTORY(./ SRC)

ADD_LIBRARY(ista-engine ${SRC})

TARGET_LINK_LIBRARIES(ista-engine sta str sdc sdc-cmd tcl idb def_service lef_service log absl::flat_hash_map) 

target_include_directories(ista-engine 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)
