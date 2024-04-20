macro(ADD_EXTERNAL_PROJ proj_name)

include(ExternalProject)

ExternalProject_Add(
    ${RUST_PROJECT_NAME}
    PREFIX ${RUST_PROJECT_DIR}
    SOURCE_DIR ${RUST_PROJECT_DIR}
    BINARY_DIR ${RUST_PROJECT_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND cargo build ${RUST_BUILD_CMD_OPTION}
    INSTALL_COMMAND ""
    BUILD_ALWAYS 1
    BUILD_BYPRODUCTS ${RUST_LIB_PATH}
)

add_dependencies(${proj_name} ${RUST_PROJECT_NAME})

endmacro()

include_directories(${HOME_DATABASE}/manager/parser/rust-common)