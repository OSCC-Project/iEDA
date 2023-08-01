add_library(imp_api
    ${IMP_API}/MPAPI.cc
)

# target_link_libraries(imp_api
#     PRIVATE
#         # imp_source
#         # imp_api_external_libs
# )

target_include_directories(imp_api
    PUBLIC
        ${IMP_API}   
)
