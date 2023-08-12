
add_dependencies(doc
   gcg_doc)

add_custom_target(gcg_doc
   COMMAND ${CMAKE_COMMAND} -E env "BINDIR=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" ${CMAKE_CURRENT_LIST_DIR}/builddoc.sh --mathjax
   WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/
   DEPENDS gcg)
