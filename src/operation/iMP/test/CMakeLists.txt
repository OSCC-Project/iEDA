add_executable(test_data_manager ${IMP_TEST}/Test_DataManager.cc)
target_link_libraries(test_data_manager 
   PRIVATE
   imp_source
   )