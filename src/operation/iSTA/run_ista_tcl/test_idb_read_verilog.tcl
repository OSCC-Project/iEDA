set work_dir /home/longshuaiying/test_ariane

set LEF_FILES    "\
      $work_dir/lef/NangateOpenCellLibrary.tech.lef \
      $work_dir/lef/fakeram45_128x116.lef \
      $work_dir/lef/fakeram45_256x16.lef \
      $work_dir/lef/fakeram45_256x64.lef \
      $work_dir/lef/fakeram45_64x124.lef \
      $work_dir/lef/fakeram45_64x64.lef \
      $work_dir/lef/fakeram45_128x256.lef \
      $work_dir/lef/fakeram45_256x32.lef \
      $work_dir/lef/fakeram45_32x32.lef \
      $work_dir/lef/fakeram45_64x256.lef \
      $work_dir/lef/NangateOpenCellLibrary.macro.mod.lef \
      $work_dir/lef/fakeram45_128x32.lef \
      $work_dir/lef/fakeram45_256x48.lef \
      $work_dir/lef/fakeram45_512x64.lef \
      $work_dir/lef/fakeram45_64x62.lef"
      
set VERILOG_FILE "/home/longshuaiying/test_ariane/ariane.v"
set top_module_name ariane
set OUTPUT_DEF_FILE "/home/longshuaiying/test_ariane/ariane.def"


verilog_to_def -lef $LEF_FILES -verilog $VERILOG_FILE -top $top_module_name -def $OUTPUT_DEF_FILE