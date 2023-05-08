set work_dir ../src/operation/iPW/source/data/example2 

set_design_workspace $work_dir/rpt

read_netlist $work_dir/gcd.v

set LIB_FILES      "\
      $work_dir/asap7sc7p5t_SIMPLE_RVT_FF_nldm_201020.lib \
      $work_dir/asap7sc7p5t_SEQ_RVT_FF_nldm_201020.lib \
      $work_dir/asap7sc7p5t_OA_RVT_FF_nldm_201020.lib \
      $work_dir/asap7sc7p5t_INVBUF_RVT_FF_nldm_201020.lib \
      $work_dir/asap7sc7p5t_AO_RVT_FF_nldm_201020.lib"


read_liberty $LIB_FILES

link_design gcd 

read_sdc  $work_dir/gcd.sdc
#read_spef $work_dir/tapout/spf/asic_top.rcworst.125c.spef

report_timing

report_power 
