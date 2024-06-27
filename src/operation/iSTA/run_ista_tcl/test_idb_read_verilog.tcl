set work_dir /home/longshuaiying/test_idb_verilog

set LEF_FILES    "\
 /data/project_share/process_node/T28_lib/tech/tsmcn28_9lm6X2ZUTRDL.tlef \
        /data/project_share/process_node/T28_lib/lef//tcbn28hpcplusbwp30p140.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp30p140hvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp30p140lvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp30p140mb.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp30p140mblvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp30p140opp.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp30p140opphvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp30p140opplvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp30p140oppuhvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp30p140oppulvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp30p140uhvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp35p140.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp35p140hvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp35p140lvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp35p140mb.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp35p140mblvt.lef \
        /data/project_share/process_node/T28_lib/lef/lef/tcbn28hpcplusbwp35p140opp.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp35p140opphvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp35p140opplvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp35p140oppuhvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp35p140oppulvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp35p140uhvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp40p140.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp40p140hvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp40p140lvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp40p140mb.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp40p140mbhvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp40p140opp.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp40p140opphvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp40p140opplvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp40p140oppuhvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp40p140uhvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp35p140mbhvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp30p140ulvt.lef \
        /data/project_share/process_node/T28_lib/lef/tcbn28hpcplusbwp35p140ulvt.lef \
        /data/project_share/process_node/T28_lib/lef/ts5n28hpcplvta64x128m2f_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts5n28hpcplvta64x128m2fw_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts5n28hpcplvta256x32m4fw_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts5n28hpcplvta128x32m2f_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts5n28hpcplvta128x64m2f_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts5n28hpcplvta128x80m2f_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts5n28hpcplvta128x8m2f_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts5n28hpcplvta512x64m4f_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts5n28hpcplvta64x32m2f_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts5n28hpcplvta64x8m2f_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts5n28hpcplvta64x64m2f_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts5n28hpcplvta8x128m2f_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts6n28hpcplvta16x128m2f_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts5n28hpcplvta8x144m2f_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts6n28hpcplvta512x2m8f_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts5n28hpcplvta256x16m2f_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts5n28hpcplvta64x80m2f_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts5n28hpcplvta32x128m2f_130a.lef \
        /data/project_share/process_node/T28_lib/lef/ts5n28hpcplvta32x32m2f_130a.lef \
        /data/project_share/process_node/T28_lib/lef/tpbn28v.lef \
        /data/project_share/process_node/T28_lib/lef/tphn28hpcpgv18_9lm.lef \
        /data/project_share/process_node/T28_lib/lef/PLLTS28HPMLAINT.lef \
        /data/project_share/process_node/T28_lib/lef/tpbn28v_9lm.lef"
      
set VERILOG_FILE "/home/taosimin/T28/ieda_1208/asic_top_1208.syn.v"
set top_module_name asic_top
set OUTPUT_DEF_FILE "/home/longshuaiying/test_idb_verilog/asic_top_rust.v"


verilog_to_def -lef $LEF_FILES -verilog $VERILOG_FILE -top $top_module_name -def $OUTPUT_DEF_FILE


        