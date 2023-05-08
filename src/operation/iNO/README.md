# iNO 功能文档

> ## 概述

NO的全称为Netlist Optimization，网表优化。iNO当前支持扇出优化，通过插入缓冲器使线网满足最大扇出约束。

> ## iNO使用示例

用户需要在Config文件中指定最大扇出（fanout）约束、使用的缓冲器类型。例如
```
"insert_buffer": "LVTX_4",
"max_fanout": 30
```

1.在Tcl文件中设置Config文件

`run_no_fixfanout -config ./iEDA_config/no_default_config_fixfanout.json`

2.使用iEDA运行tcl文件

`./iEDA -script ./script/iNO_script/run_iNO_fix_fanout.tcl`

### 报告输出

在Config文件中可设置优化结果的报告输出路径：
```
"report_file": "path"
```

fanout优化报告输出示例：
```
[Result: ] Find 0 Net with fanout violation.
[Result: ] Insert 0 Buffers.
```