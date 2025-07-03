# iNO: Netlist Optimization


## Overview

NO stands for Netlist Optimization. Currently, iNO supports fanout optimization, which makes the net meet the maximum fanout constraint by inserting buffers.

## iNO Usage Example

Users need to specify the maximum fanout constraint and the type of buffer used in the Config file. For example,
```
"insert_buffer": "LVTX_4",
"max_fanout": 30
```

1. Set the Config file in the Tcl file

`run_no_fixfanout -config./iEDA_config/no_default_config_fixfanout.json`

2. Use iEDA to run the tcl file

`./iEDA -script./script/iNO_script/run_iNO_fix_fanout.tcl`

### Report Output

In the Config file, the report output path of the optimization result can be set:
```
"report_file": "path"
```

Example of fanout optimization report output:
```
[Result: ] Find 0 Net with fanout violation.
[Result: ] Insert 0 Buffers.
```