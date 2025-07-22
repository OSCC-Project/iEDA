# iTO: Timing Optimization

## Overview

The full name of TO is Timing Optimization. In this step, the EDA tool performs timing analysis on the chip according to the timing constraint file. The purpose is to repair the timing violations of the chip as much as possible by methods such as cell sizing and buffer insertion.

The main checks for timing violations include:
1. Timing design rule violation (DRV) check;
2. Setup time violation check;
3. Hold time violation check.

Supported functions:
1. Timing design rule violation (DRV) optimization;
2. Setup time violation optimization;
3. Hold time violation optimization.

iTO provides 4 Tcl commands:
1. `run_to`: Users can arbitrarily specify the optimization steps needed in the config file;
2. `run_to_drv`: Perform DRV optimization;
2. `run_to_hold`: Perform Hold optimization;
2. `run_to_setup`: Perform Setup optimization.

## iTO Usage Example

Part of the Config description
```
"setup_slack_margin": 0.0, // When the setup slack is less than this value, it is considered a violation and is also the target of slack optimization
"hold_slack_margin": 0.4,  // When the hold slack is less than this value, it is considered a violation and is also the target of slack optimization
"max_buffer_percent": 0.2,  // The maximum proportion of the area occupied by buffer insertion to the chip area
"max_utilization": 0.8,  // The maximum proportion of the area after buffer insertion + the area of other cells to the chip area

"DRV_insert_buffers": [
    ""  // Buffers used for optimizing DRV
],
"setup_insert_buffers": [
    ""  // Buffers used for optimizing setup
],
"hold_insert_buffers": [
    ""  // Buffers used for optimizing hold
],
"number_passes_allowed_decreasing_slack": 5,  // When iteratively optimizing setup, the maximum consecutive number of iterations allowed for WNS to continuously deteriorate
"rebuffer_max_fanout": 20,  // For setup, when the fanout of a net exceeds this value, buffer insertion optimization will not be performed on it
"split_load_min_fanout": 8  // For setup, when the fanout of a net is greater than this value, the fanout is reduced by inserting buffers

```

iTO can perform a certain optimization step independently or arbitrarily specify the steps to be optimized.
To perform which step, set the corresponding step in the iTO Config file to True
```
"optimize_drv": false,
"optimize_hold": false,
"optimize_setup": false,
```

The following takes performing DRV optimization as an example:

1. Set the Config file in the Tcl file

`run_to_drv -config./iEDA_config/to_default_config_drv.json`


2. Use iEDA to run the tcl file

`./iEDA -script./script/iTO_script/run_iTO_drv.tcl`

### Report Output

The report output path of the optimization result can be set in the Config file:
```
"report_file": "path"
```

Example of DRV optimization report:

```
Found 0 slew violations.
Found 0 capacitance violations.
Found 0 fanout violations.
Found 0 long wires.
Before ViolationFix | slew_vio: 0 cap_vio: 0 fanout_vio: 0 length_vio: 0    \\ Violation situation before optimization
The 1th check
After ViolationFix | slew_vio: 0 cap_vio: 0 fanout_vio: 0 length_vio: 0 \\ Violation situation after optimization
Inserted 0 buffers in 0 nets.
Resized 0 instances.
```

Example of Hold optimization report:

```
// Hold violation situation before optimization.
---------------------------------------------------------------------------
Clock Group                                    Hold TNS            Hold WNS
---------------------------------------------------------------------------
core_clock                                            0                   0
---------------------------------------------------------------------------

Worst Hold Path Launch : dpath/a_reg/_145_:CLK
Worst Hold Path Capture: dpath/a_reg/_145_:CLK


Finish hold optimization!
Total inserted 0 hold buffers and 0 load buffers.

// Hold violation situation after optimization.
---------------------------------------------------------------------------
Clock Group                                    Hold TNS            Hold WNS
---------------------------------------------------------------------------
core_clock                                            0                   0
---------------------------------------------------------------------------
```

Example of Setup optimization report:
```
-0.304023 -0.204023    // WNS changes during setup optimization
Inserted 10 buffers.
Resized 10 instances.
Unable to repair all setup violations.
```