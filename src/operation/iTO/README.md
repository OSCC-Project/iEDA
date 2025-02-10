# iTO 功能文档

> ## 概述

TO的全称为Timing Optimization，时序优化。在该步骤，EDA工具会根据时序约束文件，对芯片进行时序分析，目的是通过单元尺寸调整和插入缓冲器等方法尽可能地修复芯片存在的时序违例。

时序违例检查主要包括：
1. 时序设计规则违例（DRV）检查；
2. 建立时间违例（Setup）检查；
3. 保持时间违例（Hold）检查。

支持功能：
1. 时序设计规则违例（DRV）优化；
2. 建立时间违例（Setup）优化；
3. 保持时间违例（Hold）优化。

iTO提供了4个Tcl命令：
1. `run_to`：由用户在config文件中任意指定需要优化的步骤；
2. `run_to_drv`：执行DRV优化；
2. `run_to_hold`：执行Hold优化；
2. `run_to_setup`：执行Setup优化。

> ## iTO使用示例

部分Config说明
```
"routing_tree": "flute",  // 连接线网所有引脚的topology，主要用于RC树构建，DRV优化，以及Setup优化。可选flute：flute构建的rsmt、hvtree：HV tree、shallow-light：通过SALT构建的shallow-light tree
"setup_target_slack": 0.0, // setup slack小于该值时认为违例，也是slack优化的目标
"hold_target_slack": 0.4,  // hold slack小于该值时认为违例，也是slack优化的目标
"max_insert_instance_percent": 0.2,  // 缓冲器插入的面积占芯片面积的最大比例
"max_core_utilization": 0.8,  // 缓冲器插入后的面积+其他单元的面积，占芯片面积的最大比例

"DRV_insert_buffers": [
    ""  // 优化DRV使用的缓冲器
],
"setup_insert_buffers": [
    ""  // 优化setup使用的缓冲器
],
"hold_insert_buffers": [
    ""  // 优化hold使用的缓冲器
],
"number_of_decreasing_slack_iter": 5,  // 迭代优化setup时，允许WNS不断变差的最大连续迭代次数
"max_allowed_buffering_fanout": 20,  // 针对setup，线网的fanout超过该值时不会对其进行缓冲器插入优化
"min_divide_fanout": 8  // 针对setup，线网的fanout大于该值时通过插入缓冲器把fanout降低
"optimize_endpoints_percent": 1.0 //针对setup，需要优化的违例端点占全部违例端点的比例 
"drv_optimize_iter_number": 5  // 针对drv，drv优化的执行次数

```

iTO可以独立执行某个优化步骤，也可以任意指定需要优化的步骤。
需要执行哪个步骤，可将iTO的Config文件中对应的步骤设置为True
```
"optimize_drv": false,
"optimize_hold": false,
"optimize_setup": false,
```

下面以执行DRV优化为例：

1.在Tcl文件中设置Config文件

`run_to_drv -config ./iEDA_config/to_default_config_drv.json`


2.使用iEDA运行tcl文件

`./iEDA -script ./script/iTO_script/run_iTO_drv.tcl`

### 报告输出

在Config文件中可设置优化结果的报告输出路径：
```
"report_file": "path"
```

DRV优化报告示例：

```
Found 0 slew violations.
Found 0 capacitance violations.
Found 0 fanout violations.
Found 0 long wires.
Before ViolationFix | slew_vio: 0 cap_vio: 0 fanout_vio: 0 length_vio: 0    \\ 优化前违例情况
The 1th check
After ViolationFix | slew_vio: 0 cap_vio: 0 fanout_vio: 0 length_vio: 0 \\ 优化后违例情况
Inserted 0 buffers in 0 nets.
Resized 0 instances.
```

Hold优化报告示例：

```
// 优化前Hold违例情况。
---------------------------------------------------------------------------
Clock Group                                    Hold TNS            Hold WNS
---------------------------------------------------------------------------
core_clock                                            0                   0
---------------------------------------------------------------------------

Worst Hold Path Launch : dpath/a_reg/_145_:CLK
Worst Hold Path Capture: dpath/a_reg/_145_:CLK


Finish hold optimization!
Total inserted 0 hold buffers and 0 load buffers.

// 优化后Hold违例情况。
---------------------------------------------------------------------------
Clock Group                                    Hold TNS            Hold WNS
---------------------------------------------------------------------------
core_clock                                            0                   0
---------------------------------------------------------------------------
```

Setup优化报告示例：
```
-0.304023 -0.204023    // setup优化过程中的WNS变化情况
Inserted 10 buffers.
Resized 10 instances.
Unable to repair all setup violations.
```