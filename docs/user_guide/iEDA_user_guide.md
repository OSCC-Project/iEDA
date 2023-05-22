# iEDA 用户手册

## iEDA概述
### iEDA系统部署图
<div align=center> <img src="pic/system_deploy/deployment.png" style="zoom:70%;" /> </div>

## 工具准备
### 环境
- 服务器配置 
- 操作系统   Ubuntu 20.04.5 LTS
- 工艺环境   SkyWater PDK

### 编译构建
```bash
# 下载iEDA仓库
git clone https://gitee.com/oscc-project/iEDA.git iEDA && cd iEDA
# 通过apt安装编译依赖，需要root权限
sudo bash build.sh -i apt
# 编译 iEDA
bash build.sh -j 16
# 若能够正常输出 "Hello iEDA!" 则编译成功
./bin/iEDA -script scripts/hello.tcl
```
拷贝 ./bin/iEDA 到目录 ./scripts/sky130
```bash
# 拷贝 iEDA 到sky130 目录 
cp ./bin/iEDA scripts/sky130/.
```

### 工艺文件准备
下载SkyWater PDK  
拷贝TechLEF 文件 和 LEF文件到目录 ./scripts/sky130/lef
```bash
# 拷贝 TechLEF 文件到目录 ./scripts/sky130/lef
cp <skywater130pdk_tlef_path>/*.tlef scripts/sky130/lef/.
# 拷贝 LEF 文件到目录 ./scripts/sky130/lef
cp <skywater130pdk_lef_path>/*.lef scripts/sky130/lef/.
```
拷贝Lib文件到 ./scripts/sky130/lib
```bash
# 拷贝 Lib 文件到目录 ./scripts/sky130/lib
cp <skywater130pdk_lib_path>/*.lib scripts/sky130/lib/.
```
拷贝sdc文件到 ./scripts/sky130/sdc
```bash
# 拷贝 sdc 文件到目录 ./scripts/sky130/sdc
cp <skywater130pdk_sdc_path>/*.sdc scripts/sky130/sdc/.
```
### 设计文件准备
拷贝.v Netlist文件到目录 scripts/sky130/result/verilog
```bash
# 拷贝 .v 文件到目录 ./scripts/sky130/result/verilog
cp <skywater130pdk_verilog_path>/gcd.v scripts/sky130/result/verilog/.
```

## 工具流程
本文档以跑通skywater PDK 130nm工艺物理后端设计流程作为示例，说明iEDA各个点工具如何配置参数、运行和分析结果。
<div align=center> <img src="pic/flow/iEDA_flow.png" style="zoom:70%;" /> </div>

### 模块划分
```
scripts/sky130
├── iEDA_config   # iEDA parameters configuration files
├── lef           # lef files
├── lib           # lib files
├── result        # iEDA result output files
├── script        # Tcl script files
├── sdc           # sdc files
├── run_iEDA.py   # Python3 script for running all iEDA flow
└── run_iEDA.sh   # POSIX shell script for running all iEDA flow
```

#### script 模块说明
script目录包含物理后端设计需要的所有流程脚本和结果分析评估脚本，并且按流程、功能划分好模块；流程脚本可支持顶层自动化运行脚本run_iEDA.py的调用，也可以支持独立运行。
```
scripts/sky130/script
├── DB_script                           # Data process flow scripts
│   ├── db_init_lef.tcl                 # initialize lef
│   ├── db_init_lib_drv.tcl             # initialize lib only for flow of drv 
│   ├── db_init_lib_fixfanout.tcl       # initialize lib only for flow of fix fanout
│   ├── db_init_lib_hold.tcl            # initialize lib only for flow of optimize hold
│   ├── db_init_lib_setup.tcl           # initialize lib only for flow of optimize setup
│   ├── db_init_lib.tcl                 # initialize lib for common flow
│   ├── db_init_sdc.tcl                 # initialize sdc 
│   ├── db_init_spef.tcl                # initialize spef
│   ├── db_path_setting.tcl             # set paths for all processing technology files, including TechLEF，LEF, Lib, sdc and spef
│   ├── run_db_checknet.tcl             # check net connectivity based on data built by DEF (.def) and LEF (.lef & .tlef)
│   ├── run_db_report_evl.tcl           # report wire length and congestion based on data built by DEF (.def) and LEF (.lef & .tlef)
│   ├── run_db.tcl                      # test building data by DEF (.def) and LEF (.lef & .tlef)
│   ├── run_def_to_gds_text.tcl         # transform data from DEF (.def) to GDSII (.gdsii)
│   ├── run_def_to_verilog.tcl          # transform data from DEF (.def) to netlist (.v)
│   ├── run_netlist_to_def.tcl          # transform data from netlist (.v) to DEF (.def)
│   └── run_read_verilog.tcl            # test read verilog file (.v)
├── iCTS_script                         # CTS flow scripts
│   ├── run_iCTS_eval.tcl               # report wire legnth for CTS result
│   ├── run_iCTS_STA.tcl                # report CTS STA
│   └── run_iCTS.tcl                    # run CTS
├── iDRC_script                         # DRC(Design Rule Check) flow scipts
│   ├── run_iDRC_gui.tcl                # show GUI for DRC result
│   └── run_iDRC.tcl                    # run DRC
├── iFP_script                          # Floorplan flow scripts
│   ├── module                          # submodule for Floorplan scripts
│   │   ├── create_tracks.tcl           # create tracks for routing layers
│   │   ├── pdn.tcl                     # create pdn networks 
│   │   └── set_clocknet.tcl            # set clock net
│   └── run_iFP.tcl                     # run Floorplan
├── iGUI_script                         # GUI flow scipts
│   └── run_iGUI.tcl                    # run GUI
├── iNO_script                          # NO(Netlist Optimization) flow scipts
│   └── run_iNO_fix_fanout.tcl          # run Fix Fanout
├── iPL_script                          # Placement flow scripts
│   ├── run_iPL_eval.tcl                # report congestion statistics and wire legnth for Placement result
│   ├── run_iPL_filler.tcl              # run standard cell filler
│   ├── run_iPL_gui.tcl                 # run gui flow that shows Global Placement Processing result
│   ├── run_iPL_legalization_eval.tcl   # report congestion statistics and wire legnth for Legalization result
│   ├── run_iPL_legalization.tcl        # run Cell Legalization
│   └── run_iPL.tcl                     # run Placement
├── iRT_script                          # Routing flow scripts
│   ├── run_iRT_DRC.tcl                 # run DRC for Routing result
│   ├── run_iRT_eval.tcl                # report wire legnth for Routing result
│   ├── run_iRT_STA.tcl                 # run STA for Routing result
│   └── run_iRT.tcl                     # run Routing
├── iSTA_script                         # STA flow scripts
│   ├── init_iSTA.tcl                   # STA initialization
│   ├── report_iSTA.tcl                 # report STA result
│   └── run_iSTA.tcl                    # run STA
└── iTO_script                          # TO(Timing Optimization) flow script
    ├── run_iTO_drv_STA.tcl             # run STA for DRV result
    ├── run_iTO_drv.tcl                 # run DRV
    ├── run_iTO_hold_STA.tcl            # run STA for Fix Hold Violation result
    ├── run_iTO_hold.tcl                # run Fix Hold Violation
    ├── run_iTO_setup_STA.tcl           # run STA for Fix Setup Violation result
    └── run_iTO_setup.tcl               # run Fix Setup Violation
```

### 运行Flow

准备好iEDA和工艺文件后，您可以选择自动运行sky130流程脚本，也可以分步骤运行各个点工具脚本，所有的结果都默认保存在script/sky130/result文件夹

#### Flow基础流程
不管是自动运行顶层 run_iEDA.py 脚本还是单独运行点工具脚本，基于 iEDA 平台设计的脚本都有着相似的步骤，具体流程如下
##### step 1 路径设置
首先必须先配置工艺环境路径，为方便查找和配置路径参数，脚本将TechLEF、LEF、Lib、sdc、spef的路径统一在文件 ./script/DB_script/db_path_setting.tcl配置，如下表所示
|      功能       |     配置命令     |     参考 TCL 样例      |
|     :---        |     :---        |     :---        |
| 设置 TechLef 路径 | set TECH_LEF_PATH xxx | set TECH_LEF_PATH "./lef/sky130_fd_sc_hs.tlef" |
| 设置 Lef 路径 | set LEF_PATH xxx | set LEF_PATH ./lef/sky130_ef_io__com_bus_slice_10um.lef |
| 设置 Lib 路径 | set LIB_PATH xxx | set LIB_PATH ./lib/sky130_dummy_io.lib |
| 设置 Fix Fanout Lib 路径 | set LIB_PATH_FIXFANOUT xxx | set LIB_PATH_FIXFANOUT ./lib/sky130_dummy_io.lib |
| 设置 Fix DRV Violation Lib 路径  | set LIB_PATH_DRV xxx | set LIB_PATH_DRV ./lib/sky130_dummy_io.lib |
| 设置 Fix Hold Violation Lib 路径 | set LIB_PATH_HOLD xxx | set LIB_PATH_HOLD ./lib/sky130_dummy_io.lib |
| 设置 Fix Setup Violation Lib 路径 | set LIB_PATH_SETUP xxx | set LIB_PATH_SETUP ./lib/sky130_dummy_io.lib |
| 设置 SDC 路径 | set SDC_PATH xxx | set SDC_PATH "./sdc/gcd.sdc" |
| 设置 SPEF 路径 | #set SPEF_PATH xxx| set SPEF_PATH "./spef/xxx.spef" |

##### step 2 配置点工具Config
所有点工具的参数设置Config都在路径 ./iEDA_config 中，可查看后面章节的 **输入输出一览表** 修改对应的点工具Config文件

##### step 3 读 .def 设计文件
以 CTS 为例，执行 def_init 命令，读取布局后的结果
```bash
#===========================================================
##   read def
#===========================================================
def_init -path ./result/iPL_result.def
```
步骤 1 - 3 后，Tech LEF、LEF、DEF 文件数据将被加载，这是点工具启动的前提条件

##### step 4 启动点工具
以 CTS 为例，执行 run_cts 命令，将启动 CTS 流程
```bash
#===========================================================
##   run CTS
#===========================================================
run_cts -config ./iEDA_config/cts_default_config.json
```
各个点工具运行的TCL命令，请参看 **点工具运行** 模块

##### step 5 保存点工具运行结果
以 CTS 为例，执行完点工具流程后，将点工具运行结果保存在路径 ./result/ 中
```bash
#===========================================================
##   Save def
#===========================================================
def_save -path ./result/iCTS_result.def

#===========================================================
##   Save netlist 
#===========================================================
netlist_save -path ./result/iCTS_result.v -exclude_cell_names {}
```

##### step 6 输出报告
以 CTS 为例，数据存储后，将输出设计结果相关的总体报告，报告路径存储在 ./result/report/ 中
```bash
#===========================================================
##   report 
#===========================================================
report_db -path "./result/report/cts_db.rpt"
```

##### step 7 退出
```bash
#===========================================================
##   Exit 
#===========================================================
flow_exit
```
以上步骤为执行单个点工具的一般流程，其中步骤 1 - 3 初始化配置和数据库，为必须的步骤，步骤 4 之后，可以按照需求灵活接入各个点工具或模块命令

#### 报告分析
点工具运行完成后，分析报告将存储在路径 ./result/report 中，模块划分如下表所示
|      报告类型       |     路径     |     说明      |
|     :---        |     :---        |     :---        |
| Tech LEF、LEF、DEF数据报告 | ./result/report | 分析、统计 Design 文件的数据，并对PR过程单元、线网数据进行详细报告 |
| 线长、拥塞评估报告 | ./result/report/eval | 分析、统计点工具输出结果的线长、单元密度等数据 |
| DRC报告 | ./result/report/drc | 主要检测布线后的DRC违例情况，已支持GUI可视化分析 |

##### 基础信息
以 CTS 后的数据报告为例，各标签含义如下表

|      标签       |     样例值     |     说明      |
|     :---        |     :---        |     :---        |
| iEDA | V23.03-OS-01 | iEDA 当前版本号 |
| Stage | iCTS - Clock Tree Synthesis | 当前结果的流程阶段，iCTS表示当前结果由 CTS 输出 |
| Runtime | 2.863340 s | 当前点工具读入数据到保存结果所需运行时间 |
| Memmory | 5745.216000 MB | 当前点工具读入数据到保存结果所需最大峰值内存 |
| Design Name | gcd | 设计名称 |
| DEF&LEF Version | 5.8 | 当前设计的工艺文件版本号 |
| DBU | 1000 | 1 微米含单位长度个数，用于转换 DEF 和 Tech LEF 参数值(DATABASE MICRONS LEFconvertFactor) |

##### Design 数据报告
**Summary**  <br>

**Summary - Instance** <br>

**Summary - Net** <br>

**Summary - Layer** <br>

**Summary - Pin Distribution** <br>

##### 线长、拥塞评估报告
**Congestion Report**

**Wire Length Report**

##### DRC违例报告
**Drc Summary**

**Connectivity Summary**

**DRC - Disconnected Net**

#### 物理后端设计全流程运行
运行sky130目录的run_iEDA.py，将自动运行从读取.v Netlist文件到最后吐出.gdsii GDSII文件的全流程，全流程使用默认参数，所有运行结果将保存在scripts/sky130/result目录下，详细的功能描述、参数配置、输入、输出和报告等可查看点工具运行。
##### 运行脚本
```bash
# 切换目录到sky130
cd <sky130 path>
# 运行自动化流程脚本
./run_iEDA.py
```
##### 输入输出一览表
全流程自动运行，前后流程的输入、输出已在脚本中配置好且存在先后依赖关系，如下表所示

|      Flow       |     Script      |     Config      |  Design Input  |    Design Output     |     Report      |
|     :---        |     :---        |     :---        |     :----      |        :----         |     :----       |
| 布图规划 (Floorpan) | ./iEDA -script ./script/iFP_script/run_iFP.tcl | | ./result/verilog/gcd.v | ./result/iFP_result.def <br> ./result/iFP_result.v | ./result/report/fp_db.rpt |
| 网表优化（Fix Fanout） | ./iEDA -script ./script/iNO_script/run_iNO_fix_fanout.tcl | ./iEDA_config/cts_default_config.json  | ./result/iFP_result.def | ./result/iTO_fix_fanout_result.def <br> ./result/iTO_fix_fanout_result.v | ./result/report/fixfanout_db.rpt| 
| 布局 (Placement)  | ./iEDA -script ./script/iPL_script/run_iPL.tcl | ./iEDA_config/pl_default_config.json | ./result/iTO_fix_fanout_result.def | ./result/iPL_result.def <br> ./result/iPL_result.v | ./result/report/pl_db.rpt |
| 布局结果评估 (评估线长和拥塞)  | ./iEDA -script ./script/iPL_script/run_iPL_eval.tcl |  | ./result/iPL_result.def |  | ./result/report/eval/iPL_result_wirelength.rpt <br> ./result/report/eval/iPL_result_congestion.rpt | 
| 时钟树综合 (CTS) | ./iEDA -script ./script/iCTS_script/run_iCTS.tcl | ./iEDA_config/cts_default_config.json | ./result/iPL_result.def | ./result/iCTS_result.def <br> ./result/iCTS_result.v | ./result/report/cts_db.rpt |
| 时钟树综合结果评估 (评估线长)  | ./iEDA -script ./script/iCTS_script/run_iCTS_eval.tcl |  | ./result/iCTS_result.def |  | ./result/report/eval/iCTS_result_wirelength.rpt |
| 时钟树综合时序评估 (评估时序)  | ./iEDA -script ./script/iCTS_script/run_iCTS_STA.tcl |  | ./result/iCTS_result.def |  | ./result/cts/sta/ |
| 修复DRV违例 (Fix DRV Violation) | ./iEDA -script ./script/iTO_script/run_iTO_drv.tcl | ./iEDA_config/to_default_config_drv.json | ./result/iCTS_result.def | ./result/iTO_drv_result.def <br> ./result/iTO_drv_result.v | ./result/report/drv_db.rpt |
| Fix DRV结果评估 (评估时序) | ./iEDA -script ./script/iTO_script/run_iTO_drv_STA.tcl |  | ./result/iTO_drv_result.def |  | ./result/to/drv/sta/ |
| 修复Hold违例（Fix Hold Violation） | ./iEDA -script ./script/iTO_script/run_iTO_hold.tcl | ./iEDA_config/to_default_config_hold.json | ./result/iTO_drv_result.def | ./result/iTO_hold_result.def <br> ./result/iTO_hold_result.v | ./result/report/hold_db.rpt |
| Fix Hold结果评估（评估时序） | ./iEDA -script ./script/iTO_script/run_iTO_hold_STA.tcl |  | ./result/iTO_hold_result.def |  | ./result/to/hold/sta/ |
| 单元合法化（Legalization） | ./iEDA -script ./script/iPL_script/run_iPL_legalization.tcl | ./result/iTO_hold_result.def | ./result/iPL_lg_result.def <br> ./result/iPL_lg_result.v | ./result/report/lg_db.rpt |
| 合法化结果评估（评估线长和拥塞） | ./iEDA -script ./script/iPL_script/run_iPL_legalization_eval.tcl | ./result/iPL_lg_result.def |  | ./result/report/eval/iPL_lg_result_wirelength.rpt <br> ./result/report/eval/iPL_lg_result_congestion.rpt |
| 布线 （Routing） | ./iEDA -script ./script/iRT_script/run_iRT.tcl |  | ./result/iPL_lg_result.def | ./result/iRT_result.def ./result/iRT_result.v | ./result/report/rt_db.rpt |
| 布线结果评估（评估线长） | ./iEDA -script ./script/iRT_script/run_iRT_eval.tcl |  | ./result/iRT_result.def |  | ./result/report/eval/iRT_result_wirelength.rpt |
| 布线结果评估 （评估时序） | ./iEDA -script ./script/iRT_script/run_iRT_STA.tcl |  | ./result/iRT_result.def |  | ./result/rt/sta/ |
| 布线结果DRC | ./iEDA -script ./script/iRT_script/run_iRT_DRC.tcl |  | ./result/iRT_result.def |  | ./result/report/drc/iRT_drc.rpt |
| 单元填充 （Filler） | ./iEDA -script ./script/iPL_script/run_iPL_filler.tcl | ./iEDA_config/pl_default_config.json | ./result/iRT_result.def | ./result/iPL_filler_result.def <br> ./result/iPL_filler_result.v | ./result/report/filler_db.rpt |
| DEF转GDSII | ./iEDA -script ./script/DB_script/run_def_to_gds_text.tcl |  | ./result/iPL_filler_result.def | ./result/final_design.gds2 | |

#### 点工具运行 - 布图规划 (Floorpan)
##### 执行脚本
```bash
./iEDA -script ./script/iFP_script/run_iFP.tcl 
```
##### 参数配置
无

##### 输入
- ./result/verilog/gcd.v  

##### 输出
- ./result/iFP_result.def

##### 评测和报告
- ./result/report/fp_db.rpt

##### GUI
step 1：修改脚本 ./script/iGUI_script/run_iGUI.tcl 的输入设计 def 为 ./result/iFP_result.def
```
#===========================================================
##   read def
#===========================================================
def_init -path ./result/iFP_result.def
```
step 2: 执行iEDA GUI脚本
```bash
./iEDA_gui -script ./script/iGUI_script/run_iGUI.tcl 
```
step 3: 查看GUI  

**初始版图**

<div align=center> <img src="pic/gui/gui_floorplan.png" style="zoom:27%;" /> </div>

**PDN**
<div align=center> <img src="pic/gui/gui_floorplan_pdn.png" style="zoom:70%;" /> </div>

#### 点工具运行 - 网表优化（Fix Fanout）
##### 执行脚本
```bash
./iEDA -script ./script/iNO_script/run_iNO_fix_fanout.tcl 
```
##### 参数配置

##### 输入

##### 输出

##### 评测和报告

##### GUI
step 1：修改脚本 ./script/iGUI_script/run_iGUI.tcl 的输入设计 def 为 ./result/iTO_fix_fanout_result.def
```
#===========================================================
##   read def
#===========================================================
def_init -path ./result/iTO_fix_fanout_result.def
```
step 2: 执行iEDA GUI脚本
```bash
./iEDA_gui -script ./script/iGUI_script/run_iGUI.tcl 
```
step 3: 查看GUI
<div align=center> <img src="pic/gui/gui_fixfanout.png" style="zoom:27%;" /> </div>

## GUI操作手册
### 运行GUI
#### 编译构建
step 1: 修改可编译选项
切换到iEDA工程目录
```bash
# 切换到工程目录 iEDA
cd iEDA
```
设置顶层 CMakelist.txt 的BUILD_GUI为 ON
```bash
# 设置顶层 CMakelist.txt 的BUILD_GUI为 ON
option(BUILD_GUI "If ON, build GUI." ON)
```
step 2: 编译构建
```bash
# 通过apt安装编译依赖，需要root权限
sudo bash build.sh -i apt
# 编译 iEDA
bash build.sh -j 16
```
step 3: 拷贝副本为 iEDA_gui
```bash
# 拷贝 iEDA 到sky130 目录 
cp ./bin/iEDA scripts/sky130/iEDA_gui
```
#### 配置设计文件
修改脚本 ./script/iGUI_script/run_iGUI.tcl 的输入设计 def 为查看的设计文件， 比如修改为 ./result/iFP_result.def
```
#===========================================================
##   read def
#===========================================================
def_init -path ./result/iFP_result.def
```

#### 运行GUI
执行iEDA GUI脚本
```bash
./iEDA_gui -script ./script/iGUI_script/run_iGUI.tcl 
```

#### 可视化
下图为读取 ./result/iFP_result.def 设计文件的可视化结果
<div align=center> <img src="pic/gui/gui_floorplan.png" style="zoom:70%;" /> </div>

### GUI操作
#### TCL命令

#### GUI操作

## TCL命令手册

