# iSTA用户指南

> ## iSTA简介

### 软件结构图

<div align="center">

<img src="../../../docs/resources/iSTA.png" width="80%" height="35%" alt="iSTA-logo" />

  **iSTA--一款用于集成电路设计的开源智能静态时序分析工具**

</div>

### 支持功能

- 完善地支持标准输入文件（Def/Verilog，sdc，spef/sdf, liberty）读取；
- 延时计算除了支持NLDM/Elmore计算模型，还支持CCS电流模型，Arnoldi降阶模型；
- 时序分析支持Clock Gate分析，Removal/Recovery分析和Muliticycle分析；
- 时序路径分析模式支持OCV模式和AOCV模式；
- 噪声分析初步支持了Crosstalk的影响，未来将进一步完善；
- 提供时序分析引擎timing engine供物理设计调用。

---

> ## iSTA使用示例

### 编写tcl文件(run_ista.tcl)

示例tcl文件位于: /src/operation/iSTA/source/data/example1/run_ista.tcl

#### 设置时序报告输出路径

```bash
set work_dir "../src/operation/iSTA/source/data/example1"
set_design_workspace $work_dir/rpt
```

#### 读取verilog文件

```bash
read_netlist $work_dir/example1.v
```

#### 读取.lib文件

```bash
set LIB_FILES $work_dir/example1_slow.lib
read_liberty $LIB_FILES
```

#### 链接设计到网表

```bash
link_design top
```

#### 读取sdc文件

```bash
read_sdc  $work_dir/example1.sdc
```

#### 读取spef文件

```bash
read_spef $work_dir/example1.spef
```

#### 获取时序报告

```bash
report_timing
```

时序报告位于第一步设置的时序报告输出路径下，包括

- top.rpt（报告 WNS,TNS 和时序路径）
- top.cap（报告违例电容）
- top.fanout（报告违例扇出）
- top.trans（报告违例转换时间）
- top_hold.skew（报告hold模式下的时钟偏斜)
- top_setup.skew（报告setup模式下的时钟偏斜）

### 编译iSTA（iSTA位于:bin/）

### 使用iSTA运行tcl文件

```bash
 cd bin/
 ./iSTA run_ista.tcl
```
