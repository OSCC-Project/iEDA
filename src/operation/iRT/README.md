# iRT 介绍

## 背景

<img src="../../../docs/resources/flow.png" width="100%" alt="iPower-structure" />

布线是继布局和时钟树综合之后的重要物理实施任务,其内容是将分布在芯片核内的模块,标准单元和输入输出接口单元按逻辑关系进行互连,并为满足各种约束条件进行优化。iRT是iEDA课题组针对布线阶段设计的一款布线器,其内部集成了全局布线和详细布线。

## 软件结构

<img src="../../../docs/resources/iRT.jpg" width="80%" alt="iPower-structure" />

### API：多种语言的iRT接口

<img src="../../../docs/resources/iRT_tcl.jpg" width="80%" alt="iPower-structure" />

### data manager：顶层数据管理器

### module：算法主要模块

- pin_accessor: 对所有pin分配access点,在port上找到可以接入的点

- space_router: 全局布线器,以GCell为单位,在三维网格上进行全局布线

- track_assigner: wire轨道分配,建模为布线问题进行轨道分配

- detailed_router: 详细布线器,以DRC驱动的,基于三维track网格的详细布线器

### solver：布线时可以使用的求解器

- flute: 以查找表的形式进行快速斯坦纳树生成

- A*: 三维路径搜索算法

### utility：工具模块

- logger: 日志模块

- monitor: 运行状态监视器

- report: 报告器

- util: 工具函数

- plotter: debug可视化模块

