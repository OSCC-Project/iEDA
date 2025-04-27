# evaluation 功能文档

## 一、概述

evaluation支持对物理设计各个阶段（floorplanning，placement，CTS，route）的各类指标（wirelength、density、congestion、timing、power）进行多种模型的评估。evaluation整体采用接口(api)与实现(src)分离、评估指标模块(src/module)与第三方库(src/util)分离的设计方式，层次结构较为合理，工具之间耦合性较低，可扩展性较好。未来evaluation会结合需求持续丰富评估功能，关于evaluation的改进建议、需求等，欢迎联系：qiuyihang23@mails.ucas.ac.cn。


## 二、接口

evaluation/api文件夹提供外部使用的接口，根据指标进行分类，分别有Wirelength、Density、Congestion、Timing(Power)的API。API设计时主要使用了结构体返回同一类指标的多种模型的评估结果。有些指标在某些阶段并无意义，用户有责任自行判断和使用指标。

### Wirelength API

支持多种模型：HPWL，FLUTE，HTree，VTree，GRWL的计算，支持多种粒度：设计级、线网级、路径级线长的计算。
- HPWL：半周长线长
- FLUTE：斯坦纳树线长
- HTree：水平树线长
- VTree：纵向树线长
- GRWL：早期布线线长

|  接口名   |                   输入参数                  |                     返回类型                     |    底层逻辑  |
| :-----: | :--------------------------------------: | :------------------------------------------: | :------------------------------------------: |
|  `totalWL()` |          —          |      `struct{HPWL,FLUTE,HTtree,VTree,GRWL}`           | 从iDB获取数据，构建评估器数据，计算各类模型的总线长    |
|  `totalWLPure()` |          —          |      `struct{HPWL,FLUTE,HTtree,VTree,GRWL}`           | 直接获取评估器数据，计算各类模型的总线长    |
|  `netWL(string)` |          线网名称          |      `struct{HPWL,FLUTE,HTtree,VTree}`           | 从iDB获取数据，构建评估器数据，计算各类模型的线网线长    |
|  `totalWL(PointSets)` |          所有线网的点集          |      `struct{HPWL,FLUTE,HTtree,VTree}`           | 根据点集计算各类模型的总线长    |
|  `netWL(PointSet)` |          单个线网的点集          |      `struct{HPWL,FLUTE,HTtree,VTree}`           | 根据点集计算各类模型的线网线长    |
|  `pathWL(PointSet, PointPair)` |         单个线网的点集，路径起点和终点          |      `struct{HPWL,FLUTE,HTtree,VTree}`           | 根据点集和点对计算各类模型的路径长度    |
|  `totalEGRWL(string)` |         guide文件路径          |      `GRWL`           | 解析iRT-egr生成的guide文件计算总线长    |
|  `netEGRWL(string, string)` |         guide文件路径, 线网名称          |      `GRWL`           | 解析iRT-egr生成的guide文件计算指定线网的线长    |
|  `pathEGRWL(string,string,string)` |         guide文件路径，线网名称，负载pin名称          |      `GRWL`           | 解析iRT-egr生成的guide文件计算指定线网的driver pin到load pin的路径长度    |
|  `evalNetInfo()` |         —          |      —           | 从iDB获取数据，构建评估器数据，生成线网名称与线长模型（HPWL、FLUTE、GRWL）映射的map在内存中   |
|  `evalNetInfoPure()` |         —          |      —           | 直接获取评估器数据，生成线网名称与线长模型（HPWL、FLUTE、GRWL）映射的map在内存中   |
|  `evalNetFlute()` |         —          |      —           | 从iDB获取数据，构建评估器数据，生成线网名称与线长模型（FLUTE）映射的map在内存中   |
|  `findNetHPWL(string)` |         线网名称         |     `HPWL`           | 根据线网名称与线长模型映射的map查找HPWL值   |
|  `findNetFLUTE(string)` |         线网名称         |     `FLUTE`           | 根据线网名称与线长模型映射的map查找FLUTE值   |
|  `findNetGRWL(string)` |         线网名称         |     `GRWL`           | 根据线网名称与线长模型映射的map查找GRWL值   |


### Density API

支持多种密度类型：单元密度，引脚密度，线网密度，宏单元间隙密度的计算，支持多种粒度：版图级、网格级的计算。密度计算对象是每个网格，即需要将版图进行划分。默认网格划分大小（长宽）为`grid_size=1`，即一倍行高，可通过调节参数`grid_size`修改网格粒度。
- 单元密度：包含宏单元、标准单元、所有单元（宏单元+标准单元）密度的计算，计算方式为： (与网格重叠的所有单元面积) / 网格面积 
- 引脚密度：包含宏单元引脚、标准单元引脚、所有单元（宏单元+标准单元）引脚密度的计算，计算方式为：每个网格内的引脚个数 
- 线网密度：包含全局线网、局部线网、所有线网（全局+局部）密度的计算。当线网的外接矩形仅在一个网格内，即为局部线网，否则为全局线网。计算方式为：线网的外接矩形覆盖当前网格的线网个数
- 宏单元间隙密度：包含水平、竖直、所有方向（水平+竖直）间隙密度的计算。宏单元间隙密度（macro margin）计算方式为：以当前网格为中心，水平/竖直方向（同时往左往右/往上往下）直到被宏单元或版图边界遮挡的最大连续距离。

|  接口名   |                   输入参数                  |                     返回类型                     |    底层逻辑  |
| :-----: | :--------------------------------------: | :------------------------------------------: | :------------------------------------------: |
|  `densityMap(string, int, bool)` |     设计阶段名称(用于文件名)，网格粒度(行高倍数)，当前网格是否累加周围一圈网格数值              |      `struct{cell_density, pin_density, net_density}`           | 从iDB获取数据，构建评估器数据，计算各类版图级密度，输出二维图    |
|  `densityMapPure(string, int, bool)` |          设计阶段名称(用于文件名)，网格粒度(行高倍数)，当前网格是否累加周围一圈网格数值          |      `struct{cell_density, pin_density, net_density}`           | 直接获取评估器数据，计算各类版图级密度，输出二维图    |
|  `cellDensityMap(string, int)` |          设计阶段名称(用于文件名)，网格粒度(行高倍数)          |      `struct{macro_density,stdcell_density,allcell_density}`           | 从iDB获取数据，构建评估器数据，计算各类版图级单元密度，输出二维图    |
|  `pinDensityMap(string, int, bool)` |     设计阶段名称(用于文件名)，网格粒度(行高倍数)，当前网格是否累加周围一圈网格数值               |      `struct{macro_pin,stdcell_pin,allcell_pin}`           |  从iDB获取数据，构建评估器数据，计算各类版图级引脚密度，输出二维图    |
|  `netDensityMap(string, int, bool)` |          设计阶段名称(用于文件名)，网格粒度(行高倍数)，当前网格是否累加周围一圈网格数值           |      `struct{local_net,global_net,allnet}`           |  从iDB获取数据，构建评估器数据，计算各类版图级线网密度，输出二维图     |
|  `cellDensityMap(DensityCells, DensityRegion, int, string )` |   单元列表，区域信息，网格粒度，设计阶段名称                 |      `struct{macro_density,stdcell_density,allcell_density}`            | 根据传入的数据结构直接计算各类版图级单元密度，输出二维图    |
|  `pinDensityMap(DensityPins, DensityRegion, int, string, bool )` |   引脚列表，区域信息，网格粒度，设计阶段名称，当前网格是否累加周围一圈网格数值                 |      `struct{macro_pin,stdcell_pin,allcell_pin}`            | 根据传入的数据结构直接计算各类版图级引脚密度，输出二维图    |
|  `netDensityMap(DensityNets, DensityRegion, int, string, bool )` |   线网列表，区域信息，网格粒度，设计阶段名称，当前网格是否累加周围一圈网格数值                 |      `struct{local_net,global_net,allnet}`            | 根据传入的数据结构直接计算各类版图级线网密度，输出二维图    |
|  `macroMarginMap(int)` |  网格粒度              |      `struct{horizontal, vertical, union}`            | 从iDB获取数据，构建评估器数据，计算版图级宏单元间隙，输出二维图    |
|  `macroMarginMap(DensityCells, DensityRegion, DensityRegion, int)` |  宏单元列表，Die信息，Core信息，网格粒度              |      `struct{horizontal, vertical, union}`            | 根据传入的数据结构直接计算版图级宏单元间隙，输出二维图    |
|  `patchPinDensity(patch_id_coord_map)` |  所有网格(patch)组成的`map<id, coord_pair>`               |      `map<patch_id, pin_density>`            | 根据传入的网格划分信息，利用iDB初始化版图数据，返回网格id及其对应的单元引脚密度    |
|  `patchCellDensity(patch_id_coord_map)` |  所有网格(patch)组成的`map<id, coord_pair>`               |      `map<patch_id, cell_density>`            | 根据传入的网格划分信息，利用iDB初始化版图数据，返回网格id及其对应的单元密度    |
|  `patchNetDensity(patch_id_coord_map)` |  所有网格(patch)组成的`map<id, coord_pair>`               |      `map<patch_id, net_density>`            | 根据传入的网格划分信息，利用iDB初始化版图数据，返回网格id及其对应的线网密度    |
|  `patchMacroMargin(patch_id_coord_map)` |  所有网格(patch)组成的`map<id, coord_pair>`               |      `map<patch_id, macro_margin>`            | 根据传入的网格划分信息，利用iDB初始化版图数据，返回网格id及其对应的宏单元间隙密度    |

### Congestion API

支持多种拥塞模型：early_global_routing(EGR)，RUDY(Rectangular Uniformwire DensitY)，LUT-RUDY，支持多种指标：overflow、utilization，支持多种粒度：水平方向、竖直方向、二维拥塞、三维拥塞的计算。拥塞计算对象是每个网格，即需要将版图进行划分。默认网格划分大小（长宽）为`grid_size=1`，即一倍行高，可通过调节参数`grid_size`修改网格粒度。
- EGR：调用早期布线器计算拥塞
- RUDY：一种基于线网密度来估计布线拥塞的概率法，所使用的线长模型是HPWL
- LUT-RUDY：RUDY的改进版本，增加以引脚个数、引脚分布、线网外接矩形长宽比三项指标来查找HPWL与RSMT的拟合系数，通过更加准确的线长模型提升线网密度估计的精度
- overflow：拥塞指标，定义为每个网格的demand - capacity
- utilizaiton：拥塞指标，定义为每个网格的demand / capacity
- TOF：total overflow，定义为同个网格经不同布线层累加后的overflow
- MOF：maximum overflow， 定义为同个网格不同布线层的最大overflow
- ACE：average congestion，对拥塞进行从大到小的排序，计算前x%的拥塞总和/网格个数的加权均值

|  接口名   |                   输入参数                  |                     返回类型                     |    底层逻辑  |
| :-----: | :--------------------------------------: | :------------------------------------------: | :------------------------------------------: |
|  `egrMap(string)` |     设计阶段名称(用于文件名)              |      `struct{horizontal, vertical, union}`           | 启动iRT进行早期全局布线，计算各个布线层的GCell拥塞，输出按照方向累加后的二维拥塞图路径    |
|  `egrMapPure(string)` |          设计阶段名称(用于文件名)       |      `struct{horizontal, vertical, union}`            | 根据已有的iRT结果，计算各个布线层的GCell拥塞，输出按照方向累加后的二维拥塞图路径    |
|  `egrOverflow(string)` |          设计阶段名称(用于文件名)       |      `struct{tof_horizontal, tof_vertical, tof_union, mof_horizontal, mof_vertical, mof_union, ace_horizontal, ace_vertical, ace_union}`            | 根据生成的EGR拥塞图，计算各类拥塞指标    |
|  `rudyMap(string, int)` |          设计阶段名称(用于文件名)，网格粒度       |      `struct{rudy_horizontal, rudy_vertical, rudy_union, lutrudy_horizontal, lutrudy_vertical, lutrudy_union}`            | 从iDB获取数据，构建评估器数据，输出RUDY/LUTRUDY二维拥塞图路径   |
|  `rudyMapPure(string, int)` |          设计阶段名称(用于文件名)，网格粒度       |      `struct{rudy_horizontal, rudy_vertical, rudy_union, lutrudy_horizontal, lutrudy_vertical, lutrudy_union}`            | 直接获取评估器数据，输出RUDY/LUTRUDY二维拥塞图路径   |
|  `rudyUtilization(string, bool)` |          设计阶段名称(用于文件名)，是否计算LUTRUDY数值       |      `struct{max_util_horizontal, max_util_vertical, max_util_union, ace_util_horizontal, ace_util_vertical, ace_util_union}`            | 根据生成的RUDY/LUTRUDY拥塞图，计算各类拥塞指标   |
|  `egrMap(string, string)` |          设计阶段名称(用于文件名)，iRT生成拥塞的文件目录       |      `struct{horizontal, vertical, union}`          | 启动iRT进行早期全局布线且结果保存到指定拥塞文件目录下，计算各个布线层的GCell拥塞，输出按照方向累加后的二维拥塞图路径   |
|  `egrMapPure(string, string)` |          设计阶段名称(用于文件名)，iRT生成拥塞的文件目录       |      `struct{horizontal, vertical, union}`          | 在指定的拥塞文件目录下查找iRT生成的拥塞文件，计算各个布线层的GCell拥塞，输出按照方向累加后的二维拥塞图路径   |
|  `egrOverflow(string, string)` |          设计阶段名称(用于文件名)，iRT生成拥塞的文件目录       |      `struct{tof_horizontal, tof_vertical, tof_union, mof_horizontal, mof_vertical, mof_union, ace_horizontal, ace_vertical, ace_union}`           | 在指定的拥塞文件目录下查找iRT生成的拥塞文件，计算各类拥塞指标   |
|  `rudyMap(string, CongestionNets, CongestionRegion, int)` |          设计阶段名称(用于文件名)，线网列表，区域信息，网格粒度       |        `struct{rudy_horizontal, rudy_vertical, rudy_union, lutrudy_horizontal, lutrudy_vertical, lutrudy_union}`             | 根据传入的数据结构，计算RUDY/LUTRUDY拥塞，返回二维图路径   |
|  `rudyUtilization(string, string, bool)` |          设计阶段名称(用于文件名)，RUDY二维图的文件目录，是否计算LUTRUDY数值       |        `struct{rudy_horizontal, rudy_vertical, rudy_union, lutrudy_horizontal, lutrudy_vertical, lutrudy_union}`             | 在指定的拥塞文件目录下查找生成的RUDY/LUTRUDY拥塞图，计算各类拥塞指标   |
|  `getEGRMap(bool)` |          是否启动iRT得到EGR拥塞图       |        `map<layer_name, congestion_matrix>`             | 返回布线层名字与拥塞矩阵的map，拥塞矩阵为每一层所有网格的overflow值   |
|  `patchRUDYCongestion(patch_id_coord_map)` |     所有网格(patch)组成的`map<id, coord_pair>`            |        `map<patch_id, RUDY>`             | 根据传入的网格划分信息，利用iDB初始化版图数据，返回网格id与其对应的RUDY拥塞值   |
|  `patchEGRCongestion(patch_id_coord_map)` |     所有网格(patch)组成的`map<id, coord_pair>`            |        `map<patch_id, EGR>`             | 根据传入的网格划分信息，利用iRT生成的拥塞图，返回网格id与其对应的EGR拥塞值   |
|  `patchLayerEGRCongestion(patch_id_coord_map)` |     所有网格(patch)组成的`map<id, coord_pair>`            |        `map<patch_id, map<layer_name,EGR>>`             | 根据传入的网格划分信息，利用iRT生成的拥塞图，返回网格id与其对应的每一层EGR拥塞值的map   |
|  `evalNetInfo()` |     —            |        —           | 从iDB获取数据，构建评估器数据，生成线网名称与各类线网属性值（引脚个数、引脚分布等）映射的map在内存中  |
|  `evalNetInfoPure()` |     —            |        —            | 直接获取评估器数据，生成线网名称与各类线网属性值（引脚个数、引脚分布等）映射的map在内存中  |
|  `findPinNumber(string)` |     线网名称           |       引脚个数             | 根据线网名称，查找引脚个数  |
|  `findAspectRatio(string)` |    线网名称            |        线网外接矩形长宽比            | 根据线网名称，查找线网外接矩形长宽比  |
|  `findLness(string)` |     线网名称            |        引脚分布           | 根据线网名称，查找引脚分布值  |
|  `findBBoxWidth(string)` |     线网名称           |       线网外接矩形宽度            | 根据线网名称，查找线网外接矩形宽度  |
|  `findBBoxHeight(string)` |     线网名称           |       线网外接矩形高度            | 根据线网名称，查找线网外接矩形高度  |
|  `findBBoxArea(string)` |     线网名称           |       线网外接矩形面积            | 根据线网名称，查找线网外接矩形面积  |
|  `findBBoxLx(string)` |     线网名称           |       线网外接矩形左下角x坐标            | 根据线网名称，查找线网外接矩形左下角x坐标  |
|  `findBBoxLy(string)` |     线网名称           |       线网外接矩形左下角y坐标            | 根据线网名称，查找线网外接矩形左下角y坐标  |
|  `findBBoxUx(string)` |     线网名称           |       线网外接矩形右上角x坐标            | 根据线网名称，查找线网外接矩形右上角x坐标  |
|  `findBBoxUy(string)` |     线网名称           |       线网外接矩形右上角y坐标            | 根据线网名称，查找线网外接矩形右上角y坐标  |
|  `egrUnionMap(string，string)` |   设计阶段名称(用于文件名)，iRT生成拥塞的文件目录          |       二维拥塞图路径            | 启动iRT进行早期全局布线且结果保存到指定拥塞文件目录下，计算各个布线层的GCell拥塞，输出按照所有方向累加后的二维拥塞图路径  |

### Timing(Power) API

支持通过五种不同的线长模型（HPWL，FLUTE，SALT，EGR，DR）完成RC Tree的构建，并进行静态时序分析和功耗分析。支持时序、功耗各项指标的计算，支持设计级、线网级、路径级的时序和功耗指标查询。值得注意的是，时序评估器尚未支持直接基于布线结果(route.def)构建RCTree进行时序和功耗分析，目前EGR和DR需要调用iRT分别执行完对应GR和DR的流程，较为耗时。
- 时序：setup_tns, setup_wns, hold_tns, hold_wns, suggest_freq
- 功耗：static_power, dynamic_power

|  接口名   |                   输入参数                  |                     返回类型                     |    底层逻辑  |
| :-----: | :--------------------------------------: | :------------------------------------------: | :------------------------------------------: |
|  `runSTA()` |    —              |      —           | 启动iSTA/iPW进行时序/功耗的评估，保存在对应线长模型与评估指标的map中，默认执行五种线长模型   |
|  `evalTiming(string, bool)` |    线长模型，是否在流程中执行过iRT            |      —           | 启动iSTA/iPW进行时序/功耗的评估，只评估指定线长模型对应的时序/功耗，保存在对应线长模型与评估指标的map中   |
|  `evalDesign()` |    —            | `struct{clock_name, setup_tns, setup_wns, hold_tns, hold_wns, suggest_freq, static_power, dynamic_power}`               | 从线长模型与评估指标的map中，查询并返回整个设计的时序和功耗信息，默认执行五种线长模型   |
|  `evalNetPower()` |    —            | `map<线长模型，<线网名称，switch power>>`               | 从线长模型与功耗指标的map中，查询并返回特定线长模型下每个线网对应的切换功耗   |
|  `getEarlySlack(string)` |    引脚名称            |        `early_slack`        | 计算引脚对应的early_slack值   |
|  `getLateSlack(string)` |    引脚名称            |        `late_slack`        | 计算引脚对应的late_slack值   |
|  `getArrivalEarlyTime(string)` |    引脚名称            |        `arrival_early_time`        | 计算引脚对应的arrival_early_time值   |
|  `getArrivalLateTime(string)` |    引脚名称            |        `arrival_late_time`        | 计算引脚对应的arrival_late_time值   |
|  `reportWNS(string, mode)` |    时钟名称，sta分析模式            |        `WNS`        | 计算时钟对应的WNS值   |
|  `reportTNS(string, mode)` |    时钟名称，sta分析模式            |        `WNS`        | 计算时钟对应的TNS值   |
|  `updateTiming(TimingNets, int)` |    线网列表，单位值            |        —        | 根据传入的线网数据结构和单位值，更新STA时序评估结果，输出评估报告   |
|  `updateTiming(TimingNets, strings, int, int)` |    线网列表，单元名称列表，传输级别，单位值            |        —        | 根据传入的数据结构，移动对应单元位置，更新STA时序评估结果  |
|  `isClockNet(string)` |    线网名称           |        `bool`        | 判断线网是否为时钟线网 |
|  `patchTimingMap(patch_id_coord_map)` |     所有网格(patch)组成的`map<id, coord_pair>`            |        `map<patch_id, min_slack>`             | 根据传入的网格划分信息，启动iSTA评估时序，返回网格id与其对应内部所有单元的min_slack   |
|  `patchPowerMap(patch_id_coord_map)` |     所有网格(patch)组成的`map<id, coord_pair>`            |        `map<patch_id, total_power>`             | 根据传入的网格划分信息，启动iPW评估功耗，返回网格id与其对应内部所有单元的total_power   |
|  `patchIRDropMap(patch_id_coord_map)` |     所有网格(patch)组成的`map<id, coord_pair>`            |        `map<patch_id, max_irdrop>`             | 根据传入的网格划分信息，启动iPW评估IRDrop，返回网格id与其对应内部所有单元的max_irdrop  |


## 三、AIEDA的特征接口

在iEDA/src/feature/feature_manager中，对evaluation的接口进一步进行封装，输出`place`和`CTS`两个阶段的汇总性评估结果：
- 设计级：wirelength、density、congestion、timing、power，保存在jsonl中；
- 线网级：各类线网级别的指标，如引脚个数、引脚分布、斯坦纳树线长、功耗等，保存在csv文件中

注意：目前暂不支持基于`route`阶段的汇总性评估，有需要可基于上述两个接口进行扩展。

|  接口名   |                   输入参数                  |                     返回类型                     |    底层逻辑  |
| :-----: | :--------------------------------------: | :------------------------------------------: | :------------------------------------------: |
|  `save_pl_eval_union(string, string, int)` |    Jsonl文件保存路径，CSV文件保存路径，网格粒度              |      `bool`           |   分别保存`Place`阶段汇总性特征到Jsonl文件（设计级）和CSV文件（线网级）  |
|  `save_cts_eval_union(string, string, int)` |    Jsonl文件保存路径，CSV文件保存路径，网格粒度            |      `bool`          | 分别保存`CTS`阶段汇总性特征到Jsonl文件（设计级）和CSV文件（线网级）   |

