# FP_TCL命令使用手册

# Dependencies

- sudo apt-get install libunwind-dev
- sudo apt-get install libspdlog-dev(env : ubuntu20.04)
- sudo apt-get install boost 1.71
- sudo apt-get install eigen

# Build

git clone --recursive 
cd iEDA
mkdir build
cd build
cmake ..
make

# Run

cd iEDA/bin

./FP gtest.tcl

# 工具更新

cd iEDA

git pull

rm -rf build

mkdir build

cd build

cmake ..

make

## 1.init_floorplan

- -die_area   die的面积，一个字符串，以空格区分每个值，此处的值为没有乘DBU的
- -core_area   core的面积，一个字符串，以空格区分每个值，此处的值为没有乘DBU的
- -core_site  core的site选取
- -io_site  针对110工艺，此处为**可选参数**，会默认选择“IOSite”

### 示例：

set DIE_AREA "0.0    0.0   2843    2843"

set CORE_AREA "150 150 2693 2693"

set PLACE_SITE HD_CoreSite

set IO_SITE IOSite

init_floorplan \

​	-die_area $DIE_AREA \

​	-core_area $CORE_AREA \

​	-core_site $PLACE_SITE \

​	-io_site $IO_SITE

## 2.placeInst

已测试的功能包括：

1. 摆放IOCELL
2. 摆放电源CELL
3. 摆放IOFILLER
4. 摆放CORNER

在摆放以上四种时，会进行摆放检查，规则是是否按照IOSite摆放以及是否在DIE BOUNDARY上

   5.摆放标准单元

- -inst_name  instance的名字
- -llx  左下角横坐标，此时的数值是绝对坐标，即需设置乘过DBU的值
- -lly  左下角纵坐标
- -orient  朝向，可使用（N，S，W，E）或（R0，R180，R90，R270）
- -cellmaster  cell的种类

### 示例：

placeInst \
   -inst_name u0_clk \
   -llx  0 \
   -lly 2510940 \
   -orient E \
   -cellmaster PX1W

## 3.placePort

该命令只针对与IOCELL连接的IOPIN的port生成，电源CELL的port不使用该接口

-  -pin_name   iopin名字
​-  -offset_x 相对于所连接的IOCELL的左下角坐标的偏移量，此处需设置绝对长度，即乘过DBU之后的值
​-  -offset_y 相对于所连接的IOCELL的左下角坐标的偏移量，此处需设置绝对长度，即乘过DBU之后的值
​-  -width 矩形宽度，相对于所连接的IOCELL的左下角坐标的偏移量，此处需设置绝对长度，即乘过DBU之后的值
​-  -height 矩形高度，相对于所连接的IOCELL的左下角坐标的偏移量，此处需设置绝对长度，即乘过DBU之后的值
​-  -layer 所在的层的名字

### 示例：

placePort \
    -pin_name osc_in_pad \
    -offset_x 9000 \
    -offset_y 71500 \
    -width 58000 \
    -height 58000 \
    -layer ALPA

## 4.placeIoFiller

摆放IOFiller，支持四个边自动填充

必选参数：-filler_types  IOFiller的种类

可选参数：

- -prefix 生成的filler的名字的前缀，默认为IOFill
- -edge  设定在哪一个边填充，不设置则为全局填充
- -begin_pos  设定在某边某一个线段内进行填充，如果不设置，则默认该边全部填充
- -end_pos  此处begin与pos的值为double，为没有乘DBU之前的值，与init类似

### 示例：

placeIoFiller \
    -filler_types "PFILL50W PFILL20W PFILL10W PFILL5W PFILL2W PFILL01W PFILL001W"

​	#-prefix

​	#-edge

​	#-begin_pos

​	#-end_pos

## 5.tapcell

放置tapcell以及endcap

- -tapcell 设置tapcell的种类
- -distance 32.5 设置tapcell的间距，此处为没有乘DBU的值
- -endcap endcap的种类

### 示例：

tapcell \
    -tapcell LVT_FILLTIEHD \
    -distance 32.5 \
    -endcap LVT_F_FILLHD1

## 6.global_net_connect

创建电源net

- -net_name 电源网络名称
- -instance_pin_name  instance连接该网络的pin的名称。当前还不支持指定某些instance的该pin连接到该电源网络，默认为全局含有该pin的instance都连接到该网络
- -is_power  需设置为1或0： 1代表use power，0代表use ground

### 示例：

global_net_connect \
    -net_name VDD \
    -instance_pin_name VDD \
    -is_power 1

global_net_connect \
    -net_name VDD \
    -instance_pin_name VDDIO \
    -is_power 1

global_net_connect \
    -net_name VSS \
    -instance_pin_name VSS \
    -is_power 0

## 7.add_pdn_io

为电源NET添加IOPIN

- -net_name	电源网络名称
- -direction   参数（INPUT、OUTPUT、INOUT、FEEDTHRU、OUTTRI），pin的数据direction
- -pin_name 可选参数，默认为电源网络名称

### 示例：

add_pdn_io \
    -net_name VDD \
    -direction INOUT 

​	#-pin_name VDD 

## 8.place_pdn_Port

为电源网络的IOPIN添加PORT

- -pin_name  iopin名字
- -io_cell_name io io cell的名字
- -offset_x  相对io cell的port矩形的左下角坐标
- -offset_y  相对io cell的port矩形的左下角坐标
- -width  矩形宽度
- -height 矩形高度
- -layer  port所属绕线层

### 示例：

place_pdn_Port \
    -pin_name VDD \
    -io_cell_name xxx\
    -offset_x 10 \
    -offset_y 10 \
    -width 100 \
    -height 100 \
    -layer ALPA

place_pdn_Port \
    -pin_name VDD \
    -io_cell_name xxx\
    -offset_x 20 \
    -offset_y 20 \
    -width 200 \
    -height 200 \
    -layer ALPA                这两个命令可以为VDD pin添加两个port

![image-20211028162925571](pic/image-20211028162925571.png)

## 9.create_grid

生成标准单元供电线，会生成绕线信息

- -layer_name 生成电源网格的层
- -net_name_power power net name
- -net_name_ground ground net name
- -width 线宽。是没有乘DBU的数值 
- -offset 相对于core边界的偏移量，建议设置为0，仅测试过偏移为0的情况。是没有乘DBU的数值 

### 示例：

create_grid \
    -layer_name "METAL1" \
    -net_name_power VDD \
    -net_name_ground VSS \
    -width 0.24 \
    -offset 0

## 10.create_stripe

生成标准单元条形电源线

- -layer_name  生成电源线的层
- -net_name_power  power net name
- -net_name_ground  ground net name
- -width 线宽。是没有乘DBU的数值 
- -pitch  同类型电源线的间距。对于标准单元来说，同层的power线与ground线间距为0.5*pitch
- -offset 相对于core边界的偏移量。是没有乘DBU的数值 

### 示例：

create_stripe \
   -layer_name "METAL5" \
   -net_name_power VDD \
   -net_name_ground VSS \
   -width 1.64 \
   -pitch 13.12 \
   -offset 3.895

## 11.connect_two_layer

连接指定的两层的电源线

- -layers ：可以一对一对的输入，也可以将全部需要连接的层信息一起输入

### 示例：

set connect1 "METAL1 METAL4" \
set connect2 "METAL4 METAL5" \
set connect3 "METAL5 METAL6" \
set connect4 "METAL6 METAL7" \
set connect5 "METAL7 METAL8" \
set connect6 "METAL8 ALPA" \
connect_two_layer \
    -layers [concat \$connect1 ​\$connect2 ​\$connect3 ​\$connect4 ​\$connect5 $connect6] 

1. connect_two_layer \
       -layers [concat  \$connect1 $connect2]
2. connect_two_layer \
       -layers "METAL1 METAL4"
   connect_two_layer \
       -layers "METAL4 METAL5"    **序号1，2的效果是一样的**

**PS：被连接的两层需要含有电源线**

## 12.connect_io_pin_to_pdn
将电源NET的IOPIN的Port连接至Core内电源线。(会对Port的坐标进行检查)
- -point_list 连接关系所经过的拐角处的坐标点（起点和终点的坐标也需要有），也可以只输入起点和终点坐标
- -layer 想要进行连线的层

### 示例：
connect_io_pin_to_pdn \
    -point_list "998 2802 915 2598" \
    -layer METAL1

## 13.connect_pdn_stripe
- -point_list 连接关系所经过的拐角处的坐标点（起点和终点的坐标也需要有）
- -net_name 想要连接到的电源网络的名称
- -layer 想要进行连线的层
### 示例：
connect_pdn_stripe \
    -point_list "2675.606 1998.707 2680.606 1998.707 2680.606 1892.165 2803.686 1892.165" \
    -net_name VDD \
    -layer ALPA

connect_pdn_stripe \
    -point_list "2675.606 1998.707 2680.606 1998.707" \
    -net_name VDD \
    -layer ALPA \
    -width 100
connect_pdn_stripe \
    -point_list "2680.606 1892.165 2803.686 1892.165" \
    -net_name VDD \
    -layer ALPA \
    -width 100

## add_segment_via
给电源线增加通孔

-  -net_name   电源线的名称
-  -layer   通孔所在层
-  -top_layer metal层 top层
-  -bottom_layer metal层 bottom层
​-  -offset_x 相对于所连接的IOCELL的左下角坐标的偏移量，此处需设置绝对长度，即乘过DBU之后的值
​-  -offset_y 相对于所连接的IOCELL的左下角坐标的偏移量，此处需设置绝对长度，即乘过DBU之后的值
​-  -width 矩形宽度，相对于所连接的IOCELL的左下角坐标的偏移量，此处需设置绝对长度，即乘过DBU之后的值
​-  -height 矩形高度，相对于所连接的IOCELL的左下角坐标的偏移量，此处需设置绝对长度，即乘过DBU之后的值

### 示例：

add_segment_via \
    -net_name VDD \
    -layer VIA45 \
    -offset_x 9000 \
    -offset_y 71500 \
    -width 58000 \
    -height 58000 

add_segment_via \
    -net_name VDD \
    -layer VIA45 \
    -offset_x 10000 \
    -offset_y 81500 \
    -width 58000 \
    -height 58000 

add_segment_via \
    -net_name VDD \
    -top_layer METAL8 \
    -bottom_layer METAL1 \
    -offset_x 10000 \
    -offset_y 81500 \
    -width 58000 \
    -height 58000 

add_segment_via \
    -net_name VDDIO \
    -top_layer METAL8 \
    -bottom_layer METAL1 \
    -offset_x 10000 \
    -offset_y 81500 \
    -width 58000 \
    -height 58000 

## add_segment_stripe
- -point_list 生成stripe的连接点，当连接点个数大于2时，前后两个连接点两两生成stripe，
- -net_name 想要连接到的电源网络的名称
- -layer 想要进行连线的层
- -width 指定线宽
add_segment_stripe \
    -point_list "2680.606 1892.165 2803.686 1892.165" \
    -net_name VDDIO \
    -layer ALPA \
    -width 100

add_segment_stripe \
    -point_list "1680.606 2892.165 2803.686 2892.165 2803.686 1792.165" \
    -net_name VDDIO \
    -layer ALPA \
    -width 100

add_segment_stripe \
    -point_begin "3680.606 2892.165" \
    -layer_start METAL4 \
    -point_end "4680.606 2892.165" \
    -layer_end METAL8 \
    -net_name VDDIO \
    -via_width 100 \
    -via_height 200
    
    


## read_lef

读入lef文件，以字符串列表的形式读入

### 示例：

read_lef "../../Designs/scc011u_8lm_1tm_thin_ALPA/scc011u_8lm_1tm_thin_ALPA.lef \
          ../../Designs/scc011u_8lm_1tm_thin_ALPA/scc011ums_hd_lvt.lef \
          ../../Designs/scc011u_8lm_1tm_thin_ALPA/S013PLLFN_8m_V1_2_1.lef \
          ../../Designs/scc011u_8lm_1tm_thin_ALPA/SP013D3WP_V1p7_8MT.lef \
          ../../Designs/scc011u_8lm_1tm_thin_ALPA/S011HD1P256X32M2B0.lef \
          ../../Designs/scc011u_8lm_1tm_thin_ALPA/S011HD1P512X58M2B0.lef \
          ../../Designs/scc011u_8lm_1tm_thin_ALPA/S011HD1P1024X64M4B0.lef \
          ../../Designs/scc011u_8lm_1tm_thin_ALPA/S011HD1P256X8M4B0.lef \
          ../../Designs/scc011u_8lm_1tm_thin_ALPA/S011HD1P512X73M2B0.lef \
          ../../Designs/scc011u_8lm_1tm_thin_ALPA/S011HD1P128X21M2B0.lef \
          ../../Designs/scc011u_8lm_1tm_thin_ALPA/S011HD1P512X19M4B0.lef \
          ../../Designs/scc011u_8lm_1tm_thin_ALPA/S011HDSP4096X64M8B0.lef "

## read_def

读入def文件，以字符串的形式读入

### 示例：

read_def "./asic_top.def";

## write_def

写出def文件。参数为写出的文件路径。

### 示例：

​	write_def "iEDA_FP.def"


​	

​	

# 使用要点

## 1.建议在所有pdn操作之前插入tapcell

## 2.建议按照手册的顺序执行命令

# 