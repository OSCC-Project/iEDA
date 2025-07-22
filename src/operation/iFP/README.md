# iFP: Floorplanning

## **FP_TCL Command User Manual**

**Dependencies**

- sudo apt-get install libunwind-dev
- sudo apt-get install libspdlog-dev (env : ubuntu20.04)
- sudo apt-get install boost 1.71
- sudo apt-get install eigen

**Build**

git clone --recursive 
cd iEDA
mkdir build
cd build
cmake..
make

**Run**

cd iEDA/bin

./FP gtest.tcl

**Tool Update**

cd iEDA

git pull

rm -rf build

mkdir build

cd build

cmake..

make

**#1. init_floorplan**

- -die_area   The area of the die, a string, with each value separated by spaces. The values here are not multiplied by DBU.
- -core_area   The area of the core, a string, with each value separated by spaces. The values here are not multiplied by DBU.
- -core_site  The selection of the core site.
- -io_site  For the 110 process, this is an **optional parameter**, and "IOSite" will be selected by default.

**## Example: **

set DIE_AREA "0.0    0.0   2843    2843"

set CORE_AREA "150 150 2693 2693"

set PLACE_SITE HD_CoreSite

set IO_SITE IOSite

init_floorplan \

​	-die_area $DIE_AREA \

​	-core_area $CORE_AREA \

​	-core_site $PLACE_SITE \

​	-io_site $IO_SITE

**#2. placeInst**

The tested functions include:

1. Placing IOCells
2. Placing power CELLs
3. Placing IOFILLERs
4. Placing CORNERs

When placing the above four types, placement checks will be performed. The rules are whether they are placed according to IOSite and whether they are on the DIE BOUNDARY.

   5. Placing standard cells

- -inst_name  The name of the instance.
- -llx  The abscissa of the lower left corner. The value at this time is the absolute coordinate, that is, the value after multiplying by DBU should be set.
- -lly  The ordinate of the lower left corner.
- -orient  Orientation, can use (N, S, W, E) or (R0, R180, R90, R270)
- -cellmaster  The type of the cell.

**## Example: **

placeInst \
   -inst_name u0_clk \
   -llx  0 \
   -lly 2510940 \
   -orient E \
   -cellmaster PX1W

**#3. placePort**

This command is only for generating ports of IOPINS connected to IOCells. Ports of power CELLs do not use this interface.

-  -pin_name   The name of the iopin.
​-  -offset_x The offset relative to the lower left corner coordinate of the connected IOCell. The absolute length should be set here, that is, the value after multiplying by DBU.
​-  -offset_y The offset relative to the lower left corner coordinate of the connected IOCell. The absolute length should be set here, that is, the value after multiplying by DBU.
​-  -width  The width of the rectangle. The offset relative to the lower left corner coordinate of the connected IOCell. The absolute length should be set here, that is, the value after multiplying by DBU.
​-  -height  The height of the rectangle. The offset relative to the lower left corner coordinate of the connected IOCell. The absolute length should be set here, that is, the value after multiplying by DBU.
​-  -layer  The name of the layer it is on.

**## Example: **

placePort \
    -pin_name osc_in_pad \
    -offset_x 9000 \
    -offset_y 71500 \
    -width 58000 \
    -height 58000 \
    -layer ALPA

**#4. placeIoFiller**

Place IOFILLERs, supporting automatic filling on four sides.

Required parameter: -filler_types  The type of IOFILLER.

Optional parameters:

- -prefix  The prefix of the generated filler name. The default is IOFill.
- -edge  Set which edge to fill. If not set, it is global filling.
- -begin_pos  Set to fill within a certain line segment on a certain edge. If not set, the entire edge is filled by default.
- -end_pos  The values of begin and pos here are doubles and are values before multiplying by DBU, similar to init.

**## Example: **

placeIoFiller \
    -filler_types "PFILL50W PFILL20W PFILL10W PFILL5W PFILL2W PFILL01W PFILL001W"

​	#-prefix

​	#-edge

​	#-begin_pos

​	#-end_pos

**#5. tapcell**

Place tapcells and endcaps.

- -tapcell Set the type of tapcell.
- -distance 32.5 Set the distance of tapcells. The value here is not multiplied by DBU.
- -endcap The type of endcap.

**## Example: **

tapcell \
    -tapcell LVT_FILLTIEHD \
    -distance 32.5 \
    -endcap LVT_F_FILLHD1

**#6. global_net_connect**

Create power nets.

- -net_name  The name of the power network.
- -instance_pin_name  The name of the pin of the instance connected to this network. Currently, it does not support specifying that the pins of certain instances are connected to this power network. By default, all instances with this pin globally are connected to this network.
- -is_power  Should be set to 1 or 0: 1 represents use power, 0 represents use ground.

**## Example: **

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

**#7. add_pdn_io**

Add IOPINS for the power NET.

- -net_name  The name of the power network.
- -direction  Parameter (INPUT, OUTPUT, INOUT, FEEDTHRU, OUTTRI), the data direction of the pin.
- -pin_name  Optional parameter. The default is the name of the power network.

**## Example: **

add_pdn_io \
    -net_name VDD \
    -direction INOUT 

​	#-pin_name VDD 

**#8. place_pdn_Port**

Add PORTs for the IOPINS of the power network.

- -pin_name  The name of the iopin.
- -io_cell_name The name of the io io cell.
- -offset_x  Offset relative to the lower left corner of the port rectangle of the io cell.
- -offset_y  Offset relative to the lower left corner of the port rectangle of the io cell.
- -width  The width of the rectangle.
- -height  The height of the rectangle.
- -layer  The routing layer to which the port belongs.

**## Example: **

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
    -layer ALPA  These two commands can add two ports for the VDD pin.

<!--![image-20211028162925571](pic/image-20211028162925571.png) -->

**#9. create_grid**

Generate power supply lines for standard cells and generate routing information.

- -layer_name  The layer on which the power grid is generated.
- -net_name_power The name of the power net.
- -net_name_ground The name of the ground net.
- -width  The line width. The value is not multiplied by DBU.
- -offset  The offset relative to the core boundary. It is recommended to set it to 0. Only the case where the offset is 0 has been tested. The value is not multiplied by DBU.

**## Example: **

create_grid \
    -layer_name "METAL1" \
    -net_name_power VDD \
    -net_name_ground VSS \
    -width 0.24 \
    -offset 0

**#10. create_stripe**

Generate bar-shaped power supply lines for standard cells.

- -layer_name  The layer on which the power supply line is generated.
- -net_name_power  The name of the power net.
- -net_name_ground  The name of the ground net.
- -width  The line width. The value is not multiplied by DBU.
- -pitch  The pitch of the same type of power supply lines. For standard cells, the pitch between the power line and the ground line of the same layer is 0.5*pitch.
- -offset  The offset relative to the core boundary. The value is not multiplied by DBU.

**## Example: **

create_stripe \
   -layer_name "METAL5" \
   -net_name_power VDD \
   -net_name_ground VSS \
   -width 1.64 \
   -pitch 13.12 \
   -offset 3.895

**#11. connect_two_layer**

Connect the specified two layers of power supply lines.

- -layers  : Can be input one pair at a time or all the layer information that needs to be connected can be input together.

**## Example: **

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
       -layers "METAL4 METAL5"    **The effects of sequence 1 and 2 are the same**

**PS: The two layers to be connected need to contain power supply lines**

**#12. connect_io_pin_to_pdn**

Connect the Port of the IOPIN of the power NET to the power supply line in the Core. (The coordinates of the Port will be checked)
- -point_list  The coordinate points of the corners passed by the connection relationship (the coordinates of the starting point and the ending point also need to be included), or only the starting and ending coordinates can be input.
- -layer  The layer on which you want to perform the connection.

**## Example: **
connect_io_pin_to_pdn \
    -point_list "998 2802 915 2598" \
    -layer METAL1

**#13. connect_pdn_stripe**
- -point_list  The coordinate points of the corners passed by the connection relationship (the starting point and the ending point coordinates also need to be included)
- -net_name  The name of the power network you want to connect to
- -layer  The layer on which you want to perform the connection
**## Example: **
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

**#14. add_segment_via**
Add vias to the power supply line.

-  -net_name   The name of the power supply line.
-  -layer   The layer where the via is located.
-  -top_layer The top layer of the metal layer.
-  -bottom_layer The bottom layer of the metal layer.
​-  -offset_x The offset relative to the lower left corner coordinate of the connected IOCell. The absolute length should be set here, that is, the value after multiplying by DBU.
​-  -offset_y The offset relative to the lower left corner coordinate of the connected IOCell. The absolute length should be set here, that is, the value after multiplying by DBU.
​-  -width  The width of the rectangle. The offset relative to the lower left corner coordinate of the connected IOCell. The absolute length should be set here, that is, the value after multiplying by DBU.
​-  -height  The height of the rectangle. The offset relative to the lower left corner coordinate of the connected IOCell. The absolute length should be set here, that is, the value after multiplying by DBU.

**## Example: **

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

**#15. add_segment_stripe**
- -point_list  The connection points for generating the stripe. When the number of connection points is greater than 2, the front and back connection points generate stripes pairwise.
- -net_name  The name of the power network you want to connect to.
- -layer  The layer on which you want to perform the connection.
- -width  Specify the line width.
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
    
    


**#16. read_lef**

Read the lef file in the form of a string list.

**## Example: **

read_lef "../../Designs/scc011u_8lm_1tm_thin_ALPA/scc011u_8lm_1tm_th