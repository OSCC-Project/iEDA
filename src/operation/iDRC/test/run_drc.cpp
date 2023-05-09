// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#include "DRC.h"
#include "DrcAPI.hpp"
#include<string>

using namespace idrc;

class DrcRect;
class DrcRectangle;

// void init_net1(LayerNameToRTreeMap& layer_to_rects_rtree_map)
// {
//   DrcRect* rect1 = new DrcRect(2, DrcRectangle<int>(100, 100, 300, 600));
//   rect1->set_net_id(1);
//   RTreeBox rTreeBox1 = DRCUtil::getRTreeBox(rect1);
//   layer_to_rects_rtree_map["METAL2"].insert(std::make_pair(rTreeBox1, rect1));

//   DrcRect* rect2 = new DrcRect(2, DrcRectangle<int>(100, 100, 600, 300));
//   rect2->set_net_id(1);
//   RTreeBox rTreeBox2 = DRCUtil::getRTreeBox(rect2);
//   layer_to_rects_rtree_map["METAL2"].insert(std::make_pair(rTreeBox2, rect2));

//   DrcRect* rect3 = new DrcRect(2, DrcRectangle<int>(400, 50, 600, 350));
//   rect3->set_net_id(1);
//   RTreeBox rTreeBox3 = DRCUtil::getRTreeBox(rect3);
//   layer_to_rects_rtree_map["METAL2"].insert(std::make_pair(rTreeBox3, rect3));

//   DrcRect* rect4 = new DrcRect(2, DrcRectangle<int>(100, 450, 300, 650));
//   rect4->set_net_id(1);
//   RTreeBox rTreeBox4 = DRCUtil::getRTreeBox(rect4);
//   layer_to_rects_rtree_map["METAL2"].insert(std::make_pair(rTreeBox4, rect4));
// }

// void init_net2(LayerNameToRTreeMap& layer_to_rects_rtree_map)
// {
//   DrcRect* rect1 = new DrcRect(2, DrcRectangle<int>(100, 700, 600, 900));
//   rect1->set_net_id(2);
//   RTreeBox rTreeBox1 = DRCUtil::getRTreeBox(rect1);
//   layer_to_rects_rtree_map["METAL2"].insert(std::make_pair(rTreeBox1, rect1));

//   DrcRect* rect2 = new DrcRect(2, DrcRectangle<int>(550, 850, 750, 1100));
//   rect2->set_net_id(2);
//   RTreeBox rTreeBox2 = DRCUtil::getRTreeBox(rect2);
//   layer_to_rects_rtree_map["METAL2"].insert(std::make_pair(rTreeBox2, rect2));
// }

// void init_net3(LayerNameToRTreeMap& layer_to_rects_rtree_map)
// {
//   DrcRect* rect1 = new DrcRect(2, DrcRectangle<int>(200, 1150, 500, 1350));
//   rect1->set_net_id(3);
//   RTreeBox rTreeBox1 = DRCUtil::getRTreeBox(rect1);
//   layer_to_rects_rtree_map["METAL2"].insert(std::make_pair(rTreeBox1, rect1));

//   DrcRect* rect2 = new DrcRect(2, DrcRectangle<int>(150, 1200, 450, 1400));
//   rect2->set_net_id(3);
//   RTreeBox rTreeBox2 = DRCUtil::getRTreeBox(rect2);
//   layer_to_rects_rtree_map["METAL2"].insert(std::make_pair(rTreeBox2, rect2));
// }

// void init_net4(LayerNameToRTreeMap& layer_to_rects_rtree_map)
// {
//   DrcRect* rect1 = new DrcRect(2, DrcRectangle<int>(1200, 400, 1400, 1000));
//   rect1->set_net_id(4);
//   RTreeBox rTreeBox1 = DRCUtil::getRTreeBox(rect1);
//   layer_to_rects_rtree_map["METAL2"].insert(std::make_pair(rTreeBox1, rect1));

//   DrcRect* rect2 = new DrcRect(2, DrcRectangle<int>(1200, 400, 1800, 600));
//   rect2->set_net_id(4);
//   RTreeBox rTreeBox2 = DRCUtil::getRTreeBox(rect2);
//   layer_to_rects_rtree_map["METAL2"].insert(std::make_pair(rTreeBox2, rect2));

//   DrcRect* rect3 = new DrcRect(2, DrcRectangle<int>(1200, 800, 1800, 1000));
//   rect3->set_net_id(4);
//   RTreeBox rTreeBox3 = DRCUtil::getRTreeBox(rect3);
//   layer_to_rects_rtree_map["METAL2"].insert(std::make_pair(rTreeBox3, rect3));

//   DrcRect* rect4 = new DrcRect(2, DrcRectangle<int>(1600, 400, 1800, 1000));
//   rect4->set_net_id(4);
//   RTreeBox rTreeBox4 = DRCUtil::getRTreeBox(rect4);
//   layer_to_rects_rtree_map["METAL2"].insert(std::make_pair(rTreeBox4, rect4));
// }

// void init_block(LayerNameToRTreeMap& layer_to_rects_rtree_map)
// {
//   DrcRect* rect = new DrcRect(2, DrcRectangle<int>(700, 50, 900, 350));
//   RTreeBox rTreeBox = DRCUtil::getRTreeBox(rect);
//   layer_to_rects_rtree_map["METAL2"].insert(std::make_pair(rTreeBox, rect));
// }

// void initLayerToRectsRtree(LayerNameToRTreeMap& layer_to_rects_rtree_map)
// {
//   init_net1(layer_to_rects_rtree_map);
//   init_net2(layer_to_rects_rtree_map);
//   init_net3(layer_to_rects_rtree_map);
//   init_net4(layer_to_rects_rtree_map);
//   init_block(layer_to_rects_rtree_map);
// }

// int getLayerIdByLayerName(const std::string& layerName)
// {
//   if (layerName == "METAL1") {
//     return 1;
//   } else if (layerName == "METAL2") {
//     return 2;
//   } else if (layerName == "METAL3") {
//     return 3;
//   } else if (layerName == "METAL4") {
//     return 4;
//   } else if (layerName == "METAL5") {
//     return 5;
//   } else if (layerName == "METAL6") {
//     return 6;
//   } else if (layerName == "METAL7") {
//     return 7;
//   } else if (layerName == "METAL8") {
//     return 8;
//   }
//   std::cout << "layer is not exist!!!" << std::endl;
//   return 9;
// }

// void writeGDS(LayerNameToRTreeMap& layer_to_rects_rtree_map)
// {
//   std::string file_path_name = "run_drc.gds";
//   std::ofstream gds_file(file_path_name);
//   if (!gds_file.is_open()) {
//     return;
//   }
//   gds_file << "HEADER 600" << std::endl;
//   gds_file << "BGNLIB" << std::endl;
//   gds_file << "LIBNAME DensityLib" << std::endl;
//   gds_file << "UNITS 0.001 1e-9" << std::endl;
//   for (auto& [layerName, rtree] : layer_to_rects_rtree_map) {
//     int layerId = getLayerIdByLayerName(layerName);
//     for (auto it = rtree.begin(); it != rtree.end(); ++it) {
//       DrcRect* rect = it->second;
//       gds_file << "BGNSTR" << std::endl;
//       gds_file << "STRNAME Rect" << rect << std::endl;
//       gds_file << "BOUNDARY" << std::endl;
//       gds_file << "LAYER " << layerId << std::endl;
//       gds_file << "DATATYPE 0" << std::endl;
//       gds_file << "XY" << std::endl;
//       gds_file << rect->get_left() << " : " << rect->get_bottom() << std::endl;
//       gds_file << rect->get_right() << " : " << rect->get_bottom() << std::endl;
//       gds_file << rect->get_right() << " : " << rect->get_top() << std::endl;
//       gds_file << rect->get_left() << " : " << rect->get_top() << std::endl;
//       gds_file << rect->get_left() << " : " << rect->get_bottom() << std::endl;
//       gds_file << "ENDEL" << std::endl;
//       gds_file << "ENDSTR" << std::endl;
//     }
//   }

//   gds_file << "BGNSTR" << std::endl;
//   gds_file << "STRNAME Rect" << std::endl;

//   for (auto& [layerName, rtree] : layer_to_rects_rtree_map) {
//     for (auto it = rtree.begin(); it != rtree.end(); ++it) {
//       DrcRect* rect = it->second;
//       gds_file << "SREF" << std::endl;
//       gds_file << "SNAME Rect" << rect << std::endl;
//       gds_file << "XY 0:0" << std::endl;
//       gds_file << "ENDEL" << std::endl;
//     }
//   }
//   gds_file << "ENDSTR" << std::endl;
//   gds_file << "ENDLIB" << std::endl;
// }

// void writeGDS(DRC* drc)
// {
//   std::string file_path_name = "run_drc.gds";
//   std::ofstream gds_file(file_path_name);
//   if (!gds_file.is_open()) {
//     return;
//   }
//   gds_file << "HEADER 600" << std::endl;
//   gds_file << "BGNLIB" << std::endl;
//   gds_file << "LIBNAME DensityLib" << std::endl;
//   gds_file << "UNITS 0.001 1e-9" << std::endl;

//   DrcDesign* drcDesign = drc->get_drc_design();

//   for (auto& [layerId, block_list] : drcDesign->get_layer_to_blockage_list()) {
//     for (auto& block : block_list) {
//       gds_file << "BGNSTR" << std::endl;
//       gds_file << "STRNAME BlockRect" << block << std::endl;
//       gds_file << "BOUNDARY" << std::endl;
//       gds_file << "LAYER " << block->get_layer_id() << std::endl;
//       gds_file << "DATATYPE 0" << std::endl;
//       gds_file << "XY" << std::endl;
//       gds_file << block->get_left() << " : " << block->get_bottom() << std::endl;
//       gds_file << block->get_right() << " : " << block->get_bottom() << std::endl;
//       gds_file << block->get_right() << " : " << block->get_top() << std::endl;
//       gds_file << block->get_left() << " : " << block->get_top() << std::endl;
//       gds_file << block->get_left() << " : " << block->get_bottom() << std::endl;
//       gds_file << "ENDEL" << std::endl;
//       gds_file << "ENDSTR" << std::endl;
//     }
//   }

//   for (auto& net : drcDesign->get_drc_net_list()) {
//     for (auto& [layerId, pin_rect_list] : net->get_layer_to_pin_rects_map()) {
//       // net pin rect
//       for (auto& pin_rect : pin_rect_list) {
//         gds_file << "BGNSTR" << std::endl;
//         gds_file << "STRNAME PinRect" << pin_rect << std::endl;
//         gds_file << "BOUNDARY" << std::endl;
//         gds_file << "LAYER " << pin_rect->get_layer_id() << std::endl;
//         gds_file << "DATATYPE 0" << std::endl;
//         gds_file << "XY" << std::endl;
//         gds_file << pin_rect->get_left() << " : " << pin_rect->get_bottom() << std::endl;
//         gds_file << pin_rect->get_right() << " : " << pin_rect->get_bottom() << std::endl;
//         gds_file << pin_rect->get_right() << " : " << pin_rect->get_top() << std::endl;
//         gds_file << pin_rect->get_left() << " : " << pin_rect->get_top() << std::endl;
//         gds_file << pin_rect->get_left() << " : " << pin_rect->get_bottom() << std::endl;
//         gds_file << "ENDEL" << std::endl;
//         gds_file << "ENDSTR" << std::endl;
//       }
//     }
//   }

//   for (auto& net : drcDesign->get_drc_net_list()) {
//     for (auto& [layerId, routing_rect_list] : net->get_layer_to_routing_rects_map()) {
//       // net rouitng rect
//       for (auto& routing_rect : routing_rect_list) {
//         gds_file << "BGNSTR" << std::endl;
//         gds_file << "STRNAME RoutingRect" << routing_rect << std::endl;
//         gds_file << "BOUNDARY" << std::endl;
//         gds_file << "LAYER " << routing_rect->get_layer_id() << std::endl;
//         gds_file << "DATATYPE 0" << std::endl;
//         gds_file << "XY" << std::endl;
//         gds_file << routing_rect->get_left() << " : " << routing_rect->get_bottom() << std::endl;
//         gds_file << routing_rect->get_right() << " : " << routing_rect->get_bottom() << std::endl;
//         gds_file << routing_rect->get_right() << " : " << routing_rect->get_top() << std::endl;
//         gds_file << routing_rect->get_left() << " : " << routing_rect->get_top() << std::endl;
//         gds_file << routing_rect->get_left() << " : " << routing_rect->get_bottom() << std::endl;
//         gds_file << "ENDEL" << std::endl;
//         gds_file << "ENDSTR" << std::endl;
//       }
//     }
//   }

//   // block
//   gds_file << "BGNSTR" << std::endl;
//   gds_file << "STRNAME BlockRect" << std::endl;

//   for (auto& [layerId, block_list] : drcDesign->get_layer_to_blockage_list()) {
//     for (auto& block : block_list) {
//       gds_file << "SREF" << std::endl;
//       gds_file << "SNAME BlockRect" << block << std::endl;
//       gds_file << "XY 0:0" << std::endl;
//       gds_file << "ENDEL" << std::endl;
//     }
//   }
//   gds_file << "ENDSTR" << std::endl;
//   // pin
//   gds_file << "BGNSTR" << std::endl;
//   gds_file << "STRNAME PinRect" << std::endl;

//   for (auto& net : drcDesign->get_drc_net_list()) {
//     for (auto& [layerId, pin_rect_list] : net->get_layer_to_pin_rects_map()) {
//       // net pin rect
//       for (auto& pin_rect : pin_rect_list) {
//         gds_file << "SREF" << std::endl;
//         gds_file << "SNAME PinRect" << pin_rect << std::endl;
//         gds_file << "XY 0:0" << std::endl;
//         gds_file << "ENDEL" << std::endl;
//       }
//     }
//   }
//   gds_file << "ENDSTR" << std::endl;

//   // via and segment
//   gds_file << "BGNSTR" << std::endl;
//   gds_file << "STRNAME RoutingRect" << std::endl;

//   for (auto& net : drcDesign->get_drc_net_list()) {
//     for (auto& [layerId, routing_rect_list] : net->get_layer_to_routing_rects_map()) {
//       // net rouitng rect
//       for (auto& routing_rect : routing_rect_list) {
//         gds_file << "SREF" << std::endl;
//         gds_file << "SNAME RoutingRect" << routing_rect << std::endl;
//         gds_file << "XY 0:0" << std::endl;
//         gds_file << "ENDEL" << std::endl;
//       }
//     }
//   }
//   gds_file << "ENDSTR" << std::endl;

//   //------------------------------------------

//   gds_file << "BGNSTR" << std::endl;
//   gds_file << "STRNAME Run Drc" << std::endl;

//   gds_file << "SREF" << std::endl;
//   gds_file << "SNAME BlockRect" << std::endl;
//   gds_file << "XY 0:0" << std::endl;
//   gds_file << "ENDEL" << std::endl;

//   gds_file << "SREF" << std::endl;
//   gds_file << "SNAME PinRect" << std::endl;
//   gds_file << "XY 0:0" << std::endl;
//   gds_file << "ENDEL" << std::endl;

//   gds_file << "SREF" << std::endl;
//   gds_file << "SNAME RoutingRect" << std::endl;
//   gds_file << "XY 0:0" << std::endl;
//   gds_file << "ENDEL" << std::endl;

//   gds_file << "ENDSTR" << std::endl;
//   gds_file << "ENDLIB" << std::endl;
// }

/////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
///////////////////////////////////////////////////////
/********************************************/
/*******check drc in the result of rt*******/
/******************************************/
// int main(int argc, char* argv[])
// {
//   if (argc != 2) {
//     std::cout << "Please run 'run_drc <drc_config_path>'!" << std::endl;
//     exit(1);
//   }
//   std::string drc_config_path = argv[1];

//   // main
//   DRC* drc = new DRC();

//   LayerNameToRTreeMap layer_to_rects_rtree_map;
//   initLayerToRectsRtree(layer_to_rects_rtree_map);
//   writeGDS(layer_to_rects_rtree_map);
//   // 1.用IDB初始化Tech
//   //函数重载，参数可换为iDB_Builder指针
//   drc->initTechFromIDB(drc_config_path);
//   // 2.初始化规则数据后初始化各个检查模块
//   drc->initCheckModule();
//   // 3.检查iRT的绕线结果，数据参数为层名字对应矩形块R树
//   drc->checkiRTResult(layer_to_rects_rtree_map);
//   return 0;
// }

/********************************************/
/*******check drc by reading def lef********/
/******************************************/
// int main(int argc, char* argv[])
// {
//   std::string drc_config_path = "/home/zhangmz/code/config/57w/drc_config.json";
//   // main
//   DRC* drc = new DRC();

//   /***************1.通过配置文件初始化DRC********************/
//   std::cout << "init DRC Tech and Design ......" << std::endl;
//   drc->initDRC(drc_config_path);
//   /***************2.初始化各个设计规则检查模块***************/
//   std::cout << "init DRC check module ......" << std::endl;
//   drc->initCheckModule();
//   /***************3.运行各个设计规则检查模块****************/
//   std::cout << "run DRC check module ......" << std::endl;
//   drc->run();
//   /***************4.文件形式报告设计规则检查结果***********/
//   std::cout << "report check result ......" << std::endl;
//   drc->report();
//   return 0;
// }

int main(int argc, char* argv[])
{
  std::string drc_config_path = "/home/zhangmz/code/config/57w/drc_config.json";
  // // main
  // std::cout << "init DRC Tech and Design ......" << std::endl;
  // DrcAPIInst.initDRC(drc_config_path);
  // std::cout << "init DRC check module ......" << std::endl;
  // std::cout << "run DRC check module ......" << std::endl;
  // DrcAPIInst.runDRC();
  // std::cout << "report check result ......" << std::endl;
  // DrcAPIInst.reportDRC();

  /***************1.通过配置文件初始化DRC********************/
  std::cout << "init DRC Tech and Design ......" << std::endl;
  DrcInst.initDRC(drc_config_path);
  /***************2.初始化各个设计规则检查模块***************/
  std::cout << "init DRC check module ......" << std::endl;
  DrcInst.initCheckModule();
  /***************3.运行各个设计规则检查模块****************/
  std::cout << "run DRC check module ......" << std::endl;
  DrcInst.run();
  /***************4.文件形式报告设计规则检查结果***********/
  std::cout << "report check result ......" << std::endl;
  DrcInst.report();
  return 0;
  // /***************1.通过配置文件初始化DRC********************/
  // std::cout << "init DRC Tech and Design ......" << std::endl;
  // drc->initDRC(drc_config_path);
  // /***************2.初始化各个设计规则检查模块***************/
  // std::cout << "init DRC check module ......" << std::endl;
  // drc->initCheckModule();
  // /***************3.运行各个设计规则检查模块****************/
  // std::cout << "run DRC check module ......" << std::endl;
  // drc->run();
  // /***************4.文件形式报告设计规则检查结果***********/
  // std::cout << "report check result ......" << std::endl;
  // drc->report();
  // return 0;
}


/********************************************/
/**************multi-patterning************/
/******************************************/

// int main(int argc, char* argv[])
// {
//   if (argc != 2) {
//     std::cout << "Please run 'run_drc <drc_config_path>'!" << std::endl;
//     exit(1);
//   }
//   std::string drc_config_path = argv[1];

//   // main
//   DRC* drc = new DRC();

//   /*************************************/
//   // std::cout << "init DRC Tech and Design ......" << std::endl;
//   // drc->initDRC(drc_config_path);
//   /*************************************/
//   // std::cout << "init DRC polygon ......" << std::endl;
//   //  drc->initNetsMergePolygon();    // changge
//   //  drc->initDesignBlockPolygon();  // changge
//   /*************************************/
//   // std::cout << "init DRC check module ......" << std::endl;
//   // drc->initCheckModule();
//   /*************************************/
//   // std::cout << "run DRC check module ......" << std::endl;
//   // drc->run();
//   /*************************************/
//   // 4.读入区域绕线结果并运行各模块检测
//   // std::cout << "report check result ......" << std::endl;
//   // drc->report();
//   /*************************************/
//   // std::cout << "init graph by polygon....." << std::endl;    // change
//   // drc->initConflictGraphByPolygon();                         // change
//   // std::cout << "check multipatterning ......" << std::endl;  // change
//   // drc->checkMultipatterning(2);                              // change
//   // writeGDS(drc);
//   // std::cout << "generate gds" << std::endl;
//   /*************************************/
//   return 0;
// }