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
/**
 * @file PowerVia.cpp
 * @author Jianrong Su
 * @brief 处理电源网络中的通孔连接
 * @version 0.1
 * @date 2025-03-12
 */

#include "PowerVia.hh"

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace ipnp {

/**
 * @brief 连接电源网络中的所有层
 * 
 * @param pnp_network 电源网格管理器
 * @param idb_design IDB设计对象
 * @return idb::IdbDesign* 修改后的IDB设计对象
 */
idb::IdbDesign* PowerVia::connectAllPowerLayers(GridManager& pnp_network, idb::IdbDesign* idb_design)
{
  if (!idb_design) {
    std::cout << "Error : Invalid IDB design object" << std::endl;
    return nullptr;
  }
  
  // 连接VDD网络的所有层
  idb_design = connectNetworkLayers(pnp_network, PowerType::kVDD, idb_design);
  if (!idb_design) {
    std::cout << "Error : Failed to connect VDD layers" << std::endl;
    return nullptr;
  }
  
  // 连接VSS网络的所有层
  idb_design = connectNetworkLayers(pnp_network, PowerType::kVSS, idb_design);
  if (!idb_design) {
    std::cout << "Error : Failed to connect VSS layers" << std::endl;
    return nullptr;
  }
  
  std::cout << "Success : Connected all power layers" << std::endl;
  return idb_design;
}

/**
 * @brief 连接指定网络的所有层
 * 
 * @param pnp_network 电源网格管理器
 * @param net_type 网络类型(VDD/VSS)
 * @param idb_design IDB设计对象
 * @return idb::IdbDesign* 修改后的IDB设计对象
 */
idb::IdbDesign* PowerVia::connectNetworkLayers(GridManager& pnp_network, PowerType net_type, idb::IdbDesign* idb_design)
{
  if (!idb_design) {
    std::cout << "Error : Invalid IDB design object" << std::endl;
    return nullptr;
  }
  
  // 获取网络名称
  std::string net_name = (net_type == PowerType::kVDD) ? "VDD" : "VSS";
  
  // 获取电源层列表
  auto power_layers = pnp_network.get_power_layers();
  int layer_count = pnp_network.get_layer_count();
  
  // 连接相邻层
  for (int i = 0; i < layer_count - 1; i++) {
    std::string top_layer = "M" + std::to_string(power_layers[i]);
    std::string bottom_layer = "M" + std::to_string(power_layers[i+1]);
    
    idb_design = connectLayers(net_name, top_layer, bottom_layer, idb_design);
    if (!idb_design) {
      std::cout << "Error : Failed to connect layers " << top_layer << " and " << bottom_layer << std::endl;
      return nullptr;
    }
  }
  
  // 遍历所有区域，处理模板信息
  auto& grid_data = pnp_network.get_grid_data();
  auto& template_data = pnp_network.get_template_data();
  
  for (int layer_idx = 0; layer_idx < layer_count; layer_idx++) {
    for (int row = 0; row < pnp_network.get_ho_region_num(); row++) {
      for (int col = 0; col < pnp_network.get_ver_region_num(); col++) {
        // 获取当前区域模板
        const SingleTemplate& curr_template = template_data[layer_idx][row][col];
        
        // 只处理当前网络类型
        if (curr_template.get_first_stripe_power_type() == net_type) {
          // 添加通孔
          addViasForTemplate(pnp_network, grid_data[layer_idx][row][col], 
                            layer_idx, row, col, net_name, idb_design);
        }
      }
    }
  }
  
  std::cout << "Success : Connected all layers for " << net_name << std::endl;
  return idb_design;
}

/**
 * @brief 将浮点值转换为数据库单位
 * 
 * @param value 浮点值
 * @param idb_design IDB设计对象
 * @return int32_t 数据库单位值
 */
int32_t PowerVia::transUnitDB(double value, idb::IdbDesign* idb_design)
{
  if (!idb_design) return -1;
  auto idb_layout = idb_design->get_layout();
  return idb_layout != nullptr ? idb_layout->transUnitDB(value) : -1;
}

/**
 * @brief 查找或创建通孔
 * 
 * @param layer_cut 切割层
 * @param width_design 宽度
 * @param height_design 高度
 * @param idb_design IDB设计对象
 * @return idb::IdbVia* 通孔对象
 */
idb::IdbVia* PowerVia::findVia(idb::IdbLayerCut* layer_cut, int32_t width_design, int32_t height_design, idb::IdbDesign* idb_design)
{
  if (!idb_design) return nullptr;
  auto via_list = idb_design->get_via_list();

  // 通孔名称格式: 切割层名称_宽度x高度
  std::string via_name = layer_cut->get_name() + "_" + std::to_string(width_design) + "x" + std::to_string(height_design);
  
  // 查找已有通孔
  idb::IdbVia* via_find = via_list->find_via(via_name);
  
  // 如果找不到，创建新通孔
  if (via_find == nullptr) {
    via_find = createVia(layer_cut, width_design, height_design, via_name, idb_design);
  }
  
  return via_find;
}

/**
 * @brief 创建通孔
 * 
 * @param layer_cut 切割层
 * @param width_design 宽度
 * @param height_design 高度
 * @param via_name 通孔名称
 * @param idb_design IDB设计对象
 * @return idb::IdbVia* 通孔对象
 */
idb::IdbVia* PowerVia::createVia(idb::IdbLayerCut* layer_cut, int32_t width_design, int32_t height_design, std::string via_name, idb::IdbDesign* idb_design)
{
  if (!idb_design) return nullptr;
  auto via_list = idb_design->get_via_list();

  // 确保通孔名称格式正确
  via_name = layer_cut->get_name() + "_" + std::to_string(width_design) + "x" + std::to_string(height_design);

  // 创建通孔
  idb::IdbVia* via = via_list->createVia(via_name, layer_cut, width_design, height_design);
  
  // 调试信息
  std::cout << "Created via: " << via_name << ", via_list size: " << via_list->get_via_list().size() << std::endl;
  
  return via;
}

/**
 * @brief 创建通孔线段
 * 
 * @param layer 层
 * @param route_width 路由宽度
 * @param wire_shape_type 线形状类型
 * @param coord 坐标
 * @param via 通孔
 * @return idb::IdbSpecialWireSegment* 线段对象
 */
idb::IdbSpecialWireSegment* PowerVia::createSpecialWireVia(idb::IdbLayer* layer, int32_t route_width, 
                                                         idb::IdbWireShapeType wire_shape_type,
                                                         idb::IdbCoordinate<int32_t>* coord, 
                                                         idb::IdbVia* via)
{
  // 创建特殊线段
  idb::IdbSpecialWireSegment* segment_via = new idb::IdbSpecialWireSegment();
  
  // 设置为通孔类型
  segment_via->set_is_via(true);
  
  // 添加坐标点
  segment_via->add_point(coord->get_x(), coord->get_y());
  
  // 设置层信息
  segment_via->set_layer(layer);
  
  // 设置形状类型
  segment_via->set_shape_type(idb::IdbWireShapeType::kStripe);
  
  // 标记为新层
  segment_via->set_layer_as_new();
  
  // 设置路由宽度
  segment_via->set_route_width(0);
  
  // 复制通孔并设置坐标
  idb::IdbVia* via_new = segment_via->copy_via(via);
  if (via_new != nullptr) {
    via_new->set_coordinate(coord);
  }

  // 设置边界框
  segment_via->set_bounding_box();
  
  return segment_via;
}

/**
 * @brief 计算两个线段的交叉区域
 * 
 * @param segment_top 顶层线段
 * @param segment_bottom 底层线段
 * @param intersection_rect 输出交叉区域
 * @return true 有交叉
 * @return false 无交叉
 */
bool PowerVia::getIntersectCoordinate(idb::IdbSpecialWireSegment* segment_top, 
                                    idb::IdbSpecialWireSegment* segment_bottom,
                                    idb::IdbRect& intersection_rect)
{
  // 获取线段的边界框
  idb::IdbRect* top_bbox = segment_top->get_bounding_box();
  idb::IdbRect* bottom_bbox = segment_bottom->get_bounding_box();
  
  // 检查是否有交叉
  if (!top_bbox->isIntersection(bottom_bbox)) {
    return false;
  }
  
  // 计算交叉区域
  int32_t ll_x = std::max(top_bbox->get_low_x(), bottom_bbox->get_low_x());
  int32_t ll_y = std::max(top_bbox->get_low_y(), bottom_bbox->get_low_y());
  int32_t ur_x = std::min(top_bbox->get_high_x(), bottom_bbox->get_high_x());
  int32_t ur_y = std::min(top_bbox->get_high_y(), bottom_bbox->get_high_y());
  
  // 设置交叉区域
  intersection_rect.set_rect(ll_x, ll_y, ur_x, ur_y);
  
  return true;
}

/**
 * @brief 根据模板信息添加通孔连接
 * 
 * @param pnp_network 电源网格管理器
 * @param region 网格区域
 * @param layer_idx 层索引
 * @param row 行索引
 * @param col 列索引
 * @param net_name 网络名称
 * @param idb_design IDB设计对象
 * @return true 添加成功
 * @return false 添加失败
 */
bool PowerVia::addViasForTemplate(GridManager& pnp_network, 
                                const PDNRectanGridRegion& region,
                                int layer_idx, int row, int col,
                                std::string net_name,
                                idb::IdbDesign* idb_design)
{
  // 获取当前模板信息
  const SingleTemplate& curr_template = pnp_network.get_template_data()[layer_idx][row][col];
  
  // 计算区域范围
  auto left_bottom = region.get_left_bottom_coordinate();
  auto right_top = region.get_right_top_coordinate();
  
  // 获取电源层名称
  std::string layer_name = "M" + std::to_string(pnp_network.get_power_layers()[layer_idx]);
  
  // 如果不是最后一层，添加与下层的连接
  if (layer_idx < pnp_network.get_layer_count() - 1) {
    std::string next_layer = "M" + std::to_string(pnp_network.get_power_layers()[layer_idx + 1]);
    
    // 根据模板方向确定通孔位置
    if (curr_template.get_direction() == StripeDirection::kHorizontal) {
      // 水平方向条纹
      double stripe_y = left_bottom.second + curr_template.get_offset();
      double stripe_width = curr_template.get_width();
      
      // 在水平条纹上均匀分布通孔
      for (double x = left_bottom.first; x <= right_top.first; x += 10.0) {
        addSingleVia(net_name, layer_name, next_layer, x, stripe_y, 
                     transUnitDB(stripe_width, idb_design), 
                     transUnitDB(stripe_width, idb_design), 
                     idb_design);
      }
    } else {
      // 垂直方向条纹
      double stripe_x = left_bottom.first + curr_template.get_offset();
      double stripe_width = curr_template.get_width();
      
      // 在垂直条纹上均匀分布通孔
      for (double y = left_bottom.second; y <= right_top.second; y += 10.0) {
        addSingleVia(net_name, layer_name, next_layer, stripe_x, y, 
                     transUnitDB(stripe_width, idb_design), 
                     transUnitDB(stripe_width, idb_design), 
                     idb_design);
      }
    }
  }
  
  return true;
}

/**
 * @brief 在指定位置添加单个通孔
 * 
 * @param net_name 网络名称
 * @param top_layer 顶层
 * @param bottom_layer 底层
 * @param x 坐标x
 * @param y 坐标y
 * @param width 宽度
 * @param height 高度
 * @param idb_design IDB设计对象
 * @return true 添加成功
 * @return false 添加失败
 */
bool PowerVia::addSingleVia(std::string net_name,
                          std::string top_layer,
                          std::string bottom_layer,
                          double x, double y,
                          int32_t width, int32_t height,
                          idb::IdbDesign* idb_design)
{
  if (!idb_design) return false;
  
  auto idb_layout = idb_design->get_layout();
  auto idb_layer_list = idb_layout->get_layers();
  auto idb_pdn_list = idb_design->get_special_net_list();
  
  // 找到网络
  idb::IdbSpecialNet* net = idb_pdn_list->find_net(net_name);
  if (net == nullptr) {
    std::cout << "Error: Cannot find net " << net_name << std::endl;
    return false;
  }
  
  // 获取线
  idb::IdbSpecialWire* wire = nullptr;
  if (net->get_wire_list()->get_num() > 0) {
    wire = net->get_wire_list()->find_wire(0);
  } else {
    wire = net->get_wire_list()->add_wire(nullptr);
  }
  
  if (wire == nullptr) {
    std::cout << "Error: Cannot get wire for net " << net_name << std::endl;
    return false;
  }
  
  // 找到所有的切割层
  std::vector<idb::IdbLayerCut*> cut_layer_list = 
      idb_layer_list->find_cut_layer_list(top_layer, bottom_layer);
      
  if (cut_layer_list.empty()) {
    std::cout << "Error: No cut layers found between " << top_layer << " and " << bottom_layer << std::endl;
    return false;
  }
  
  // 转换坐标为数据库单位
  int32_t dbu_x = transUnitDB(x, idb_design);
  int32_t dbu_y = transUnitDB(y, idb_design);
  
  // 为每个切割层添加通孔
  for (auto layer_cut : cut_layer_list) {
    if (!layer_cut->is_cut()) continue;
    
    // 查找或创建通孔
    idb::IdbVia* via = findVia(layer_cut, width, height, idb_design);
    if (via == nullptr) {
      std::cout << "Error: Failed to create via for " << layer_cut->get_name() << std::endl;
      continue;
    }
    
    // 获取通孔顶层
    idb::IdbLayer* via_top_layer = via->get_top_layer_shape().get_layer();
    
    // 创建坐标
    idb::IdbCoordinate<int32_t>* coord = new idb::IdbCoordinate<int32_t>(dbu_x, dbu_y);
    
    // 创建通孔线段
    idb::IdbSpecialWireSegment* segment_via = createSpecialWireVia(
        via_top_layer, 0, idb::IdbWireShapeType::kStripe, coord, via);
        
    // 添加到线中
    wire->add_segment(segment_via);
  }
  
  return true;
}

/**
 * @brief 连接两个指定层之间的电源线
 * 
 * @param net_name 网络名称
 * @param top_layer_name 顶层名称
 * @param bottom_layer_name 底层名称
 * @param idb_design IDB设计对象
 * @return idb::IdbDesign* 修改后的IDB设计对象
 */
idb::IdbDesign* PowerVia::connectLayers(std::string net_name, std::string top_layer_name, std::string bottom_layer_name, idb::IdbDesign* idb_design)
{
  if (!idb_design) {
    std::cout << "Error : Invalid IDB design object" << std::endl;
    return nullptr;
  }
  
  auto idb_layout = idb_design->get_layout();
  auto idb_layer_list = idb_layout->get_layers();
  auto idb_pdn_list = idb_design->get_special_net_list();

  // 获取层信息
  idb::IdbLayerRouting* layer_bottom = dynamic_cast<idb::IdbLayerRouting*>(
      idb_layer_list->find_layer(bottom_layer_name));
  idb::IdbLayerRouting* layer_top = dynamic_cast<idb::IdbLayerRouting*>(
      idb_layer_list->find_layer(top_layer_name));

  // 确保层存在且不相同
  if (layer_bottom == nullptr || layer_top == nullptr || layer_bottom == layer_top) {
    std::cout << "Error : layers not exist or same layer." << std::endl;
    return nullptr;
  }
  
  // 确保bottom层在top层下面
  if (layer_top->get_order() < layer_bottom->get_order()) {
    std::swap(layer_top, layer_bottom);
  }
  
  // 不支持同方向的两层
  if ((layer_top->is_horizontal() && layer_bottom->is_horizontal()) || 
      (layer_top->is_vertical() && layer_bottom->is_vertical())) {
    std::cout << "Error : layers have the same direction." << std::endl;
    return nullptr;
  }

  // 获取网络
  idb::IdbSpecialNet* net = idb_pdn_list->find_net(net_name);
  if (net == nullptr) {
    std::cout << "Error : can't find the net " << net_name << std::endl;
    return nullptr;
  }
  
  // 获取网络的线列表
  idb::IdbSpecialWireList* wire_list = net->get_wire_list();
  if (wire_list == nullptr) {
    std::cout << "Error : not wire in Special net " << net_name << std::endl;
    return nullptr;
  }
  
  // 对每条线处理
  for (idb::IdbSpecialWire* wire : wire_list->get_wire_list()) {
    // 收集顶层和底层的线段
    std::vector<idb::IdbSpecialWireSegment*> segment_list_top;
    std::vector<idb::IdbSpecialWireSegment*> segment_list_bottom;
    
    for (idb::IdbSpecialWireSegment* segment : wire->get_segment_list()) {
      if (segment->is_tripe() || segment->is_follow_pin()) {
        if (segment->get_layer()->compareLayer(layer_top)) {
          segment_list_top.emplace_back(segment);
        }

        if (segment->get_layer()->compareLayer(layer_bottom)) {
          segment_list_bottom.emplace_back(segment);
        }
      }
    }

    // 如果没有找到线段，跳过
    if (segment_list_top.size() <= 0 || segment_list_bottom.size() <= 0) {
      continue;
    }

    // 对每个顶层线段
    for (idb::IdbSpecialWireSegment* segment_top : segment_list_top) {
      // 对每个底层线段
      for (idb::IdbSpecialWireSegment* segment_bottom : segment_list_bottom) {
        // 计算交叉区域
        idb::IdbRect intersection_rect;
        if (getIntersectCoordinate(segment_top, segment_bottom, intersection_rect)) {
          // 调试信息
          std::cout << "Found intersection at (" << intersection_rect.get_middle_point().get_x() 
                    << ", " << intersection_rect.get_middle_point().get_y() 
                    << ") with size " << intersection_rect.get_width() << "x" << intersection_rect.get_height() << std::endl;
                    
          // 对每个中间层添加通孔
          for (int32_t layer_order = layer_bottom->get_order(); 
               layer_order <= (layer_top->get_order() - 2);) {
            // 获取切割层
            idb::IdbLayerCut* layer_cut_find = dynamic_cast<idb::IdbLayerCut*>(
                idb_layer_list->find_layer_by_order(layer_order + 1));
                
            if (layer_cut_find == nullptr) {
              std::cout << "Error : layer input illegal." << std::endl;
              return nullptr;
            }
            
            // 查找或创建通孔
            idb::IdbVia* via_find = findVia(layer_cut_find, 
                                           intersection_rect.get_width(), 
                                           intersection_rect.get_height(),
                                           idb_design);
                                           
            if (via_find == nullptr) {
              std::cout << "Error : can not find VIA matchs." << std::endl;
              continue;
            }
            
            // 获取通孔顶层
            idb::IdbLayer* layer_top_via = via_find->get_top_layer_shape().get_layer();
            
            // 创建坐标
            idb::IdbCoordinate<int32_t> middle = intersection_rect.get_middle_point();
            idb::IdbCoordinate<int32_t>* middle_ptr = new idb::IdbCoordinate<int32_t>(middle.get_x(), middle.get_y());
            
            // 创建通孔线段
            idb::IdbSpecialWireSegment* segment_via = createSpecialWireVia(
                layer_top_via, 0, idb::IdbWireShapeType::kStripe, middle_ptr, via_find);
            
            // 添加到线中
            wire->add_segment(segment_via);
            
            // 移动到下一个切割层
            layer_order += 2;
          }
        }
      }
    }
  }

  std::cout << "Success : connectLayers " << top_layer_name << " & " << bottom_layer_name << std::endl;
  return idb_design;
}

}  // namespace ipnp 