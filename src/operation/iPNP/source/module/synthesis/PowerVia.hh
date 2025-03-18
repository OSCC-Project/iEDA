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
 * @file PowerVia.hh
 * @author Jianrong Su
 * @brief 处理电源网络中的通孔连接
 * @version 0.1
 * @date 2025-03-12
 */

#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "GridManager.hh"
#include "iPNPCommon.hh"

namespace idb {
  class IdbDesign;
  class IdbSpecialNet;
  class IdbSpecialNetList;
  class IdbSpecialWireList;
  class IdbSpecialWire;
  class IdbSpecialWireSegment;
  class IdbLayer;
  class IdbLayerCut;
  class IdbLayerRouting;
  class IdbVia;
  class IdbPin;
  class IdbRect;
  class IdbInstance;

  enum class SegmentType : int8_t;
  enum class IdbWireShapeType : uint8_t;
  enum class IdbOrient : uint8_t;

  template <typename T>
  class IdbCoordinate;
}  // namespace idb

namespace ipnp {

  /**
   * @brief 处理电源网络中的通孔连接
   *
   * 该类负责在电源线交叉处添加通孔，连接不同金属层的电源线
   */
  class PowerVia
  {
  public:
    PowerVia() = default;
    ~PowerVia() = default;

    /**
     * @brief 连接电源网络中的所有层
     *
     * @param pnp_network 电源网格管理器
     * @param idb_design IDB设计对象
     * @return idb::IdbDesign* 修改后的IDB设计对象
     */
    idb::IdbDesign* connectAllPowerLayers(GridManager& pnp_network, idb::IdbDesign* idb_design);

  private:
    /**
     * @brief 连接指定网络的所有层
     *
     * @param pnp_network 电源网格管理器
     * @param net_type 网络类型(VDD/VSS)
     * @param idb_design IDB设计对象
     * @return idb::IdbDesign* 修改后的IDB设计对象
     */
    idb::IdbDesign* connectNetworkLayers(GridManager& pnp_network, PowerType net_type, idb::IdbDesign* idb_design);

    /**
     * @brief 连接两个指定层之间的电源线
     *
     * @param net_name 网络名称
     * @param top_layer_name 顶层名称
     * @param bottom_layer_name 底层名称
     * @param idb_design IDB设计对象
     * @return idb::IdbDesign* 修改后的IDB设计对象
     */
    idb::IdbDesign* connectLayers(std::string net_name, std::string top_layer_name, std::string bottom_layer_name, idb::IdbDesign* idb_design);

    /**
     * @brief 将浮点值转换为数据库单位
     *
     * @param value 浮点值
     * @param idb_design IDB设计对象
     * @return int32_t 数据库单位值
     */
    int32_t transUnitDB(double value, idb::IdbDesign* idb_design);

    /**
     * @brief 查找或创建通孔
     *
     * @param layer_cut 切割层
     * @param width_design 宽度
     * @param height_design 高度
     * @param idb_design IDB设计对象
     * @return idb::IdbVia* 通孔对象
     */
    idb::IdbVia* findVia(idb::IdbLayerCut* layer_cut, int32_t width_design, int32_t height_design, idb::IdbDesign* idb_design);

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
    idb::IdbVia* createVia(idb::IdbLayerCut* layer_cut, int32_t width_design, int32_t height_design, std::string via_name, idb::IdbDesign* idb_design);

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
    idb::IdbSpecialWireSegment* createSpecialWireVia(idb::IdbLayer* layer, int32_t route_width,
      idb::IdbWireShapeType wire_shape_type,
      idb::IdbCoordinate<int32_t>* coord,
      idb::IdbVia* via);

    /**
     * @brief 计算两个线段的交叉区域
     *
     * @param segment_top 顶层线段
     * @param segment_bottom 底层线段
     * @param intersection_rect 输出交叉区域
     * @return true 有交叉
     * @return false 无交叉
     */
    bool getIntersectCoordinate(idb::IdbSpecialWireSegment* segment_top,
      idb::IdbSpecialWireSegment* segment_bottom,
      idb::IdbRect& intersection_rect);

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
    bool addViasForTemplate(GridManager& pnp_network,
      const PDNRectanGridRegion& region,
      int layer_idx, int row, int col,
      std::string net_name,
      idb::IdbDesign* idb_design);

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
    bool addSingleVia(std::string net_name,
      std::string top_layer,
      std::string bottom_layer,
      double x, double y,
      int32_t width, int32_t height,
      idb::IdbDesign* idb_design);
  };

}  // namespace ipnp 