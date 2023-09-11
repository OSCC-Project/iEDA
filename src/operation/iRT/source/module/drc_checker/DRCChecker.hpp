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
#pragma once

#include "ChangeType.hpp"
#include "DRCRect.hpp"
#include "DataManager.hpp"
#include "RTAPI.hpp"
#include "RegionQuery.hpp"
#include "ViolationInfo.hpp"

namespace irt {

#define DC_INST (irt::DRCChecker::getInst())

class DRCChecker
{
 public:
  static void initInst();
  static DRCChecker& getInst();
  static void destroyInst();
  // function
  /**
   * 获得DRCRectList
   * 将多种类型的结果转换为DRCRectList
   */
  std::vector<DRCRect> getDRCRectList(irt_int net_idx, std::vector<Segment<LayerCoord>>& segment_list);
  std::vector<DRCRect> getDRCRectList(irt_int net_idx, Segment<LayerCoord>& segment);
  std::vector<DRCRect> getDRCRectList(irt_int net_idx, MTree<LayerCoord>& coord_tree);
  std::vector<DRCRect> getDRCRectList(irt_int net_idx, MTree<PHYNode>& phy_node_tree);
  std::vector<DRCRect> getDRCRectList(irt_int net_idx, PHYNode& phy_node);
  /**
   * 返回RegionQuery中的形状map
   */
  std::map<irt_int, std::map<irt_int, std::set<LayerRect, CmpLayerRectByXASC>>>& getLayerNetRectMap(RegionQuery& region_query,
                                                                                                    bool is_routing);
  /**
   * 更新RegionQuery的DRCRectList
   */
  void updateRectList(RegionQuery& region_query, ChangeType change_type, const DRCRect& drc_rect);
  void updateRectList(RegionQuery& region_query, ChangeType change_type, const std::vector<DRCRect>& drc_rect_list);
  /**
   * 碰撞一定会产生DRC的最小膨胀矩形
   * 注意：现在只能扩大spacing的最小范围，其他的由于在布线过程中可能有误差，比如在没连上的线进行eol规则的扩大可能会有问题
   */
  std::vector<LayerRect> getMinScope(const DRCRect& drc_rect);
  std::vector<LayerRect> getMinScope(const std::vector<DRCRect>& drc_rect_list);
  /**
   * 碰撞可能会产生DRC的最大膨胀矩形
   */
  std::vector<LayerRect> getMaxScope(const DRCRect& drc_rect);
  std::vector<LayerRect> getMaxScope(const std::vector<DRCRect>& drc_rect_list);
  /**
   * 获得违例信息
   * 暂时不进行线网自检
   */
  std::map<std::string, std::vector<ViolationInfo>> getViolationInfo(RegionQuery& region_query, const DRCRect& drc_rect);
  std::map<std::string, std::vector<ViolationInfo>> getViolationInfo(RegionQuery& region_query, const std::vector<DRCRect>& drc_rect_list);
  std::map<std::string, std::vector<ViolationInfo>> getViolationInfo(RegionQuery& region_query);

 private:
  // self
  static DRCChecker* _dc_instance;

  DRCChecker() = default;
  DRCChecker(const DRCChecker& other) = delete;
  DRCChecker(DRCChecker&& other) = delete;
  ~DRCChecker() = default;
  DRCChecker& operator=(const DRCChecker& other) = delete;
  DRCChecker& operator=(DRCChecker&& other) = delete;
  // function
  std::vector<ids::DRCRect> convertToIDSRect(const std::vector<DRCRect>& drc_rect_list);
  void addNetRectMap(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list);
  void addEnvRectListByRTDRC(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list);
  void delNetRectMap(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list);
  void delEnvRectListByRTDRC(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list);
  RQShape convertToRQShape(const DRCRect& drc_rect);
  bool checkMinSpacingByRTDRC(RQShape& net_shape1, RQShape& net_shape2, std::vector<RQShape>& net_shape_list);
  std::vector<LayerRect> getMinSpacingRect(const std::vector<ids::DRCRect>& drc_rect_list);
#if 1  // violation info
  void checkMinSpacingByOther(RegionQuery* region_query, const DRCRect& drc_rect_list, std::vector<ViolationInfo>& violation_info_list);
  void checkMinSpacingByOther(RegionQuery* region_query, const std::vector<DRCRect>& drc_rect_list,
                              std::vector<ViolationInfo>& violation_info_list);
  void uniqueViolationInfoList(std::vector<ViolationInfo>& violation_info_list);
  void checkMinSpacingBySelf(RegionQuery* region_query, std::vector<ViolationInfo>& violation_info_list);
  void checkMinSpacingBySelf(const std::vector<DRCRect>& drc_rect_list, std::vector<ViolationInfo>& violation_info_list);
  void checkMinSpacingBySelf(std::map<irt_int, std::vector<RQShape>>& net_shape_map, std::vector<ViolationInfo>& violation_info_list);

#endif
};
}  // namespace irt
