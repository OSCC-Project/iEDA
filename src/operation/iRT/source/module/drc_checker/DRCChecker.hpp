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
#include "DRCShape.hpp"
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
   * 获得DRCShapeList
   * 将多种类型的结果转换为DRCShapeList
   */
  std::vector<DRCShape> getDRCShapeList(irt_int net_idx, std::vector<Segment<LayerCoord>>& segment_list);
  std::vector<DRCShape> getDRCShapeList(irt_int net_idx, Segment<LayerCoord>& segment);
  std::vector<DRCShape> getDRCShapeList(irt_int net_idx, MTree<LayerCoord>& coord_tree);
  std::vector<DRCShape> getDRCShapeList(irt_int net_idx, MTree<PhysicalNode>& physical_node_tree);
  std::vector<DRCShape> getDRCShapeList(irt_int net_idx, PhysicalNode& physical_node);
  /**
   * 返回RegionQuery中的形状map
   */
  std::map<irt_int, std::map<BaseInfo, std::set<LayerRect, CmpLayerRectByXASC>, CmpBaseInfo>>& getLayerInfoRectMap(
      RegionQuery& region_query, bool is_routing);
  /**
   * 更新RegionQuery的DRCShapeList
   */
  void updateRectList(RegionQuery& region_query, ChangeType change_type, const DRCShape& drc_shape);
  void updateRectList(RegionQuery& region_query, ChangeType change_type, const std::vector<DRCShape>& drc_shape_list);
  /**
   * 碰撞一定会产生DRC的最小膨胀矩形
   * 注意：现在只能扩大spacing的最小范围，其他的由于在布线过程中可能有误差，比如在没连上的线进行eol规则的扩大可能会有问题
   */
  std::vector<LayerRect> getMinScope(const DRCShape& drc_shape);
  std::vector<LayerRect> getMinScope(const std::vector<DRCShape>& drc_shape_list);
  /**
   * 碰撞可能会产生DRC的最大膨胀矩形
   */
  std::vector<LayerRect> getMaxScope(const DRCShape& drc_shape);
  std::vector<LayerRect> getMaxScope(const std::vector<DRCShape>& drc_shape_list);
  /**
   * drc_shape_list在region_query的环境里产生的违例信息，如spacing
   * 关注于非同net之间的违例
   */
  std::map<std::string, std::vector<ViolationInfo>> getEnvViolationInfo(RegionQuery& region_query,
                                                                        const std::vector<DRCCheckType>& check_type_list,
                                                                        const std::vector<DRCShape>& drc_shape_list);
  /**
   * drc_shape_list组成的自身违例信息，如min_area,min_step
   * 关注于net内的违例
   */
  std::map<std::string, std::vector<ViolationInfo>> getSelfViolationInfo(const std::vector<DRCCheckType>& check_type_list,
                                                                         const std::vector<DRCShape>& drc_shape_list);

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
  std::map<std::string, std::vector<ViolationInfo>> getEnvViolationInfoByRT(RegionQuery& region_query,
                                                                            const std::vector<DRCCheckType>& check_type_list,
                                                                            const std::vector<DRCShape>& drc_shape_list);
  std::map<std::string, std::vector<ViolationInfo>> getEnvViolationInfoByiDRC(RegionQuery& region_query,
                                                                              const std::vector<DRCCheckType>& check_type_list,
                                                                              const std::vector<DRCShape>& drc_shape_list);
  BaseShape convert(const DRCShape& drc_shape);
  ViolationInfo convert(BaseViolationInfo& base_violation_info);
  std::map<std::string, std::vector<ViolationInfo>> getSelfViolationInfoByRT(const std::vector<DRCCheckType>& check_type_list,
                                                                             const std::vector<DRCShape>& drc_shape_list);
  std::map<std::string, std::vector<ViolationInfo>> getSelfViolationInfoByiDRC(const std::vector<DRCCheckType>& check_type_list,
                                                                               const std::vector<DRCShape>& drc_shape_list);
  void addEnvRectList(RegionQuery* region_query, const std::vector<DRCShape>& drc_shape_list);
  void delEnvRectList(RegionQuery* region_query, const std::vector<DRCShape>& drc_shape_list);
  BaseShape convertToBaseShape(const DRCShape& drc_shape);
  std::vector<LayerRect> getMinSpacingRect(const std::vector<DRCShape>& drc_shape_list);
  std::vector<LayerRect> getMaxSpacingRect(const std::vector<DRCShape>& drc_shape_list);
#if 1  // violation info
  void checkMinSpacingByOther(RegionQuery* region_query, const DRCShape& drc_shape_list, std::vector<ViolationInfo>& violation_info_list);
  void checkMinSpacingByOther(RegionQuery* region_query, const std::vector<DRCShape>& drc_shape_list,
                              std::vector<ViolationInfo>& violation_info_list);
  void checkMinArea(const std::vector<DRCShape>& drc_shape_list, std::vector<ViolationInfo>& violation_info_list);
  void uniqueViolationInfoList(std::vector<ViolationInfo>& violation_info_list);
#endif
};
}  // namespace irt
