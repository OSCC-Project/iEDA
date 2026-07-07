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

#include <limits>

#include "RTHeader.hpp"
#include "Utility.hpp"
#include "ViaMasterIdx.hpp"

namespace irt {

#if 1  // astar
enum class PANodeState
{
  kNone = 0,
  kOpen = 1,
  kClose = 2
};
#endif

class PANode : public LayerCoord
{
 public:
  using OrientNetList = std::vector<std::pair<Orientation, std::vector<int32_t>>>;

  PANode() = default;
  ~PANode() = default;
  // getter
  OrientNetList& get_orient_fixed_rect_list() { return _orient_fixed_rect_list; }
  OrientNetList& get_orient_routed_rect_list() { return _orient_routed_rect_list; }
  // setter
  void set_neighbor_node_map(const std::map<Orientation, PANode*>& neighbor_node_map)
  {
    clearNeighborNodeList();
    for (auto& [orientation, neighbor_node] : neighbor_node_map) {
      setNeighborNode(orientation, neighbor_node);
    }
  }
  void set_orient_fixed_rect_map(const std::map<Orientation, std::set<int32_t>>& orient_fixed_rect_map)
  {
    setOrientNetList(_orient_fixed_rect_list, _orient_fixed_rect_net_idx_list, orient_fixed_rect_map);
  }
  void set_orient_routed_rect_map(const std::map<Orientation, std::set<int32_t>>& orient_routed_rect_map)
  {
    setOrientNetList(_orient_routed_rect_list, _orient_routed_rect_net_idx_list, orient_routed_rect_map);
  }
  void set_orient_violation_number_map(const std::map<Orientation, int32_t>& orient_violation_number_map)
  {
    _orient_violation_number_list.fill(0);
    for (auto& [orientation, violation_number] : orient_violation_number_map) {
      if (isNeighborOrientation(orientation)) {
        _orient_violation_number_list[getNeighborOrientationIdx(orientation)] = violation_number;
      }
    }
  }
  // function
  void setNeighborNode(Orientation orientation, PANode* neighbor_node)
  {
    if (!isNeighborOrientation(orientation)) {
      RTLOG.error(Loc::current(), "The neighbor orientation is invalid!");
      return;
    }
    PANode*& exist_neighbor_node = _neighbor_node_list[getNeighborOrientationIdx(orientation)];
    if (exist_neighbor_node == nullptr && neighbor_node != nullptr) {
      _neighbor_node_num++;
    } else if (exist_neighbor_node != nullptr && neighbor_node == nullptr) {
      _neighbor_node_num--;
    }
    exist_neighbor_node = neighbor_node;
  }
  PANode* getNeighborNode(Orientation orientation) const
  {
    if (!isNeighborOrientation(orientation)) {
      return nullptr;
    }
    return _neighbor_node_list[getNeighborOrientationIdx(orientation)];
  }
  int32_t get_neighbor_node_num() const { return _neighbor_node_num; }
  bool hasNeighborNode(Orientation orientation) const { return getNeighborNode(orientation) != nullptr; }
  bool hasAnyNeighborNode() const { return _neighbor_node_num > 0; }
  template <typename Func>
  void forEachNeighborNode(Func func) const
  {
    for (int32_t orient_idx = static_cast<int32_t>(Orientation::kEast); orient_idx <= static_cast<int32_t>(Orientation::kBelow); orient_idx++) {
      Orientation orientation = static_cast<Orientation>(orient_idx);
      PANode* neighbor_node = _neighbor_node_list[getNeighborOrientationIdx(orientation)];
      if (neighbor_node != nullptr) {
        func(orientation, neighbor_node);
      }
    }
  }
  double getFixedRectCost(int32_t net_idx, Orientation orientation, double fixed_rect_unit)
  {
    return getOrientNetCost(_orient_fixed_rect_net_idx_list, net_idx, orientation, fixed_rect_unit);
  }
  double getRoutedRectCost(int32_t net_idx, Orientation orientation, double routed_rect_unit)
  {
    return getOrientNetCost(_orient_routed_rect_net_idx_list, net_idx, orientation, routed_rect_unit);
  }
  double getViolationCost(Orientation orientation, double violation_unit)
  {
    double cost = 0;
    if (getViolationNumber(orientation) > 0) {
      cost = violation_unit;
    }
    return cost;
  }
  void addFixedRectNet(Orientation orientation, int32_t net_idx) { addOrientNet(_orient_fixed_rect_list, _orient_fixed_rect_net_idx_list, orientation, net_idx); }
  void addRoutedRectNet(Orientation orientation, int32_t net_idx) { addOrientNet(_orient_routed_rect_list, _orient_routed_rect_net_idx_list, orientation, net_idx); }
  void delFixedRectNet(Orientation orientation, int32_t net_idx) { delOrientNet(_orient_fixed_rect_list, _orient_fixed_rect_net_idx_list, orientation, net_idx); }
  void delRoutedRectNet(Orientation orientation, int32_t net_idx) { delOrientNet(_orient_routed_rect_list, _orient_routed_rect_net_idx_list, orientation, net_idx); }
  bool hasFixedRectOrient(Orientation orientation) const { return getOrientNet(_orient_fixed_rect_list, orientation) != nullptr; }
  bool hasOrientFixedRect() const { return hasOrientNet(_orient_fixed_rect_net_idx_list); }
  bool hasOrientRoutedRect() const { return hasOrientNet(_orient_routed_rect_net_idx_list); }
  bool hasOrientViolationNumber() const
  {
    for (int32_t violation_number : _orient_violation_number_list) {
      if (violation_number != 0) {
        return true;
      }
    }
    return false;
  }
  void addViolationNumber(Orientation orientation)
  {
    if (isNeighborOrientation(orientation)) {
      _orient_violation_number_list[getNeighborOrientationIdx(orientation)]++;
    }
  }
  int32_t getViolationNumber(Orientation orientation) const
  {
    if (!isNeighborOrientation(orientation)) {
      return 0;
    }
    return _orient_violation_number_list[getNeighborOrientationIdx(orientation)];
  }
  template <typename Func>
  void forEachViolationNumber(Func func) const
  {
    for (int32_t orient_idx = static_cast<int32_t>(Orientation::kEast); orient_idx <= static_cast<int32_t>(Orientation::kBelow); orient_idx++) {
      Orientation orientation = static_cast<Orientation>(orient_idx);
      int32_t violation_number = getViolationNumber(orientation);
      if (violation_number != 0) {
        func(orientation, violation_number);
      }
    }
  }
#if 1  // astar
  // single path
  PANodeState& get_state() { return _state; }
  PANode* get_parent_node() const { return _parent_node; }
  ViaMasterIdx& get_parent_via_master_idx() { return _parent_via_master_idx; }
  double get_known_cost() const { return _known_cost; }
  double get_estimated_cost() const { return _estimated_cost; }
  void set_state(PANodeState state) { _state = state; }
  void set_parent_node(PANode* parent_node) { _parent_node = parent_node; }
  void set_parent_via_master_idx(const ViaMasterIdx& parent_via_master_idx) { _parent_via_master_idx = parent_via_master_idx; }
  void set_known_cost(const double known_cost) { _known_cost = known_cost; }
  void set_estimated_cost(const double estimated_cost) { _estimated_cost = estimated_cost; }
  // function
  bool isNone() { return _state == PANodeState::kNone; }
  bool isOpen() { return _state == PANodeState::kOpen; }
  bool isClose() { return _state == PANodeState::kClose; }
  double getTotalCost() { return (_known_cost + _estimated_cost); }
#endif

 private:
  static constexpr int32_t kNeighborOrientationNum = static_cast<int32_t>(Orientation::kBelow) - static_cast<int32_t>(Orientation::kEast) + 1;
  static constexpr int32_t kNoOrientNetIdx = std::numeric_limits<int32_t>::min();
  static constexpr int32_t kManyOrientNetIdx = kNoOrientNetIdx + 1;
  static constexpr std::array<int32_t, kNeighborOrientationNum> makeNoOrientNetIdxList()
  {
    std::array<int32_t, kNeighborOrientationNum> orient_net_idx_list = {};
    orient_net_idx_list.fill(kNoOrientNetIdx);
    return orient_net_idx_list;
  }
  static constexpr int32_t getNeighborOrientationIdx(Orientation orientation)
  {
    return static_cast<int32_t>(orientation) - static_cast<int32_t>(Orientation::kEast);
  }
  static constexpr bool isNeighborOrientation(Orientation orientation)
  {
    return (static_cast<int32_t>(Orientation::kEast) <= static_cast<int32_t>(orientation)
            && static_cast<int32_t>(orientation) <= static_cast<int32_t>(Orientation::kBelow));
  }
  void clearNeighborNodeList()
  {
    _neighbor_node_list.fill(nullptr);
    _neighbor_node_num = 0;
  }
  static size_t getOrientLowerIdx(const OrientNetList& orient_net_list, Orientation orientation)
  {
    size_t lower_idx = 0;
    while (lower_idx < orient_net_list.size() && static_cast<int32_t>(orient_net_list[lower_idx].first) < static_cast<int32_t>(orientation)) {
      lower_idx++;
    }
    return lower_idx;
  }
  static std::vector<int32_t>* getOrientNet(OrientNetList& orient_net_list, Orientation orientation)
  {
    size_t lower_idx = getOrientLowerIdx(orient_net_list, orientation);
    if (lower_idx < orient_net_list.size() && orient_net_list[lower_idx].first == orientation) {
      return &orient_net_list[lower_idx].second;
    }
    return nullptr;
  }
  static const std::vector<int32_t>* getOrientNet(const OrientNetList& orient_net_list, Orientation orientation)
  {
    size_t lower_idx = getOrientLowerIdx(orient_net_list, orientation);
    if (lower_idx < orient_net_list.size() && orient_net_list[lower_idx].first == orientation) {
      return &orient_net_list[lower_idx].second;
    }
    return nullptr;
  }
  static std::vector<int32_t>& getOrCreateOrientNet(OrientNetList& orient_net_list, Orientation orientation)
  {
    size_t lower_idx = getOrientLowerIdx(orient_net_list, orientation);
    if (lower_idx == orient_net_list.size() || orient_net_list[lower_idx].first != orientation) {
      auto iter = orient_net_list.emplace(orient_net_list.begin() + lower_idx, orientation, std::vector<int32_t>());
      lower_idx = iter - orient_net_list.begin();
    }
    return orient_net_list[lower_idx].second;
  }
  static size_t getNetLowerIdx(const std::vector<int32_t>& net_list, int32_t net_idx)
  {
    size_t left = 0;
    size_t right = net_list.size();
    while (left < right) {
      size_t mid = (left + right) / 2;
      if (net_list[mid] < net_idx) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    return left;
  }
  static bool existNet(const std::vector<int32_t>& net_list, int32_t net_idx)
  {
    size_t lower_idx = getNetLowerIdx(net_list, net_idx);
    return lower_idx < net_list.size() && net_list[lower_idx] == net_idx;
  }
  static void updateOrientNetIdx(std::array<int32_t, kNeighborOrientationNum>& orient_net_idx_list, Orientation orientation,
                                 const std::vector<int32_t>& net_list)
  {
    int32_t& orient_net_idx = orient_net_idx_list[getNeighborOrientationIdx(orientation)];
    if (net_list.empty()) {
      orient_net_idx = kNoOrientNetIdx;
    } else if (net_list.size() == 1) {
      orient_net_idx = net_list.front();
    } else {
      orient_net_idx = kManyOrientNetIdx;
    }
  }
  static void addOrientNet(OrientNetList& orient_net_list, std::array<int32_t, kNeighborOrientationNum>& orient_net_idx_list, Orientation orientation,
                           int32_t net_idx)
  {
    if (!isNeighborOrientation(orientation)) {
      return;
    }
    std::vector<int32_t>& net_list = getOrCreateOrientNet(orient_net_list, orientation);
    size_t lower_idx = getNetLowerIdx(net_list, net_idx);
    if (lower_idx == net_list.size() || net_list[lower_idx] != net_idx) {
      net_list.insert(net_list.begin() + lower_idx, net_idx);
      updateOrientNetIdx(orient_net_idx_list, orientation, net_list);
    }
  }
  static void delOrientNet(OrientNetList& orient_net_list, std::array<int32_t, kNeighborOrientationNum>& orient_net_idx_list, Orientation orientation,
                           int32_t net_idx)
  {
    std::vector<int32_t>* net_list = getOrientNet(orient_net_list, orientation);
    if (net_list == nullptr) {
      return;
    }
    size_t lower_idx = getNetLowerIdx(*net_list, net_idx);
    if (lower_idx < net_list->size() && (*net_list)[lower_idx] == net_idx) {
      net_list->erase(net_list->begin() + lower_idx);
      updateOrientNetIdx(orient_net_idx_list, orientation, *net_list);
    }
  }
  static bool hasOrientNet(const std::array<int32_t, kNeighborOrientationNum>& orient_net_idx_list)
  {
    for (int32_t orient_net_idx : orient_net_idx_list) {
      if (orient_net_idx != kNoOrientNetIdx) {
        return true;
      }
    }
    return false;
  }
  static void setOrientNetList(OrientNetList& orient_net_list, std::array<int32_t, kNeighborOrientationNum>& orient_net_idx_list,
                               const std::map<Orientation, std::set<int32_t>>& orient_net_map)
  {
    orient_net_list.clear();
    orient_net_idx_list.fill(kNoOrientNetIdx);
    orient_net_list.reserve(orient_net_map.size());
    for (auto& [orientation, net_set] : orient_net_map) {
      if (!isNeighborOrientation(orientation)) {
        continue;
      }
      orient_net_list.emplace_back(orientation, std::vector<int32_t>(net_set.begin(), net_set.end()));
      updateOrientNetIdx(orient_net_idx_list, orientation, orient_net_list.back().second);
    }
  }
  static double getOrientNetCost(const std::array<int32_t, kNeighborOrientationNum>& orient_net_idx_list, int32_t net_idx, Orientation orientation,
                                 double unit)
  {
    if (!isNeighborOrientation(orientation)) {
      return 0;
    }
    int32_t orient_net_idx = orient_net_idx_list[getNeighborOrientationIdx(orientation)];
    if (orient_net_idx == kNoOrientNetIdx) {
      return 0;
    }
    if (orient_net_idx == kManyOrientNetIdx) {
      return unit;
    }
    return orient_net_idx == net_idx ? 0 : unit;
  }

  std::array<PANode*, kNeighborOrientationNum> _neighbor_node_list = {};
  int32_t _neighbor_node_num = 0;
  // obstacle & pin_shape
  OrientNetList _orient_fixed_rect_list;
  std::array<int32_t, kNeighborOrientationNum> _orient_fixed_rect_net_idx_list = makeNoOrientNetIdxList();
  // net_result
  OrientNetList _orient_routed_rect_list;
  std::array<int32_t, kNeighborOrientationNum> _orient_routed_rect_net_idx_list = makeNoOrientNetIdxList();
  // violation
  std::array<int32_t, kNeighborOrientationNum> _orient_violation_number_list = {};
#if 1  // astar
  // single path
  PANodeState _state = PANodeState::kNone;
  PANode* _parent_node = nullptr;
  ViaMasterIdx _parent_via_master_idx;
  double _known_cost = 0.0;  // include curr
  double _estimated_cost = 0.0;
#endif
};

#if 1  // astar
struct CmpPANodeCost
{
  bool operator()(PANode* a, PANode* b)
  {
    if (RTUTIL.equalDoubleByError(a->getTotalCost(), b->getTotalCost(), RT_ERROR)) {
      if (RTUTIL.equalDoubleByError(a->get_estimated_cost(), b->get_estimated_cost(), RT_ERROR)) {
        return a->get_neighbor_node_num() < b->get_neighbor_node_num();
      } else {
        return a->get_estimated_cost() > b->get_estimated_cost();
      }
    } else {
      return a->getTotalCost() > b->getTotalCost();
    }
  }
};
#endif

}  // namespace irt
