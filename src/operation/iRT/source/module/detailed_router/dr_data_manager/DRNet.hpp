#pragma once

#include "ConnectType.hpp"
#include "DRBox.hpp"
#include "DRPin.hpp"
#include "GridMap.hpp"
#include "Guide.hpp"
#include "MTree.hpp"
#include "Net.hpp"
#include "PHYNode.hpp"
#include "Pin.hpp"
#include "TNode.hpp"

namespace irt {

class DRNet
{
 public:
  DRNet() = default;
  ~DRNet() = default;
  // getter
  Net* get_origin_net() { return _origin_net; }
  irt_int get_net_idx() const { return _net_idx; }
  ConnectType get_connect_type() const { return _connect_type; }
  std::vector<DRPin>& get_dr_pin_list() { return _dr_pin_list; }
  MTree<RTNode>& get_ta_result_tree() { return _ta_result_tree; }
  MTree<RTNode>& get_dr_result_tree() { return _dr_result_tree; }
  // setter
  void set_origin_net(Net* origin_net) { _origin_net = origin_net; }
  void set_net_idx(const irt_int net_idx) { _net_idx = net_idx; }
  void set_connect_type(const ConnectType& connect_type) { _connect_type = connect_type; };
  void set_dr_pin_list(const std::vector<DRPin>& dr_pin_list) { _dr_pin_list = dr_pin_list; }
  void set_ta_result_tree(const MTree<RTNode>& ta_result_tree) { _ta_result_tree = ta_result_tree; }
  void set_dr_result_tree(const MTree<RTNode>& dr_result_tree) { _dr_result_tree = dr_result_tree; }

 private:
  Net* _origin_net = nullptr;
  irt_int _net_idx = -1;
  ConnectType _connect_type = ConnectType::kNone;
  std::vector<DRPin> _dr_pin_list;
  MTree<RTNode> _ta_result_tree;
  MTree<RTNode> _dr_result_tree;
};

}  // namespace irt
