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
#ifndef SRC_EVALUATOR_SOURCE_WIRELENGTH_WIRELENGTHEVAL_HPP_
#define SRC_EVALUATOR_SOURCE_WIRELENGTH_WIRELENGTHEVAL_HPP_

#include "WLFactory.hpp"
#include "WLNet.hpp"
#include "idm.h"

namespace eval {

class WirelengthEval
{
 public:
  WirelengthEval();
  ~WirelengthEval() = default;

  void initWLNetList();
  int64_t evalTotalWL(WIRELENGTH_TYPE wl_type);
  int64_t evalTotalWL(const std::string& wl_type);

  void reportWirelength(const std::string& plot_path, const std::string& input_def_name);
  void checkWLType(const std::string& wl_type);

  int64_t evalOneNetWL(const std::string& net_name, const std::string& wl_type);
  int64_t evalOneNetWL(const std::string& wl_type, WLNet* wl_net);
  int64_t evalDriver2LoadWL(const std::string& net_name, const std::string& sink_pin_name);
  int64_t evalDriver2LoadWL(WLNet* wl_net, const std::string& sink_pin_name);
  // int64_t evalDriver2LoadWL(WLNet* wl_net, const std::string& sink_pin_name);

  std::map<std::string, int64_t> getName2WLmap(const std::string& wl_type);

  void set_net_list(const std::vector<WLNet*>& net_list) { _net_list = net_list; }
  void set_name2net_map(const std::map<std::string, WLNet*>& name2net_map) { _name2net_map = name2net_map; }

  std::vector<WLNet*>& get_net_list() { return _net_list; }

  void add_net(WLNet* net) { _net_list.push_back(net); }
  void clear_net_list();
  void add_net(const std::string& name, const std::vector<std::pair<int64_t, int64_t>>& pin_list);

  WLNet* find_net(const std::string& net_name) const;

 private:
  std::vector<WLNet*> _net_list;
  std::map<std::string, WLNet*> _name2net_map;

  std::string fixSlash(std::string raw_str);
  WLPin* wrapWLPin(idb::IdbPin* idb_pin);
};
}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_WIRELENGTH_WIRELENGTHEVAL_HPP_
