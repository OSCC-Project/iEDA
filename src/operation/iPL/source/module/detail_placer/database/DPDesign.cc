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
#include "DPDesign.hh"

namespace ipl {

DPDesign::DPDesign() : _dp_instances_range(0), _dp_nets_range(0), _dp_pins_range(0)
{
}

DPDesign::~DPDesign()
{
  // delete instance list
  for (auto* dp_inst : _dpInstance_list) {
    delete dp_inst;
  }
  _dpInstance_list.clear();
  _dpInst_inst_map.clear();

  // delete net list
  for (auto* dp_net : _dpNet_list) {
    delete dp_net;
  }
  _dpNet_list.clear();
  _dpNet_map.clear();

  // delete pin list
  for (auto* dp_pin : _dpPin_list) {
    delete dp_pin;
  }
  _dpPin_list.clear();
  _dpPin_map.clear();

  _dpInst_inst_map.clear();
  _inst_dpInst_map.clear();

  // delete cluster
  for (auto pair : _dpCluster_map) {
    delete pair.second;
  }
  _dpCluster_map.clear();
}

void DPDesign::add_instance(DPInstance* inst)
{
  _dpInstance_list.push_back(inst);
  _dpInstance_map.emplace(inst->get_name(), inst);
  inst->set_inst_id(_dp_instances_range);
  _dp_instances_range += 1;
}

void DPDesign::add_net(DPNet* net)
{
  _dpNet_list.push_back(net);
  _dpNet_map.emplace(net->get_name(), net);
  net->set_net_id(_dp_nets_range);
  _dp_nets_range += 1;
}

void DPDesign::add_pin(DPPin* pin)
{
  _dpPin_list.push_back(pin);
  _dpPin_map.emplace(pin->get_name(), pin);
  pin->set_pin_id(_dp_pins_range);
  _dp_pins_range += 1;
}

void DPDesign::add_cluster(DPCluster* cluster)
{
  _dpCluster_map.emplace(cluster->get_name(), cluster);
}

void DPDesign::connectInst(DPInstance* dp_inst, Instance* pl_inst)
{
  _dpInst_inst_map.emplace(dp_inst, pl_inst);
  _inst_dpInst_map.emplace(pl_inst, dp_inst);
}

DPInstance* DPDesign::find_instance(std::string inst_name)
{
  DPInstance* dp_inst = nullptr;
  auto it = _dpInstance_map.find(inst_name);
  if (it != _dpInstance_map.end()) {
    dp_inst = it->second;
  }
  return dp_inst;
}

DPNet* DPDesign::find_net(std::string net_name)
{
  DPNet* dp_net = nullptr;
  auto it = _dpNet_map.find(net_name);
  if (it != _dpNet_map.end()) {
    dp_net = it->second;
  }
  return dp_net;
}

DPPin* DPDesign::find_pin(std::string pin_name)
{
  DPPin* dp_pin = nullptr;
  auto it = _dpPin_map.find(pin_name);
  if (it != _dpPin_map.end()) {
    dp_pin = it->second;
  }
  return dp_pin;
}

DPCluster* DPDesign::find_cluster(std::string cluster_name)
{
  DPCluster* dp_cluster = nullptr;
  auto it = _dpCluster_map.find(cluster_name);
  if (it != _dpCluster_map.end()) {
    dp_cluster = it->second;
  }
  return dp_cluster;
}

void DPDesign::writeBackToPL(int32_t shift_x, int32_t shift_y)
{
  for (auto pair : _dpInst_inst_map) {
    auto* pl_inst = pair.second;
    if (pl_inst->isFixed() || pl_inst->isFakeInstance() || pl_inst->isOutsideInstance()) {
      continue;
    }

    auto* dp_inst = pair.first;
    int32_t inst_x = dp_inst->get_coordi().get_x() - shift_x;
    int32_t inst_y = dp_inst->get_coordi().get_y() - shift_y;

    pl_inst->set_orient(dp_inst->get_orient());
    pl_inst->update_coordi(inst_x, inst_y);

    pl_inst->set_instance_state(INSTANCE_STATE::kPlaced);
  }
}

int64_t DPDesign::calInstAffectiveHPWL(DPInstance* inst)
{
  int64_t affective_hpwl = 0;
  for (auto* pin : inst->get_pin_list()) {
    auto* pin_net = pin->get_net();
    affective_hpwl += pin_net->calCurrentHPWL();
  }
  return affective_hpwl;
}

void DPDesign::deleteCluster(std::string cluster_name)
{
  auto it = _dpCluster_map.find(cluster_name);
  if (it != _dpCluster_map.end()) {
    delete it->second;
    _dpCluster_map.erase(it);
  } else {
    LOG_WARNING << cluster_name << " has not been add";
  }
}

void DPDesign::clearClusterInfo()
{
  for (auto* inst : _dpInstance_list) {
    inst->set_belong_cluster(nullptr);
  }

  for (auto pair : _dpCluster_map) {
    delete pair.second;
  }
  _dpCluster_map.clear();
}

}  // namespace ipl