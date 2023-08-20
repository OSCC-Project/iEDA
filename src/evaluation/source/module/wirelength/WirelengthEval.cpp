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
#include "WirelengthEval.hpp"

#include <fstream>
#include <regex>
#include <set>

#include "EvalLog.hpp"
#include "EvalUtil.hpp"
#include "flute.h"

namespace eval {

WirelengthEval::WirelengthEval()
{
  Flute::readLUT();
  LOG_INFO << "FLUTE initialized";
}

void WirelengthEval::initWLNetList()
{
  auto* idb_builder = dmInst->get_idb_builder();
  idb::IdbDesign* idb_design = idb_builder->get_def_service()->get_design();

  for (auto* idb_net : idb_design->get_net_list()->get_net_list()) {
    std::string net_name = fixSlash(idb_net->get_net_name());
    WLNet* net_ptr = new WLNet();
    net_ptr->set_name(net_name);

    auto connect_type = idb_net->get_connect_type();
    if (connect_type == IdbConnectType::kSignal) {
      net_ptr->set_type(NET_TYPE::kSignal);
    } else if (connect_type == IdbConnectType::kClock) {
      net_ptr->set_type(NET_TYPE::kClock);
    } else if (connect_type == IdbConnectType::kReset) {
      net_ptr->set_type(NET_TYPE::kReset);
    } else {
      net_ptr->set_type(NET_TYPE::kNone);
    }

    auto* idb_driving_pin = idb_net->get_driving_pin();
    if (idb_driving_pin) {
      WLPin* pin_ptr = wrapWLPin(idb_driving_pin);
      net_ptr->add_pin(pin_ptr);
      net_ptr->set_driver_pin(pin_ptr);
    }
    for (auto* idb_load_pin : idb_net->get_load_pins()) {
      WLPin* pin_ptr = wrapWLPin(idb_load_pin);
      net_ptr->add_pin(pin_ptr);
      net_ptr->add_sink_pin(pin_ptr);
    }
    net_ptr->set_real_wirelength(idb_net->wireLength());

    _net_list.emplace_back(net_ptr);
  }
}

WLPin* WirelengthEval::wrapWLPin(IdbPin* idb_pin)
{
  auto* idb_inst = idb_pin->get_instance();
  WLPin* pin_ptr = nullptr;

  if (!idb_inst) {
    pin_ptr = new WLPin();
    pin_ptr->set_name(idb_pin->get_pin_name());
    pin_ptr->set_type(PIN_TYPE::kIOPort);
  } else {
    std::string pin_name = idb_inst->get_name() + ":" + idb_pin->get_pin_name();
    pin_ptr = new WLPin();
    pin_ptr->set_name(idb_pin->get_pin_name());
    pin_ptr->set_type(PIN_TYPE::kInstancePort);
  }

  LOG_ERROR_IF(!pin_ptr) << "Fail on creating ieval PIN!";

  // set pin io type.
  auto pin_direction = idb_pin->get_term()->get_direction();
  if (pin_direction == IdbConnectDirection::kInput) {
    pin_ptr->set_io_type(PIN_IO_TYPE::kInput);
  } else if (pin_direction == IdbConnectDirection::kOutput) {
    pin_ptr->set_io_type(PIN_IO_TYPE::kOutput);
  } else if (pin_direction == IdbConnectDirection::kInOut) {
    pin_ptr->set_io_type(PIN_IO_TYPE::kInputOutput);
  } else {
    pin_ptr->set_io_type(PIN_IO_TYPE::kNone);
  }

  // set pin center coordinate.
  pin_ptr->set_x(idb_pin->get_average_coordinate()->get_x());
  pin_ptr->set_y(idb_pin->get_average_coordinate()->get_y());

  return pin_ptr;
}

std::string WirelengthEval::fixSlash(std::string raw_str)
{
  std::regex re(R"(\\)");
  return std::regex_replace(raw_str, re, "");
}

int64_t WirelengthEval::evalTotalWL(const std::string& wl_type)
{
  int64_t total_wl = 0;
  WLFactory wirelength_factory;
  WL* p_wirelength = wirelength_factory.createWL(wl_type);
  if (p_wirelength != NULL) {
    total_wl = p_wirelength->getTotalWL(_net_list);
    delete p_wirelength;
    p_wirelength = NULL;
  }
  return total_wl;
}

int64_t WirelengthEval::evalTotalWL(WIRELENGTH_TYPE wl_type)
{
  int64_t total_wl = 0;
  WLFactory wirelength_factory;
  WL* p_wirelength = wirelength_factory.createWL(wl_type);
  if (p_wirelength != NULL) {
    total_wl = p_wirelength->getTotalWL(_net_list);
    delete p_wirelength;
    p_wirelength = NULL;
  }
  return total_wl;
}

void WirelengthEval::reportWirelength(const std::string& plot_path, const std::string& output_file_name)
{
  LOG_INFO << " Wirelength Evaluator is working ... ... ";

  std::ofstream plot(plot_path + output_file_name + ".csv");
  if (!plot.good()) {
    std::cerr << "plot wirelength:: cannot open " << output_file_name << "for writing" << std::endl;
    exit(1);
  }

  std::stringstream feed;
  feed.precision(5);
  feed << "Design"
       << "," << output_file_name + ".def"
       << "," << std::endl;

  int64_t HPWL = 0;
  int64_t HTree = 0;
  int64_t VTree = 0;
  int64_t Star = 0;
  int64_t Clique = 0;
  int64_t B2B = 0;
  int64_t Flute = 0;
  int64_t DetailRouteWL = 0;

  for (WLNet* net : _net_list) {
    HPWL += net->HPWL();
    HTree += net->HTree();
    VTree += net->VTree();
    Star += net->Star();
    Clique += net->Clique();
    B2B += net->B2B();
    Flute += net->FluteWL();
    DetailRouteWL += net->detailRouteWL();
  }

  feed << "Total HPWL"
       << "," << HPWL << "," << std::endl;
  feed << "Total HTree"
       << "," << HTree << "," << std::endl;
  feed << "Total VTree"
       << "," << VTree << "," << std::endl;
  feed << "Total Star"
       << "," << Star << "," << std::endl;
  feed << "Total Clique"
       << "," << Clique << "," << std::endl;
  feed << "Total B2B"
       << "," << B2B << "," << std::endl;
  feed << "Total Flute"
       << "," << Flute << "," << std::endl;
  feed << "Total DetailRouteWL"
       << "," << DetailRouteWL << "," << std::endl;

  feed << std::endl;
  feed << "net_name"
       << ","
       << "HPWL"
       << ","
       << "HTree"
       << ","
       << "VTree"
       << ","
       << "Star"
       << ","
       << "Clique"
       << ","
       << "Bound2Bound"
       << ","
       << "FLUTE"
       << ","
       << "DetailRouteWL"
       << "," << std::endl;

  for (size_t i = 0; i < _net_list.size(); i++) {
    feed << _net_list[i]->get_name() << ",";
    feed << _net_list[i]->HPWL() << ",";
    feed << _net_list[i]->HTree() << ",";
    feed << _net_list[i]->VTree() << ",";
    feed << _net_list[i]->Star() << ",";
    feed << _net_list[i]->Clique() << ",";
    feed << _net_list[i]->B2B() << ",";
    feed << _net_list[i]->FluteWL() << ",";
    feed << _net_list[i]->detailRouteWL() << ",";
    feed << std::endl;
  }

  plot << feed.str();
  feed.clear();
  plot.close();
  LOG_INFO << output_file_name << " has been created in " << plot_path;
}

void WirelengthEval::checkWLType(const std::string& wl_type)
{
  std::set<std::string> wl_type_set
      = {"kWLM", "kHPWL", "kHTree", "kVTree", "kClique", "kStar", "kB2B", "kFlute", "kPlaneRoute", "kSpaceRoute", "kDR"};
  auto it = wl_type_set.find(wl_type);
  if (it == wl_type_set.end()) {
    LOG_ERROR << wl_type << " is not be supported in our evaluator";
    LOG_ERROR << "Only the following types are supported: kWLM, kHPWL, kHTree, kVTree, kClique, kStar, kB2B, kFlute, kPlaneRoute, "
                 "kSpaceRoute, kDR";
    LOG_ERROR << "EXIT";
    exit(1);
  } else {
    LOG_INFO << wl_type << " is selected in Wirelength Evaluator";
  }
}

int64_t WirelengthEval::evalOneNetWL(const std::string& net_name, const std::string& wl_type)
{
  int64_t net_WL = 0;

  auto enum_type = magic_enum::enum_cast<WL_TYPE>(wl_type);
  switch (enum_type.value()) {
    case kWLM:
      net_WL = _name2net_map[net_name]->wireLoadModel();
      break;
    case kHPWL:
      net_WL = _name2net_map[net_name]->HPWL();
      break;
    case kHTree:
      net_WL = _name2net_map[net_name]->HTree();
      break;
    case kVTree:
      net_WL = _name2net_map[net_name]->VTree();
      break;
    case kClique:
      net_WL = _name2net_map[net_name]->Clique();
      break;
    case kStar:
      net_WL = _name2net_map[net_name]->Star();
      break;
    case kB2B:
      net_WL = _name2net_map[net_name]->B2B();
      break;
    case kFlute:
      net_WL = _name2net_map[net_name]->FluteWL();
      break;
    case kDR:
      net_WL = _name2net_map[net_name]->detailRouteWL();
      break;
    default:
      break;
  }
  LOG_INFO << "net_name:" << net_name << "   wirelength_type:" << wl_type << " = " << net_WL;
  return net_WL;
}

int64_t WirelengthEval::evalOneNetWL(const std::string& wl_type, WLNet* wl_net)
{
  int64_t net_WL = 0;

  auto enum_type = magic_enum::enum_cast<WL_TYPE>(wl_type);
  switch (enum_type.value()) {
    case kWLM:
      net_WL = wl_net->wireLoadModel();
      break;
    case kHPWL:
      net_WL = wl_net->HPWL();
      break;
    case kHTree:
      net_WL = wl_net->HTree();
      break;
    case kVTree:
      net_WL = wl_net->VTree();
      break;
    case kClique:
      net_WL = wl_net->Clique();
      break;
    case kStar:
      net_WL = wl_net->Star();
      break;
    case kB2B:
      net_WL = wl_net->B2B();
      break;
    case kFlute:
      net_WL = wl_net->FluteWL();
      break;
    case kDR:
      net_WL = wl_net->detailRouteWL();
      break;
    default:
      LOG_ERROR << wl_type << "is not be supported";
      break;
  }
  return net_WL;
}

int64_t WirelengthEval::evalDriver2LoadWL(WLNet* wl_net, const std::string& sink_pin_name)
{
  return wl_net->LShapedWL(sink_pin_name);
}

int64_t WirelengthEval::evalDriver2LoadWL(const std::string& net_name, const std::string& sink_pin_name)
{
  int64_t net_WL = 0;
  std::string wirelength_type;
  net_WL = _name2net_map[net_name]->LShapedWL(sink_pin_name);
  wirelength_type = "driver2load";
  LOG_INFO << "net_name:" << net_name << "    wirelength_type:" << wirelength_type << " = " << net_WL << std::endl;
  return net_WL;
}

std::map<std::string, int64_t> WirelengthEval::getName2WLmap(const std::string& wl_type)
{
  std::map<std::string, int64_t> name_WL_map;

  if (wl_type == "SteinerRUDY") {
    for (size_t i = 0; i < _net_list.size(); i++) {
      name_WL_map.emplace(_net_list[i]->get_name(), _net_list[i]->FluteWL());
    }
  } else if (wl_type == "TrueRUDY") {
    for (size_t i = 0; i < _net_list.size(); i++) {
      name_WL_map.emplace(_net_list[i]->get_name(), _net_list[i]->detailRouteWL());
    }
  }
  return name_WL_map;
}

void WirelengthEval::add_net(const std::string& name, const std::vector<std::pair<int64_t, int64_t>>& pin_list)
{
  WLNet* length_net = new WLNet();
  length_net->set_name(name);
  for (auto& pin : pin_list) {
    length_net->add_pin(pin.first, pin.second);
  }
  _net_list.push_back(length_net);
  _name2net_map[name] = length_net;
}

WLNet* WirelengthEval::find_net(const std::string& net_name) const
{
  auto net_pair = _name2net_map.find(net_name);
  if (net_pair != _name2net_map.end()) {
    return net_pair->second;
  } else {
    return nullptr;
  }
}

}  // namespace eval
