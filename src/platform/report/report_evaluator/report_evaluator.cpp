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
#include "report_evaluator.h"

#include <algorithm>
#include <fstream>
#include <future>
#include <memory>
#include <tuple>

#include "CongBin.hpp"
#include "IdbPins.h"
#include "ReportTable.hh"
#include "WLNet.hpp"
#include "WLPin.hpp"
#include "flute.h"
#include "fort.hpp"
#include "idm.h"
namespace iplf {

template <typename T>
static void freeWrapped(std::vector<T*>& obj_vec)
{
  for (auto*& obj : obj_vec) {
    for (auto*& pin : obj->get_pin_list()) {
      if (pin) {
        delete pin;
        pin = nullptr;
      }
    }
    delete obj;
    obj = nullptr;
  }
}

/**
 * @brief given a vector<float> of data, threshold, ceiling and step,
 * return a list of range and value counts.
 */
auto ReportEvaluator::CongStats(float threshold, float step, vector<float>& data)
{
  float ceiling = *std::max_element(data.begin(), data.end());
  vector<float> range;
  for (auto r = threshold; r < ceiling; r += step) {
    range.push_back(r);
  }
  vector<int> count(range.size(), 0);
  for (auto value : data) {
    if (value < threshold || value > ceiling) {
      continue;
    }
    size_t index = 0;
    for (; index < range.size() - 1 && range[index] < value; ++index)
      ;
    count[index]++;
  }
  return std::tuple(range, count);
}

std::shared_ptr<ieda::ReportTable> ReportEvaluator::createWireLengthReport()
{
  // auto start = chrono::steady_clock::now();
  // auto end = chrono::steady_clock::now();
  // auto ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();

  // prepare data & initialization work
  auto& nets = dmInst->get_idb_design()->get_net_list()->get_net_list();
  auto wl_nets = EvalWrapper::parallelWrap<eval::WLNet>(nets, EvalWrapper::wrapWLNet);
  eval::EvalAPI::initInst();
  Flute::readLUT();

  // calculate wire length asynchronously
  auto real = std::async(std::launch::async, [&nets]() { return computeWireLength(nets, &idb::IdbNet::wireLength); });
  auto hpwl = std::async(std::launch::async, [&wl_nets]() { return computeWireLength(wl_nets, &eval::WLNet::HPWL); });
  auto b2b = std::async(std::launch::async, [&wl_nets]() { return computeWireLength(wl_nets, &eval::WLNet::B2B); });
  auto flute = std::async(std::launch::async, [&wl_nets]() { return computeWireLength(wl_nets, &eval::WLNet::FluteWL); });

  auto [real_total, real_max, real_max_net] = real.get();
  auto [hpwl_total, hpwl_max, hpwl_max_net] = hpwl.get();
  auto [b2b_total, b2b_max, b2b_max_net] = b2b.get();
  auto [flute_total, flute_max, flute_max_net] = flute.get();
  auto net_num = nets.size();

  // output results to report table
  std::vector<std::string> header = {"Wire-length Model", "Total Length", "Average Length", "Longest Net Name", "Longest Length"};
  auto tbl = std::make_shared<ieda::ReportTable>("Wire Length Report", header, static_cast<int>(ReportEvaluatorType::kWireLength));
  if (real_total > 0) {
    *tbl << "Real Length" << real_total << real_total / net_num << real_max_net->get_net_name() << real_max << TABLE_ENDLINE;
  }
  *tbl << "HPWL" << hpwl_total << hpwl_total / net_num << hpwl_max_net->get_name() << hpwl_max << TABLE_ENDLINE;
  *tbl << "Bound2Bound" << b2b_total << b2b_total / net_num << b2b_max_net->get_name() << b2b_max << TABLE_ENDLINE;
  *tbl << "Flute" << flute_total << flute_total / net_num << flute_max_net->get_name() << flute_max << TABLE_ENDLINE;

  // free allocated data asynchoronously
  std::thread([](std::vector<eval::WLNet*>&& nets) { freeWrapped(nets); }, std::move(wl_nets)).detach();

  return tbl;
}

std::shared_ptr<ieda::ReportTable> ReportEvaluator::createCongestionReport()
{
  // prepare data & initialize EvalAPI
  eval::EvalAPI::initInst();
  // auto& nets = dmInst->get_idb_design()->get_net_list()->get_net_list();
  auto& inst_list = dmInst->get_idb_design()->get_instance_list()->get_instance_list();

  // RUDY congestion model apply only to place evaluation.
  // TODO( Need real net congestion evaluation, i.e. net congestion after route)
  // auto cog_nets_future
  //     = std::async(std::launch::async, [&nets]() { return EvalWrapper::parallelWrap<eval::CongNet>(nets, EvalWrapper::wrapCongNet); });
  // auto cong_nets = cog_nets_future.get();
  // auto net_congestion = EvalInst.evalNetCong(cong_grid.get(), cong_nets, "RUDY");

  // wrap  CongInst

  auto cong_inst = EvalWrapper::parallelWrap<eval::CongInst>(inst_list, EvalWrapper::wrapCongInst);
  auto cong_grid = initCongGrid();

  // evaluate Instance Density & Pin Density
  auto inst_density = EvalInst.evalInstDens(cong_grid.get(), cong_inst);
  auto pin_density = EvalInst.evalPinDens(cong_grid.get(), cong_inst);

  float inst_den_max = *std::max_element(inst_density.begin(), inst_density.end());
  float pin_den_max = *std::max_element(pin_density.begin(), pin_density.end());
  // prepare report table

  auto [inst_den_range, inst_den_cnt] = CongStats(inst_den_max * 0.75, 0.05 * inst_den_max, inst_density);
  auto [pin_den_range, pin_den_cnt] = CongStats(pin_den_max * 0.5, pin_den_max * 0.1, pin_density);
  inst_den_range.push_back(inst_den_max);
  pin_den_range.push_back(pin_den_max);

  std::vector<std::string> header = {"Grid Bin Size", "Bin Partition", "Total Count"};
  auto tbl = std::make_shared<ieda::ReportTable>("Congestion Report", header, static_cast<int>(ReportEvaluatorType::kCongestion));
  // Bin information
  *tbl << Str::printf("%d * %d", cong_grid->get_bin_size_x(), cong_grid->get_bin_size_y())
       << Str::printf("%d by %d", cong_grid->get_bin_cnt_x(), cong_grid->get_bin_cnt_y())
       << cong_grid->get_bin_cnt_x() * cong_grid->get_bin_cnt_y() << TABLE_ENDLINE;

  // Instance Density Information
  *tbl << TABLE_HEAD << "Instance Density Range"
       << "Bins Count"
       << "Percentage " << TABLE_ENDLINE;
  for (int i = inst_den_cnt.size() - 1; i >= 0; --i) {
    *tbl << Str::printf("%.2f ~ %.2f", inst_den_range[i], inst_den_range[i + 1]) << inst_den_cnt[i]
         << Str::printf("%.2f", 100 * inst_den_cnt[i] / static_cast<double>(inst_density.size())) << TABLE_ENDLINE;
  }

  // Pin Density Information
  *tbl << TABLE_HEAD << "Pin Count Range"
       << "Bins Count"
       << "Percentage" << TABLE_ENDLINE;
  for (int i = pin_den_cnt.size() - 1; i >= 0; --i) {
    *tbl << Str::printf("%.0f ~ %.0f", pin_den_range[i], pin_den_range[i + 1]) << pin_den_cnt[i]
         << Str::printf("%.2f", 100 * pin_den_cnt[i] / static_cast<double>(pin_density.size())) << TABLE_ENDLINE;
  }

  // Release wrapped congestion instance objects.
  std::thread([](std::vector<eval::CongInst*>&& insts) { freeWrapped(insts); }, std::move(cong_inst)).detach();

  return tbl;
}

std::unique_ptr<eval::CongGrid> ReportEvaluator::initCongGrid(int32_t bin_cnt_x, int32_t bin_cnt_y)
{
  auto cong_grid = std::make_unique<eval::CongGrid>();
  int32_t grid_lx = dmInst->get_idb_layout()->get_core()->get_bounding_box()->get_low_x();
  int32_t grid_ly = dmInst->get_idb_layout()->get_core()->get_bounding_box()->get_low_y();
  cong_grid->set_lx(grid_lx);
  cong_grid->set_ly(grid_ly);
  cong_grid->set_bin_cnt_x(bin_cnt_x);
  cong_grid->set_bin_cnt_y(bin_cnt_y);
  int32_t grid_width = dmInst->get_idb_layout()->get_core()->get_bounding_box()->get_width();
  int32_t grid_height = dmInst->get_idb_layout()->get_core()->get_bounding_box()->get_height();
  cong_grid->set_bin_size_x(ceil(grid_width / (float) bin_cnt_x));
  cong_grid->set_bin_size_y(ceil(grid_height / (float) bin_cnt_y));
  cong_grid->set_routing_layers_number(dmInst->get_idb_layout()->get_layers()->get_routing_layers_number());
  cong_grid->initBins(dmInst->get_idb_layout()->get_layers());
  return cong_grid;
}

/////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// W R A P P E R //////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

eval::WLNet* EvalWrapper::wrapWLNet(idb::IdbNet* idb_net)
{
  eval::WLNet* net_ptr = new eval::WLNet();
  net_ptr->set_name(idb_net->get_net_name());
  auto connect_type = idb_net->get_connect_type();
  if (connect_type == idb::IdbConnectType::kSignal) {
    net_ptr->set_type(eval::NET_TYPE::kSignal);
  } else if (connect_type == idb::IdbConnectType::kClock) {
    net_ptr->set_type(eval::NET_TYPE::kClock);
  } else if (connect_type == idb::IdbConnectType::kReset) {
    net_ptr->set_type(eval::NET_TYPE::kReset);
  } else {
    net_ptr->set_type(eval::NET_TYPE::kNone);
  }
  // set pins.
  auto* idb_driving_pin = idb_net->get_driving_pin();
  if (idb_driving_pin) {
    eval::WLPin* pin_ptr = wrapWLPin(idb_driving_pin);
    net_ptr->add_pin(pin_ptr);
    net_ptr->set_driver_pin(pin_ptr);
  }
  for (auto* idb_load_pin : idb_net->get_load_pins()) {
    eval::WLPin* pin_ptr = wrapWLPin(idb_load_pin);
    net_ptr->add_pin(pin_ptr);
    net_ptr->add_sink_pin(pin_ptr);
  }
  net_ptr->set_real_wirelength(idb_net->wireLength());

  return net_ptr;
}

eval::WLPin* EvalWrapper::wrapWLPin(idb::IdbPin* idb_pin)
{
  auto* idb_inst = idb_pin->get_instance();
  eval::WLPin* pin_ptr;
  if (!idb_inst) {
    pin_ptr = new eval::WLPin();
    pin_ptr->set_name(idb_pin->get_pin_name());
    pin_ptr->set_type(eval::PIN_TYPE::kIOPort);
  } else {
    // default separator
    std::string pin_name = idb_inst->get_name() + "\\" + idb_pin->get_pin_name();
    pin_ptr = new eval::WLPin();
    pin_ptr->set_name(idb_pin->get_pin_name());
    pin_ptr->set_type(eval::PIN_TYPE::kInstancePort);
  }

  // set pin io type.
  auto pin_direction = idb_pin->get_term()->get_direction();
  if (pin_direction == idb::IdbConnectDirection::kInput) {
    pin_ptr->set_io_type(eval::PIN_IO_TYPE::kInput);
  } else if (pin_direction == idb::IdbConnectDirection::kOutput) {
    pin_ptr->set_io_type(eval::PIN_IO_TYPE::kOutput);
  } else if (pin_direction == idb::IdbConnectDirection::kInOut) {
    pin_ptr->set_io_type(eval::PIN_IO_TYPE::kInputOutput);
  } else {
    pin_ptr->set_io_type(eval::PIN_IO_TYPE::kNone);
  }
  // set pin center coordinate.
  pin_ptr->set_x(idb_pin->get_average_coordinate()->get_x());
  pin_ptr->set_y(idb_pin->get_average_coordinate()->get_y());
  return pin_ptr;
}

eval::CongInst* EvalWrapper::wrapCongInst(idb::IdbInstance* idb_inst)
{
  eval::CongInst* cong_inst = new eval::CongInst;
  cong_inst->set_name(idb_inst->get_name());
  auto* box = idb_inst->get_bounding_box();
  cong_inst->set_loc_type(computeInstType(idb_inst));
  cong_inst->set_shape(box->get_low_x(), box->get_low_y(), box->get_high_x(), box->get_high_y());
  for (auto* pin : idb_inst->get_pin_list()->get_pin_list()) {
    cong_inst->add_pin(wrapCongPin(pin));
  }

  return cong_inst;
}

eval::CongNet* EvalWrapper::wrapCongNet(idb::IdbNet* idb_net)
{
  eval::CongNet* cong_net = new eval::CongNet();
  cong_net->set_name(idb_net->get_net_name());

  auto* ipl_driver_pin = idb_net->get_driving_pin();
  if (ipl_driver_pin) {
    eval::CongPin* cong_pin = wrapCongPin(ipl_driver_pin);
    cong_net->add_pin(cong_pin);
  }
  for (auto& ipl_load_pin : idb_net->get_load_pins()) {
    eval::CongPin* cong_pin = wrapCongPin(ipl_load_pin);
    cong_net->add_pin(cong_pin);
  }
  return cong_net;
}
eval::CongPin* EvalWrapper::wrapCongPin(idb::IdbPin* pin)
{
  eval::CongPin* cong_pin = new eval::CongPin();
  cong_pin->set_name(pin->get_pin_name());
  int64_t x = pin->get_average_coordinate()->get_x();
  int64_t y = pin->get_average_coordinate()->get_y();

  cong_pin->set_x(x);
  cong_pin->set_y(y);
  cong_pin->set_coord(eval::Point<int64_t>(x, y));
  if (pin->get_instance()) {
    cong_pin->set_type(eval::PIN_TYPE::kInstancePort);
  } else {
    cong_pin->set_type(eval::PIN_TYPE::kIOPort);
  }
  return cong_pin;
}

eval::INSTANCE_LOC_TYPE EvalWrapper::computeInstType(idb::IdbInstance* idb_inst)
{
  auto* core = dmInst->get_idb_layout()->get_core()->get_bounding_box();
  auto* die = dmInst->get_idb_layout()->get_die()->get_bounding_box();
  auto* inst = idb_inst->get_bounding_box();
  auto check = [](int32_t b1, int32_t a1, int32_t a2, int32_t b2) { return b1 <= a1 && a2 <= b2; };
  if (check(die->get_low_x(), inst->get_low_x(), inst->get_high_x(), core->get_low_x())
      || check(die->get_low_y(), inst->get_low_y(), inst->get_high_y(), core->get_low_y())
      || check(core->get_high_x(), inst->get_low_x(), inst->get_high_x(), die->get_high_x())
      || check(core->get_high_y(), inst->get_low_y(), inst->get_high_y(), die->get_high_y())) {
    return eval::INSTANCE_LOC_TYPE::kOutside;
  }
  return eval::INSTANCE_LOC_TYPE::kNormal;
}

}  // namespace iplf