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
#include "CongestionEval.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <regex>

#include "../manager.hpp"
#include "EvalLog.hpp"

namespace eval {

void CongestionEval::initCongGrid(const int bin_cnt_x, const int bin_cnt_y)
{
  auto* idb_builder = dmInst->get_idb_builder();
  idb::IdbLayout* idb_layout = idb_builder->get_def_service()->get_layout();
  idb::IdbLayers* idb_layers = idb_layout->get_layers();
  idb::IdbRect* core_bbox = idb_layout->get_core()->get_bounding_box();
  int32_t lx = core_bbox->get_low_x();
  int32_t ly = core_bbox->get_low_y();
  int32_t width = core_bbox->get_width();
  int32_t height = core_bbox->get_height();

  _cong_grid->set_lx(lx);
  _cong_grid->set_ly(ly);
  _cong_grid->set_bin_cnt_x(bin_cnt_x);
  _cong_grid->set_bin_cnt_y(bin_cnt_y);
  _cong_grid->set_bin_size_x(ceil(width / (float) bin_cnt_x));
  _cong_grid->set_bin_size_y(ceil(height / (float) bin_cnt_y));
  _cong_grid->set_routing_layers_number(idb_layers->get_routing_layers_number());
  _cong_grid->initBins(idb_layers);
  _cong_grid->initTracksNum(idb_layers);
}

void CongestionEval::initCongInst()
{
  auto* idb_builder = dmInst->get_idb_builder();
  idb::IdbDesign* idb_design = idb_builder->get_def_service()->get_design();
  idb::IdbLayout* idb_layout = idb_builder->get_def_service()->get_layout();
  idb::IdbDie* idb_die = idb_layout->get_die();
  idb::IdbRect* idb_core = idb_layout->get_core()->get_bounding_box();
  int32_t die_lx = idb_die->get_llx();
  int32_t die_ly = idb_die->get_lly();
  int32_t die_ux = idb_die->get_urx();
  int32_t die_uy = idb_die->get_ury();
  int32_t core_lx = idb_core->get_low_x();
  int32_t core_ly = idb_core->get_low_y();
  int32_t core_ux = idb_core->get_high_x();
  int32_t core_uy = idb_core->get_high_y();

  for (auto* idb_inst : idb_design->get_instance_list()->get_instance_list()) {
    CongInst* inst_ptr = new CongInst();
    inst_ptr->set_name(idb_inst->get_name());
    auto bbox = idb_inst->get_bounding_box();
    inst_ptr->set_shape(bbox->get_low_x(), bbox->get_low_y(), bbox->get_high_x(), bbox->get_high_y());

    auto inst_status = idb_inst->get_status();
    if (inst_status == IdbPlacementStatus::kNone) {
      inst_ptr->set_status(INSTANCE_STATUS::kNone);
    } else if (inst_status == IdbPlacementStatus::kFixed) {
      inst_ptr->set_status(INSTANCE_STATUS::kFixed);
    } else if (inst_status == IdbPlacementStatus::kCover) {
      inst_ptr->set_status(INSTANCE_STATUS::kCover);
    } else if (inst_status == IdbPlacementStatus::kPlaced) {
      inst_ptr->set_status(INSTANCE_STATUS::kPlaced);
    } else if (inst_status == IdbPlacementStatus::kUnplaced) {
      inst_ptr->set_status(INSTANCE_STATUS::kUnplaced);
    } else {
      inst_ptr->set_status(INSTANCE_STATUS::kMax);
    }

    if (idb_inst->is_flip_flop()) {
      inst_ptr->set_flip_flop(true);
      std::cout << "flip flop " << std::endl;
    }

    if ((bbox->get_low_x() >= die_lx && bbox->get_high_x() <= core_lx) || (bbox->get_low_x() >= core_ux && bbox->get_high_x() <= die_ux)
        || (bbox->get_low_y() >= die_ly && bbox->get_high_y() <= core_ly)
        || (bbox->get_low_y() >= core_uy && bbox->get_high_y() <= die_uy)) {
      inst_ptr->set_loc_type(INSTANCE_LOC_TYPE::kOutside);
    } else {
      inst_ptr->set_loc_type(INSTANCE_LOC_TYPE::kNormal);
    }

    _cong_inst_list.emplace_back(inst_ptr);
    _name_to_inst_map.emplace(inst_ptr->get_name(), inst_ptr);
  }
}

void CongestionEval::initCongNetList()
{
  auto* idb_builder = dmInst->get_idb_builder();
  idb::IdbDesign* idb_design = idb_builder->get_def_service()->get_design();

  for (auto* idb_net : idb_design->get_net_list()->get_net_list()) {
    std::string net_name = fixSlash(idb_net->get_net_name());
    CongNet* net_ptr = new CongNet();
    net_ptr->set_name(net_name);

    auto connect_type = idb_net->get_connect_type();
    if (connect_type == IdbConnectType::kSignal) {
      net_ptr->set_connect_type(NET_CONNECT_TYPE::kSignal);
    } else if (connect_type == IdbConnectType::kClock) {
      net_ptr->set_connect_type(NET_CONNECT_TYPE::kClock);
    } else if (connect_type == IdbConnectType::kPower) {
      net_ptr->set_connect_type(NET_CONNECT_TYPE::kPower);
    } else if (connect_type == IdbConnectType::kGround) {
      net_ptr->set_connect_type(NET_CONNECT_TYPE::kGround);
    } else {
      net_ptr->set_connect_type(NET_CONNECT_TYPE::kNone);
    }

    auto* idb_driving_pin = idb_net->get_driving_pin();
    if (idb_driving_pin) {
      CongPin* pin_ptr = wrapCongPin(idb_driving_pin);
      pin_ptr->set_two_pin_net_num(std::min(idb_net->get_pin_number() - 1, 50));
      net_ptr->add_pin(pin_ptr);
    }
    for (auto* idb_load_pin : idb_net->get_load_pins()) {
      CongPin* pin_ptr = wrapCongPin(idb_load_pin);
      pin_ptr->set_two_pin_net_num(std::min(idb_net->get_pin_number() - 1, 50));
      net_ptr->add_pin(pin_ptr);
    }

    _cong_net_list.emplace_back(net_ptr);
  }
}

void CongestionEval::mapInst2Bin()
{
  for (auto& inst : _cong_inst_list) {
    if (inst->isNormalInst()) {
      std::pair<int, int> pair_x = _cong_grid->getMinMaxX(inst);
      std::pair<int, int> pair_y = _cong_grid->getMinMaxY(inst);
      // fix the out of core bug
      if (pair_x.second >= _cong_grid->get_bin_cnt_x()) {
        pair_x.second = _cong_grid->get_bin_cnt_x() - 1;
      }
      if (pair_y.second >= _cong_grid->get_bin_cnt_y()) {
        pair_y.second = _cong_grid->get_bin_cnt_y() - 1;
      }
      if (pair_x.first < 0) {
        pair_x.first = 0;
      }
      if (pair_y.first < 0) {
        pair_y.first = 0;
      }
      for (int i = pair_x.first; i <= pair_x.second; ++i) {
        for (int j = pair_y.first; j <= pair_y.second; ++j) {
          CongBin* bin = _cong_grid->get_bin_list()[j * _cong_grid->get_bin_cnt_x() + i];
          bin->add_inst(inst);
        }
      }
    }
  }
}

void CongestionEval::mapNetCoord2Grid()
{
  for (auto& net : _cong_net_list) {
    if (net->get_pin_list().size() == 1) {
      continue;
    }
    std::pair<int, int> pair_x = _cong_grid->getMinMaxX(net);
    std::pair<int, int> pair_y = _cong_grid->getMinMaxY(net);
    // fix the out of core bug
    if (pair_x.first < 0) {
      pair_x.first = 0;
    }
    if (pair_y.first < 0) {
      pair_y.first = 0;
    }
    if (pair_x.second >= _cong_grid->get_bin_cnt_x()) {
      pair_x.second = _cong_grid->get_bin_cnt_x() - 1;
    }
    if (pair_y.second >= _cong_grid->get_bin_cnt_y()) {
      pair_y.second = _cong_grid->get_bin_cnt_y() - 1;
    }
    for (int i = pair_x.first; i <= pair_x.second; i++) {
      for (int j = pair_y.first; j <= pair_y.second; j++) {
        CongBin* bin = _cong_grid->get_bin_list()[j * _cong_grid->get_bin_cnt_x() + i];
        bin->add_net(net);
      }
    }
  }
}

void CongestionEval::evalInstDens(INSTANCE_STATUS inst_status, bool eval_flip_flop)
{
  for (auto& bin : _cong_grid->get_bin_list()) {
    double overlap_area = 0.0;
    double density = 0.0;
    for (auto& inst : bin->get_inst_list()) {
      auto status = inst->get_status();
      if (inst_status == status) {
        if (eval_flip_flop == false) {
          overlap_area += getOverlapArea(bin, inst);
        } else if (inst->isFlipFlop()) {
          overlap_area += getOverlapArea(bin, inst);
        }
      }
    }
    density = overlap_area / bin->get_area();
    bin->set_inst_density(density);
  }
}

void CongestionEval::evalPinDens(INSTANCE_STATUS inst_status, int level)
{
  // Reset pin numbers for all bins
  for (auto& bin : _cong_grid->get_bin_list()) {
    bin->set_pin_num(0);
  }

  // Calculate pin numbers for each bin
  for (auto& bin : _cong_grid->get_bin_list()) {
    for (auto& inst : bin->get_inst_list()) {
      auto status = inst->get_status();
      if (inst_status == status) {
        for (auto& pin : inst->get_pin_list()) {
          auto pin_x = pin->get_x();
          auto pin_y = pin->get_y();
          if (pin_x > bin->get_lx() && pin_x < bin->get_ux() && pin_y > bin->get_ly() && pin_y < bin->get_uy()) {
            bin->increPinNum();
          }
        }
      }
    }
  }
  // Update pin numbers for each bin using the pin sum in the square region
  if (level != 0) {
    const std::vector<std::vector<int>> kernel = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    const int height = _cong_grid->get_bin_cnt_y();
    const int width = _cong_grid->get_bin_cnt_x();
    std::vector<std::vector<int>> padded_matrix(height + 2, std::vector<int>(width + 2, 0));
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        padded_matrix[i + 1][j + 1] = _cong_grid->get_bin_list()[i * width + j]->get_pin_num();
      }
    }
    std::vector<std::vector<int>> output_matrix(height, std::vector<int>(width, 0));
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        int sum = 0;
        for (int k = 0; k < 3; k++) {
          for (int l = 0; l < 3; l++) {
            sum += padded_matrix[i + k][j + l] * kernel[k][l];
          }
        }
        output_matrix[i][j] = sum;
      }
    }
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        _cong_grid->get_bin_list()[i * width + j]->set_pin_num(output_matrix[i][j]);
      }
    }
  }
}

void CongestionEval::evalNetDens(INSTANCE_STATUS inst_status)
{
  for (auto& bin : _cong_grid->get_bin_list()) {
    bin->set_net_cong(0.0);
  }
  for (auto& bin : _cong_grid->get_bin_list()) {
    for (auto& inst : bin->get_inst_list()) {
      auto status = inst->get_status();
      if (inst_status == status) {
        for (auto& pin : inst->get_pin_list()) {
          auto pin_x = pin->get_x();
          auto pin_y = pin->get_y();
          if (pin_x > bin->get_lx() && pin_x < bin->get_ux() && pin_y > bin->get_ly() && pin_y < bin->get_uy()) {
            bin->increNetCong(pin->get_two_pin_net_num());
          }
        }
      }
    }
  }
}

void CongestionEval::evalLocalNetDens()
{
  for (auto& bin : _cong_grid->get_bin_list()) {
    bin->set_net_cong(0.0);
  }
  for (auto& bin : _cong_grid->get_bin_list()) {
    for (auto& net : bin->get_net_list()) {
      auto net_bbox = net->get_height() * net->get_width();
      if (getOverlapArea(bin, net) == net_bbox) {
        bin->increNetCong(1);
      }
    }
  }
}

void CongestionEval::evalGlobalNetDens()
{
  for (auto& bin : _cong_grid->get_bin_list()) {
    bin->set_net_cong(0.0);
  }
  for (auto& bin : _cong_grid->get_bin_list()) {
    for (auto& net : bin->get_net_list()) {
      auto net_bbox = net->get_height() * net->get_width();
      if (getOverlapArea(bin, net) != net_bbox) {
        bin->increNetCong(1);
      }
    }
  }
}

void CongestionEval::plotBinValue(const string& plot_path, const string& output_file_name, CONGESTION_TYPE cong_type)
{
  std::ofstream plot(plot_path + output_file_name + ".csv");
  if (!plot.good()) {
    std::cerr << "plot bin value:: cannot open " << output_file_name << "for writing" << std::endl;
    exit(1);
  }
  std::stringstream feed;
  feed.precision(5);
  int32_t x_cnt = _cong_grid->get_bin_cnt_x();
  int32_t y_cnt = _cong_grid->get_bin_cnt_y();

  for (int i = 0; i < x_cnt; i++) {
    if (i == x_cnt - 1) {
      feed << "col_" << i;
    } else {
      feed << "col_" << i << ",";
    }
  }
  feed << std::endl;

  if (cong_type == CONGESTION_TYPE::kInstDens) {
    for (int i = y_cnt - 1; i >= 0; i--) {
      for (int j = 0; j < x_cnt; j++) {
        double inst_density = _cong_grid->get_bin_list()[i * x_cnt + j]->get_inst_density();
        if (j == x_cnt - 1) {
          feed << inst_density;
        } else {
          feed << inst_density << ",";
        }
      }
      feed << std::endl;
    }
  } else if (cong_type == CONGESTION_TYPE::kPinDens) {
    for (int i = y_cnt - 1; i >= 0; i--) {
      for (int j = 0; j < x_cnt; j++) {
        int pin_density = _cong_grid->get_bin_list()[i * x_cnt + j]->get_pin_num();
        if (j == x_cnt - 1) {
          feed << pin_density;
        } else {
          feed << pin_density << ",";
        }
      }
      feed << std::endl;
    }
  } else if (cong_type == CONGESTION_TYPE::kNetCong) {
    for (int i = y_cnt - 1; i >= 0; i--) {
      for (int j = 0; j < x_cnt; j++) {
        double net_cong = _cong_grid->get_bin_list()[i * x_cnt + j]->get_net_cong();
        if (j == x_cnt - 1) {
          feed << net_cong;
        } else {
          feed << net_cong << ",";
        }
      }
      feed << std::endl;
    }
  }

  plot << feed.str();
  feed.clear();
  plot.close();
  LOG_INFO << output_file_name + ".csv"
           << " has been created in " << plot_path;
}

int32_t CongestionEval::evalInstNum(INSTANCE_STATUS inst_status)
{
  int32_t inst_num = 0;
  for (auto& inst : _cong_inst_list) {
    if (inst->get_status() == inst_status) {
      ++inst_num;
    }
  }
  return inst_num;
}

int32_t CongestionEval::evalNetNum(NET_CONNECT_TYPE net_type)
{
  int32_t net_num = 0;
  for (auto& net : _cong_net_list) {
    if (net_type == net->get_connect_type()) {
      net_num++;
    }
  }
  return net_num;
}

int32_t CongestionEval::evalPinTotalNum(INSTANCE_STATUS inst_status)
{
  int32_t pin_num = 0;
  if (inst_status == INSTANCE_STATUS::kNone) {
    for (auto& inst : _cong_inst_list) {
      pin_num += inst->get_pin_list().size();
    }
  } else {
    for (auto& inst : _cong_inst_list) {
      if (inst->get_status() == inst_status) {
        pin_num += inst->get_pin_list().size();
      }
    }
  }
  return pin_num;
}

int32_t CongestionEval::evalRoutingLayerNum()
{
  return _cong_grid->get_routing_layers_number();
}

int32_t CongestionEval::evalTrackNum(DIRECTION direction)
{
  if (direction == DIRECTION::kH) {
    return _cong_grid->get_track_num_h();
  } else if (direction == DIRECTION::kV) {
    return _cong_grid->get_track_num_v();
  } else {
    return _cong_grid->get_track_num_h() + _cong_grid->get_track_num_v();
  }
}

std::vector<int64_t> CongestionEval::evalChipWidthHeightArea(CHIP_REGION_TYPE chip_region_type)
{
  auto* idb_builder = dmInst->get_idb_builder();
  idb::IdbLayout* idb_layout = idb_builder->get_def_service()->get_layout();
  idb::IdbDie* idb_die = idb_layout->get_die();
  idb::IdbRect* idb_core = idb_layout->get_core()->get_bounding_box();
  int32_t die_width = idb_die->get_width();
  int32_t die_height = idb_die->get_height();
  int64_t die_area = idb_die->get_area();
  int32_t core_width = idb_core->get_width();
  int32_t core_height = idb_core->get_height();
  int64_t core_area = idb_core->get_area();

  std::vector<int64_t> chip_info_list;
  chip_info_list.reserve(3);
  if (chip_region_type == CHIP_REGION_TYPE::kDie) {
    chip_info_list.emplace_back(die_width);
    chip_info_list.emplace_back(die_height);
    chip_info_list.emplace_back(die_area);
  } else if (chip_region_type == CHIP_REGION_TYPE::kCore) {
    chip_info_list.emplace_back(core_width);
    chip_info_list.emplace_back(core_height);
    chip_info_list.emplace_back(core_area);
  }
  return chip_info_list;
}

vector<pair<string, pair<int32_t, int32_t>>> CongestionEval::evalInstSize(INSTANCE_STATUS inst_status)
{
  vector<pair<string, pair<int32_t, int32_t>>> inst_name_to_size_list;
  for (auto& inst : _cong_inst_list) {
    if (inst->get_status() == inst_status) {
      pair<string, pair<int32_t, int32_t>> name_to_size
          = std::make_pair(inst->get_name(), std::make_pair(inst->get_width(), inst->get_height()));
      inst_name_to_size_list.emplace_back(name_to_size);
    }
  }
  return inst_name_to_size_list;
}

vector<pair<string, pair<int32_t, int32_t>>> CongestionEval::evalNetSize()
{
  vector<pair<string, pair<int32_t, int32_t>>> net_name_to_size_list;
  for (auto& net : _cong_net_list) {
    pair<string, pair<int32_t, int32_t>> name_to_size
        = std::make_pair(net->get_name(), std::make_pair(net->get_width(), net->get_height()));
    net_name_to_size_list.emplace_back(name_to_size);
  }
  return net_name_to_size_list;
}

void CongestionEval::evalNetCong(RUDY_TYPE rudy_type, DIRECTION direction)
{
  for (auto& bin : _cong_grid->get_bin_list()) {
    bin->set_net_cong(0.0);
  }
  for (auto& bin : _cong_grid->get_bin_list()) {
    int32_t overlap_area = 0;
    double congestion = 0.0;
    for (auto& net : bin->get_net_list()) {
      overlap_area = getOverlapArea(bin, net);
      if (rudy_type == RUDY_TYPE::kRUDY) {
        congestion += overlap_area * getRudy(bin, net, direction);
      } else if (rudy_type == RUDY_TYPE::kPinRUDY) {
        congestion += overlap_area * getPinRudy(bin, net, direction);
      } else if (rudy_type == RUDY_TYPE::kLUTRUDY) {
        int32_t pin_num = net->get_pin_list().size();
        int64_t net_width = net->get_width();
        int64_t net_height = net->get_height();
        int32_t aspect_ratio = 0;
        if (net_width >= net_height && net_height != 0) {
          aspect_ratio = std::round(net_width / net_height);
        } else if (net_width < net_height && net_width != 0) {
          aspect_ratio = std::round(net_height / net_width);
        } else {
          aspect_ratio = 1;
        }
        float l_ness = 0.0;
        if (pin_num <= 3) {
          l_ness = 1.0;
        } else if (pin_num <= 15) {
          std::vector<std::pair<int32_t, int32_t>> point_set;
          for (int i = 0; i < pin_num; ++i) {
            const int32_t pin_x = net->get_pin_list()[i]->get_x();
            const int32_t pin_y = net->get_pin_list()[i]->get_y();
            point_set.emplace_back(std::make_pair(pin_x, pin_y));
          }
          l_ness = calcLness(point_set, net->get_lx(), net->get_ux(), net->get_ly(), net->get_uy());
        } else {
          l_ness = 0.5;
        }
        congestion += overlap_area * getLUT(pin_num, aspect_ratio, l_ness) * getRudy(bin, net, direction);
      }
    }
    bin->set_net_cong(congestion);
  }
}

void CongestionEval::plotTileValue(const string& plot_path, const string& output_file_name)
{
  // plot each layer congestion map
  plotGRCong(plot_path, output_file_name);
  // plot two congestion map:  TOF/MOF
  plotOverflow(plot_path, output_file_name);
}

float CongestionEval::evalAreaUtils(INSTANCE_STATUS status)
{
  float utilization = 0.f;
  int64_t macro_area = evalArea(INSTANCE_STATUS::kFixed);
  int64_t core_area = evalChipWidthHeightArea(CHIP_REGION_TYPE::kCore)[2];
  int64_t stdcell_area = evalArea(INSTANCE_STATUS::kPlaced);
  if (status == INSTANCE_STATUS::kFixed) {
    utilization = macro_area / (float) core_area;
  } else if (status == INSTANCE_STATUS::kPlaced) {
    utilization = stdcell_area / (float) (core_area - macro_area);
  }

  return utilization;
}

int64_t CongestionEval::evalArea(INSTANCE_STATUS status)
{
  int64_t area = 0;
  for (auto& inst : _cong_inst_list) {
    if (inst->get_status() == status) {
      area += inst->get_shape().get_area();
    }
  }
  return area;
}

std::vector<int64_t> CongestionEval::evalMacroPeriBias()
{
  std::vector<int64_t> bias_list;
  auto core_info = evalChipWidthHeightArea(CHIP_REGION_TYPE::kCore);
  auto core_width = core_info[0];
  auto core_height = core_info[1];
  for (auto& inst : _cong_inst_list) {
    if (inst->get_status() == INSTANCE_STATUS::kFixed) {
      auto left = inst->get_lx();
      auto bottom = inst->get_ly();
      auto right = core_width - left - inst->get_width();
      auto top = core_height - bottom - inst->get_height();
      auto min_bias = std::min(std::min(left, bottom), std::min(right, top));
      min_bias = min_bias * min_bias;
      bias_list.emplace_back(min_bias);
    }
  }
  return bias_list;
}

int32_t CongestionEval::evalRmTrackNum()
{
  int32_t remain_track_num = 0;
  for (int i = 0; i < _tile_grid->get_num_routing_layers(); ++i) {
    remain_track_num += evalRemain(i);
  }
  return remain_track_num;
}

int32_t CongestionEval::evalOfTrackNum()
{
  int32_t overflow_track_num = 0;
  for (int i = 0; i < _tile_grid->get_num_routing_layers(); ++i) {
    overflow_track_num += evalOverflow(i)[0];
  }
  return overflow_track_num;
}

int32_t CongestionEval::evalRemain(int layer_index)
{
  int32_t x_cnt = _tile_grid->get_tile_cnt_x();
  int32_t y_cnt = _tile_grid->get_tile_cnt_y();
  int32_t start_index = layer_index * x_cnt * y_cnt;
  int32_t one_layer_remain = 0;

  if (_tile_grid->get_tiles()[start_index]->is_horizontal()) {
    for (int i = y_cnt - 1; i >= 0; i--) {
      for (int j = 0; j < x_cnt; j++) {
        auto tile = _tile_grid->get_tiles()[i * x_cnt + j + start_index];
        one_layer_remain = std::min((tile->get_east_cap() - tile->get_east_use()), (tile->get_west_cap() - tile->get_west_use()));
        one_layer_remain = std::max(one_layer_remain, 0);
      }
    }

  } else {
    for (int i = y_cnt - 1; i >= 0; i--) {
      for (int j = 0; j < x_cnt; j++) {
        auto tile = _tile_grid->get_tiles()[i * x_cnt + j + start_index];
        one_layer_remain = std::min((tile->get_north_cap() - tile->get_north_use()), (tile->get_south_cap() - tile->get_south_use()));
        one_layer_remain = std::max(one_layer_remain, 0);
      }
    }
  }

  return one_layer_remain;
}

int32_t CongestionEval::evalMacroGuidance(int32_t cx, int32_t cy, int32_t width, int32_t height, const string& name)
{
  int32_t guidance = 0;
  for (auto& inst : _cong_inst_list) {
    if (inst->get_name() == name) {
      int32_t g_width = width + inst->get_width();
      int32_t g_height = height + inst->get_height();
      int32_t x_dist = std::abs(cx - (inst->get_lx() + inst->get_width() * 0.5)) - g_width;
      int32_t y_dist = std::abs(cy - (inst->get_ly() + inst->get_height() * 0.5)) - g_height;
      guidance = std::max(0, x_dist) + std::max(0, y_dist);
    }
  }
  return guidance;
}

double CongestionEval::evalMacroChannelUtil(float dist_ratio)
{
  int64_t channel_area = 0;
  auto core_info = evalChipWidthHeightArea(CHIP_REGION_TYPE::kCore);
  int32_t check_width = core_info[0] * dist_ratio;
  int32_t check_height = core_info[1] * dist_ratio;
  int64_t core_area = core_info[2];

  std::vector<CongInst*> macro_list;
  for (auto& inst : _cong_inst_list) {
    if (inst->get_status() == INSTANCE_STATUS::kFixed) {
      macro_list.emplace_back(inst);
    }
  }

  if (macro_list.size() == 0) {
    return 0.0;
  } else {
    for (size_t i = 0; i < macro_list.size(); i++) {
      for (size_t j = i + 1; j < macro_list.size(); j++) {
        auto x_dist = std::abs(macro_list[i]->get_lx() - macro_list[j]->get_lx());
        auto y_dist = std::abs(macro_list[i]->get_ly() - macro_list[j]->get_ly());
        // 确保在沟道空间内
        if ((x_dist < check_width) && (y_dist < check_height)) {
          // 确定模块的位置相对关系
          if (x_dist < y_dist) {
            // 模块是上下沟道
            auto lx = std::max(macro_list[i]->get_lx(), macro_list[j]->get_lx());
            auto ux = std::min(macro_list[i]->get_ux(), macro_list[j]->get_ux());
            int64_t height = 0;
            if (macro_list[i]->get_ly() < macro_list[j]->get_ly()) {
              // i在下，j在上
              height = macro_list[j]->get_ly() - macro_list[i]->get_ly() - macro_list[i]->get_height();
            } else {
              // i在上，j在下
              height = macro_list[i]->get_ly() - macro_list[j]->get_ly() - macro_list[j]->get_height();
            }
            channel_area += (ux - lx) * height;
          } else {
            // 模块是左右沟道
            auto ly = std::max(macro_list[i]->get_ly(), macro_list[j]->get_ly());
            auto uy = std::min(macro_list[i]->get_uy(), macro_list[j]->get_uy());
            int64_t width = 0;
            if (macro_list[i]->get_lx() < macro_list[j]->get_lx()) {
              // i在左，j在右
              width = macro_list[j]->get_lx() - macro_list[i]->get_lx() - macro_list[i]->get_width();
            } else {
              // i在右，j在左
              width = macro_list[i]->get_lx() - macro_list[j]->get_lx() - macro_list[j]->get_width();
            }
            channel_area += (uy - ly) * width;
          }
        }
      }
    }
  }

  return channel_area / (double) core_area;
}

double CongestionEval::evalMacroChannelPinRatio(float dist_ratio)
{
  double pin_ratio = 0.0;
  int pin_num = 0;

  auto core_info = evalChipWidthHeightArea(CHIP_REGION_TYPE::kCore);
  int32_t check_width = core_info[0] * dist_ratio;
  int32_t check_height = core_info[1] * dist_ratio;

  std::vector<CongInst*> macro_list;
  for (auto& inst : _cong_inst_list) {
    if (inst->get_status() == INSTANCE_STATUS::kFixed) {
      macro_list.emplace_back(inst);
    }
  }

  if (macro_list.size() == 0) {
    return 0;
  } else {
    for (size_t i = 0; i < macro_list.size(); i++) {
      for (size_t j = i + 1; j < macro_list.size(); j++) {
        auto x_dist = std::abs(macro_list[i]->get_lx() - macro_list[j]->get_lx());
        auto y_dist = std::abs(macro_list[i]->get_ly() - macro_list[j]->get_ly());
        // 确保在沟道空间内
        if ((x_dist < check_width) && (y_dist < check_height)) {
          // 确定模块的位置相对关系
          if (x_dist < y_dist) {
            // 模块是上下沟道
            auto lx = std::max(macro_list[i]->get_lx(), macro_list[j]->get_lx());
            auto ux = std::min(macro_list[i]->get_ux(), macro_list[j]->get_ux());
            if (macro_list[i]->get_ly() < macro_list[j]->get_ly()) {
              // i在下，j在上
              for (auto& pin : macro_list[i]->get_pin_list()) {
                auto pin_x = pin->get_x();
                auto pin_y = pin->get_y();
                if (pin_x >= lx && pin_x <= ux && pin_y >= macro_list[i]->get_uy() * 0.9) {
                  pin_num++;
                }
              }
              for (auto& pin : macro_list[j]->get_pin_list()) {
                auto pin_x = pin->get_x();
                auto pin_y = pin->get_y();
                if (pin_x >= lx && pin_x <= ux && pin_y <= macro_list[j]->get_ly() * 1.1) {
                  pin_num++;
                }
              }
            } else {
              // i在上，j在下
              for (auto& pin : macro_list[j]->get_pin_list()) {
                auto pin_x = pin->get_x();
                auto pin_y = pin->get_y();
                if (pin_x >= lx && pin_x <= ux && pin_y >= macro_list[j]->get_uy() * 0.9) {
                  pin_num++;
                }
              }
              for (auto& pin : macro_list[i]->get_pin_list()) {
                auto pin_x = pin->get_x();
                auto pin_y = pin->get_y();
                if (pin_x >= lx && pin_x <= ux && pin_y <= macro_list[i]->get_ly() * 1.1) {
                  pin_num++;
                }
              }
            }

          } else {
            // 模块是左右沟道
            auto ly = std::max(macro_list[i]->get_ly(), macro_list[j]->get_ly());
            auto uy = std::min(macro_list[i]->get_uy(), macro_list[j]->get_uy());
            if (macro_list[i]->get_lx() < macro_list[j]->get_lx()) {
              // i在左，j在右
              for (auto& pin : macro_list[j]->get_pin_list()) {
                auto pin_x = pin->get_x();
                auto pin_y = pin->get_y();
                if (pin_y >= ly && pin_y <= uy && pin_x <= macro_list[j]->get_lx() * 1.1) {
                  pin_num++;
                }
              }
              for (auto& pin : macro_list[i]->get_pin_list()) {
                auto pin_x = pin->get_x();
                auto pin_y = pin->get_y();
                if (pin_y >= ly && pin_y <= uy && pin_x >= macro_list[i]->get_ux() * 0.9) {
                  pin_num++;
                }
              }
            } else {
              // i在右，j在左
              for (auto& pin : macro_list[i]->get_pin_list()) {
                auto pin_x = pin->get_x();
                auto pin_y = pin->get_y();
                if (pin_y >= ly && pin_y <= uy && pin_x <= macro_list[i]->get_lx() * 1.1) {
                  pin_num++;
                }
              }
              for (auto& pin : macro_list[j]->get_pin_list()) {
                auto pin_x = pin->get_x();
                auto pin_y = pin->get_y();
                if (pin_y >= ly && pin_y <= uy && pin_x >= macro_list[j]->get_ux() * 0.9) {
                  pin_num++;
                }
              }
            }
          }
        }
      }
    }
  }

  pin_ratio = pin_num / (double) evalPinTotalNum();

  return pin_ratio;
}

float CongestionEval::calcLness(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmin, int32_t xmax, int32_t ymin, int32_t ymax)
{
  int64_t bbox = (xmax - xmin) * (ymax - ymin);
  int64_t R1 = calcLowerLeftRP(point_set, xmin, ymin);
  int64_t R2 = calcLowerRightRP(point_set, xmax, ymin);
  int64_t R3 = calcUpperLeftRP(point_set, xmin, ymax);
  int64_t R4 = calcUpperRightRP(point_set, xmax, ymax);
  int64_t R = std::max({R1, R2, R3, R4});
  float l_ness;
  if (bbox != 0) {
    l_ness = R / bbox;
  } else {
    l_ness = 1.0;
  }
  return l_ness;
}

int64_t CongestionEval::calcLowerLeftRP(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmin, int32_t ymin)
{
  std::sort(point_set.begin(), point_set.end());  // Sort point_set with x-coordinates in ascending order
  int64_t R = 0, y0 = point_set[0].second;
  for (size_t i = 1; i < point_set.size(); i++) {
    int32_t xi = point_set[i].first;
    if (point_set[i].second <= y0) {
      R = std::max(R, (xi - xmin) * (y0 - ymin));
      y0 = point_set[i].second;
    }
  }
  return R;
}

int64_t CongestionEval::calcLowerRightRP(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmax, int32_t ymin)
{
  std::sort(point_set.begin(), point_set.end(), std::greater<std::pair<int32_t, int32_t>>());  // Sort point_set with x-coordinates in
                                                                                               // descending order
  int64_t R = 0, y0 = point_set[0].second, xi;
  for (size_t i = 1; i < point_set.size(); i++) {
    xi = point_set[i].first;
    if (point_set[i].second <= y0) {
      R = std::max(R, (xmax - xi) * (y0 - ymin));
      y0 = point_set[i].second;
    }
  }
  return R;
}

int64_t CongestionEval::calcUpperLeftRP(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmin, int32_t ymax)
{
  std::sort(point_set.begin(), point_set.end(), [](const std::pair<int32_t, int32_t>& a, const std::pair<int32_t, int32_t>& b) {
    return a.second > b.second;
  });  // Sort point_set with y-coordinates in descending order
  int64_t R = 0, x0 = point_set[0].first, yi;
  for (size_t i = 1; i < point_set.size(); i++) {
    yi = point_set[i].second;
    if (point_set[i].first <= x0) {
      R = std::max(R, (ymax - yi) * (x0 - xmin));
      x0 = point_set[i].first;
    }
  }
  return R;
}

int64_t CongestionEval::calcUpperRightRP(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmax, int32_t ymax)
{
  std::sort(point_set.begin(), point_set.end(), std::greater<std::pair<int32_t, int32_t>>());  // Sort point_set with x-coordinates in
                                                                                               // descending order
  int64_t R = 0, y0 = point_set[0].second, xi;
  for (size_t i = 1; i < point_set.size(); i++) {
    xi = point_set[i].first;
    if (point_set[i].second >= y0) {
      R = std::max(R, (ymax - y0) * (xmax - xi));
      y0 = point_set[i].second;
    }
  }
  return R;
}

double CongestionEval::getLUT(const int32_t& pin_num, const int32_t& aspect_ratio, const float& l_ness)
{
  switch (aspect_ratio) {
    case 1:
      switch (pin_num) {
        case 1:
        case 2:
        case 3:
          return 1.00;
          break;
        case 4:
          if (l_ness <= 0.25)
            return 1.330;
          if (l_ness <= 0.50)
            return 1.115;
          if (l_ness <= 0.75)
            return 1.050;
          if (l_ness <= 1.00)
            return 1.020;
          break;
        case 5:
          if (l_ness <= 0.25)
            return 1.330;
          if (l_ness <= 0.50)
            return 1.150;
          if (l_ness <= 0.75)
            return 1.080;
          if (l_ness <= 1.00)
            return 1.040;
          break;
        case 6:
          if (l_ness <= 0.25)
            return 1.355;
          if (l_ness <= 0.50)
            return 1.185;
          if (l_ness <= 0.75)
            return 1.110;
          if (l_ness <= 1.00)
            return 1.050;
          break;
        case 7:
          if (l_ness <= 0.25)
            return 1.390;
          if (l_ness <= 0.50)
            return 1.220;
          if (l_ness <= 0.75)
            return 1.135;
          if (l_ness <= 1.00)
            return 1.065;
          break;
        case 8:
          if (l_ness <= 0.25)
            return 1.415;
          if (l_ness <= 0.50)
            return 1.250;
          if (l_ness <= 0.75)
            return 1.160;
          if (l_ness <= 1.00)
            return 1.080;
          break;
        case 9:
          if (l_ness <= 0.25)
            return 1.450;
          if (l_ness <= 0.50)
            return 1.285;
          if (l_ness <= 0.75)
            return 1.185;
          if (l_ness <= 1.00)
            return 1.090;
          break;
        case 10:
          if (l_ness <= 0.25)
            return 1.490;
          if (l_ness <= 0.50)
            return 1.325;
          if (l_ness <= 0.75)
            return 1.210;
          if (l_ness <= 1.00)
            return 1.105;
          break;
        case 11:
          if (l_ness <= 0.25)
            return 1.515;
          if (l_ness <= 0.50)
            return 1.355;
          if (l_ness <= 0.75)
            return 1.235;
          if (l_ness <= 1.00)
            return 1.120;
          break;
        case 12:
          if (l_ness <= 0.25)
            return 1.555;
          if (l_ness <= 0.50)
            return 1.385;
          if (l_ness <= 0.75)
            return 1.260;
          if (l_ness <= 1.00)
            return 1.135;
          break;
        case 13:
          if (l_ness <= 0.25)
            return 1.590;
          if (l_ness <= 0.50)
            return 1.420;
          if (l_ness <= 0.75)
            return 1.280;
          if (l_ness <= 1.00)
            return 1.145;
          break;
        case 14:
          if (l_ness <= 0.25)
            return 1.620;
          if (l_ness <= 0.50)
            return 1.450;
          if (l_ness <= 0.75)
            return 1.310;
          if (l_ness <= 1.00)
            return 1.160;
          break;
        case 15:
          if (l_ness <= 0.25)
            return 1.660;
          if (l_ness <= 0.50)
            return 1.485;
          if (l_ness <= 0.75)
            return 1.330;
          if (l_ness <= 1.00)
            return 1.175;
          break;
        default:
          if (l_ness <= 0.25)
            return 1.660;
          if (l_ness <= 0.50)
            return 1.485;
          if (l_ness <= 0.75)
            return 1.330;
          if (l_ness <= 1.00)
            return 1.175;
          break;
      }
      break;
    case 2:
    case 3:
      switch (pin_num) {
        case 1:
        case 2:
        case 3:
          return 1.00;
          break;
        case 4:
          if (l_ness <= 0.25)
            return 1.240;
          if (l_ness <= 0.50)
            return 1.094;
          if (l_ness <= 0.75)
            return 1.047;
          if (l_ness <= 1.00)
            return 1.023;
          break;
        case 5:
          if (l_ness <= 0.25)
            return 1.240;
          if (l_ness <= 0.50)
            return 1.127;
          if (l_ness <= 0.75)
            return 1.070;
          if (l_ness <= 1.00)
            return 1.037;
          break;
        case 6:
          if (l_ness <= 0.25)
            return 1.273;
          if (l_ness <= 0.50)
            return 1.155;
          if (l_ness <= 0.75)
            return 1.103;
          if (l_ness <= 1.00)
            return 1.051;
          break;
        case 7:
          if (l_ness <= 0.25)
            return 1.306;
          if (l_ness <= 0.50)
            return 1.193;
          if (l_ness <= 0.75)
            return 1.127;
          if (l_ness <= 1.00)
            return 1.065;
          break;
        case 8:
          if (l_ness <= 0.25)
            return 1.348;
          if (l_ness <= 0.50)
            return 1.221;
          if (l_ness <= 0.75)
            return 1.150;
          if (l_ness <= 1.00)
            return 1.080;
          break;
        case 9:
          if (l_ness <= 0.25)
            return 1.377;
          if (l_ness <= 0.50)
            return 1.259;
          if (l_ness <= 0.75)
            return 1.174;
          if (l_ness <= 1.00)
            return 1.094;
          break;
        case 10:
          if (l_ness <= 0.25)
            return 1.409;
          if (l_ness <= 0.50)
            return 1.287;
          if (l_ness <= 0.75)
            return 1.197;
          if (l_ness <= 1.00)
            return 1.103;
          break;
        case 11:
          if (l_ness <= 0.25)
            return 1.447;
          if (l_ness <= 0.50)
            return 1.315;
          if (l_ness <= 0.75)
            return 1.221;
          if (l_ness <= 1.00)
            return 1.117;
          break;
        case 12:
          if (l_ness <= 0.25)
            return 1.485;
          if (l_ness <= 0.50)
            return 1.344;
          if (l_ness <= 0.75)
            return 1.240;
          if (l_ness <= 1.00)
            return 1.131;
          break;
        case 13:
          if (l_ness <= 0.25)
            return 1.513;
          if (l_ness <= 0.50)
            return 1.377;
          if (l_ness <= 0.75)
            return 1.263;
          if (l_ness <= 1.00)
            return 1.141;
          break;
        case 14:
          if (l_ness <= 0.25)
            return 1.546;
          if (l_ness <= 0.50)
            return 1.405;
          if (l_ness <= 0.75)
            return 1.287;
          if (l_ness <= 1.00)
            return 1.150;
          break;
        case 15:
          if (l_ness <= 0.25)
            return 1.579;
          if (l_ness <= 0.50)
            return 1.433;
          if (l_ness <= 0.75)
            return 1.306;
          if (l_ness <= 1.00)
            return 1.169;
          break;
        default:
          if (l_ness <= 0.25)
            return 1.579;
          if (l_ness <= 0.50)
            return 1.433;
          if (l_ness <= 0.75)
            return 1.306;
          if (l_ness <= 1.00)
            return 1.169;
          break;
      }
      break;
    case 4:
      switch (pin_num) {
        case 1:
        case 2:
        case 3:
          return 1.00;
          break;
        case 4:
          if (l_ness <= 0.25)
            return 1.144;
          if (l_ness <= 0.50)
            return 1.064;
          if (l_ness <= 0.75)
            return 1.032;
          if (l_ness <= 1.00)
            return 1.016;
          break;
        case 5:
          if (l_ness <= 0.25)
            return 1.144;
          if (l_ness <= 0.50)
            return 1.084;
          if (l_ness <= 0.75)
            return 1.052;
          if (l_ness <= 1.00)
            return 1.032;
          break;
        case 6:
          if (l_ness <= 0.25)
            return 1.172;
          if (l_ness <= 0.50)
            return 1.108;
          if (l_ness <= 0.75)
            return 1.076;
          if (l_ness <= 1.00)
            return 1.044;
          break;
        case 7:
          if (l_ness <= 0.25)
            return 1.200;
          if (l_ness <= 0.50)
            return 1.128;
          if (l_ness <= 0.75)
            return 1.092;
          if (l_ness <= 1.00)
            return 1.056;
          break;
        case 8:
          if (l_ness <= 0.25)
            return 1.224;
          if (l_ness <= 0.50)
            return 1.156;
          if (l_ness <= 0.75)
            return 1.116;
          if (l_ness <= 1.00)
            return 1.068;
          break;
        case 9:
          if (l_ness <= 0.25)
            return 1.252;
          if (l_ness <= 0.50)
            return 1.180;
          if (l_ness <= 0.75)
            return 1.132;
          if (l_ness <= 1.00)
            return 1.076;
          break;
        case 10:
          if (l_ness <= 0.25)
            return 1.276;
          if (l_ness <= 0.50)
            return 1.204;
          if (l_ness <= 0.75)
            return 1.152;
          if (l_ness <= 1.00)
            return 1.088;
          break;
        case 11:
          if (l_ness <= 0.25)
            return 1.308;
          if (l_ness <= 0.50)
            return 1.228;
          if (l_ness <= 0.75)
            return 1.164;
          if (l_ness <= 1.00)
            return 1.100;
          break;
        case 12:
          if (l_ness <= 0.25)
            return 1.332;
          if (l_ness <= 0.50)
            return 1.252;
          if (l_ness <= 0.75)
            return 1.188;
          if (l_ness <= 1.00)
            return 1.108;
          break;
        case 13:
          if (l_ness <= 0.25)
            return 1.360;
          if (l_ness <= 0.50)
            return 1.276;
          if (l_ness <= 0.75)
            return 1.208;
          if (l_ness <= 1.00)
            return 1.120;
          break;
        case 14:
          if (l_ness <= 0.25)
            return 1.388;
          if (l_ness <= 0.50)
            return 1.300;
          if (l_ness <= 0.75)
            return 1.220;
          if (l_ness <= 1.00)
            return 1.132;
          break;
        case 15:
          if (l_ness <= 0.25)
            return 1.416;
          if (l_ness <= 0.50)
            return 1.316;
          if (l_ness <= 0.75)
            return 1.240;
          if (l_ness <= 1.00)
            return 1.144;
          break;
        default:
          if (l_ness <= 0.25)
            return 1.416;
          if (l_ness <= 0.50)
            return 1.316;
          if (l_ness <= 0.75)
            return 1.240;
          if (l_ness <= 1.00)
            return 1.144;
          break;
      }
      break;
    default:
      switch (pin_num) {
        case 1:
        case 2:
        case 3:
          return 1.00;
          break;
        case 4:
          if (l_ness <= 0.25)
            return 1.144;
          if (l_ness <= 0.50)
            return 1.064;
          if (l_ness <= 0.75)
            return 1.032;
          if (l_ness <= 1.00)
            return 1.016;
          break;
        case 5:
          if (l_ness <= 0.25)
            return 1.144;
          if (l_ness <= 0.50)
            return 1.084;
          if (l_ness <= 0.75)
            return 1.052;
          if (l_ness <= 1.00)
            return 1.032;
          break;
        case 6:
          if (l_ness <= 0.25)
            return 1.172;
          if (l_ness <= 0.50)
            return 1.108;
          if (l_ness <= 0.75)
            return 1.076;
          if (l_ness <= 1.00)
            return 1.044;
          break;
        case 7:
          if (l_ness <= 0.25)
            return 1.200;
          if (l_ness <= 0.50)
            return 1.128;
          if (l_ness <= 0.75)
            return 1.092;
          if (l_ness <= 1.00)
            return 1.056;
          break;
        case 8:
          if (l_ness <= 0.25)
            return 1.224;
          if (l_ness <= 0.50)
            return 1.156;
          if (l_ness <= 0.75)
            return 1.116;
          if (l_ness <= 1.00)
            return 1.068;
          break;
        case 9:
          if (l_ness <= 0.25)
            return 1.252;
          if (l_ness <= 0.50)
            return 1.180;
          if (l_ness <= 0.75)
            return 1.132;
          if (l_ness <= 1.00)
            return 1.076;
          break;
        case 10:
          if (l_ness <= 0.25)
            return 1.276;
          if (l_ness <= 0.50)
            return 1.204;
          if (l_ness <= 0.75)
            return 1.152;
          if (l_ness <= 1.00)
            return 1.088;
          break;
        case 11:
          if (l_ness <= 0.25)
            return 1.308;
          if (l_ness <= 0.50)
            return 1.228;
          if (l_ness <= 0.75)
            return 1.164;
          if (l_ness <= 1.00)
            return 1.100;
          break;
        case 12:
          if (l_ness <= 0.25)
            return 1.332;
          if (l_ness <= 0.50)
            return 1.252;
          if (l_ness <= 0.75)
            return 1.188;
          if (l_ness <= 1.00)
            return 1.108;
          break;
        case 13:
          if (l_ness <= 0.25)
            return 1.360;
          if (l_ness <= 0.50)
            return 1.276;
          if (l_ness <= 0.75)
            return 1.208;
          if (l_ness <= 1.00)
            return 1.120;
          break;
        case 14:
          if (l_ness <= 0.25)
            return 1.388;
          if (l_ness <= 0.50)
            return 1.300;
          if (l_ness <= 0.75)
            return 1.220;
          if (l_ness <= 1.00)
            return 1.132;
          break;
        case 15:
          if (l_ness <= 0.25)
            return 1.416;
          if (l_ness <= 0.50)
            return 1.316;
          if (l_ness <= 0.75)
            return 1.240;
          if (l_ness <= 1.00)
            return 1.144;
          break;
        default:
          if (l_ness <= 0.25)
            return 1.416;
          if (l_ness <= 0.50)
            return 1.316;
          if (l_ness <= 0.75)
            return 1.240;
          if (l_ness <= 1.00)
            return 1.144;
          break;
      }
      break;
  }
  return 0;
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
void CongestionEval::reportCongestion(const std::string& plot_path, const std::string& output_file_name)
{
  LOG_INFO << " ==========================================";
  LOG_INFO << " Congestion Evaluator is working ... ... ";
  LOG_INFO << " ==========================================";
  LOG_INFO << " Evaluating Pin Num for Each Bin ... ... ";
  mapInst2Bin();
  evalPinNum();
  reportPinNum();
  plotPinNum(plot_path, output_file_name);
  LOG_INFO << " Evaluating Cell Density for Each Bin ... ... ";
  evalInstDens();
  reportInstDens();
  plotInstDens(plot_path, output_file_name);
  LOG_INFO << " Evaluating Net Congestion for Each Bin ... ... ";
  mapNetCoord2Grid();
  evalNetCong("RUDY");
  reportNetCong();
  plotNetCong(plot_path, output_file_name, "RUDY");
  evalNetCong("RUDYDev");
  reportNetCong();
  plotNetCong(plot_path, output_file_name, "RUDYDev");
  evalNetCong("SteinerRUDY");
  reportNetCong();
  plotNetCong(plot_path, output_file_name, "StenierRUDY");
  evalNetCong("PinRUDY");
  reportNetCong();
  plotNetCong(plot_path, output_file_name, "PinRUDY");
  // evalNetCong("PinSteinerRUDY");
  // reportNetCong();
  // plotNetCong(plot_path, output_file_name, "PinSteinerRUDY");
  // evalNetCong("TrueRUDY");
  // reportNetCong();
  // plotNetCong(plot_path, output_file_name, "TrueRUDY");
  LOG_INFO << " Evaluating Routing Congestion for Each Tile ... ... ";
}

void CongestionEval::evalPinNum()
{
  for (auto& bin : _cong_grid->get_bin_list()) {
    bin->reset();
  }
  for (auto& bin : _cong_grid->get_bin_list()) {
    for (auto& inst : bin->get_inst_list()) {
      for (auto& pin : inst->get_pin_list()) {
        auto pin_x = pin->get_x();
        auto pin_y = pin->get_y();
        if (pin_x > bin->get_lx() && pin_x < bin->get_ux() && pin_y > bin->get_ly() && pin_y < bin->get_uy()) {
          bin->increPinNum();
        }
      }
    }
  }
}

std::vector<float> CongestionEval::evalPinDens()
{
  evalPinNum();
  auto bin_list = _cong_grid->get_bin_list();
  std::vector<float> pin_density;
  pin_density.reserve(bin_list.size());
  for (size_t i = 0; i < bin_list.size(); ++i) {
    float density = bin_list[i]->get_pin_num();
    pin_density.emplace_back(density);
  }
  return pin_density;
}

void CongestionEval::reportPinNum()
{
  for (auto& bin : _cong_grid->get_bin_list()) {
    LOG_INFO << "Bin: (" << bin->get_lx() << "," << bin->get_ly() << "),(" << bin->get_ux() << "," << bin->get_uy()
             << "), pin num: " << bin->get_pin_num();
  }
}

void CongestionEval::plotPinNum(const std::string& plot_path, const std::string& output_file_name)
{
  std::ofstream plot(plot_path + output_file_name + "_" + "PinNum" + ".csv");
  if (!plot.good()) {
    std::cerr << "plot PinNum: cannot open " << output_file_name << "for writing" << std::endl;
    exit(1);
  }
  std::stringstream feed;
  feed.precision(5);
  int y_cnt = _cong_grid->get_bin_cnt_y();
  int x_cnt = _cong_grid->get_bin_cnt_x();
  for (int i = 0; i < x_cnt; i++) {
    if (i == x_cnt - 1) {
      feed << "clo_" << i;
    } else {
      feed << "col_" << i << ",";
    }
  }
  feed << std::endl;
  for (int i = y_cnt - 1; i >= 0; i--) {
    for (int j = 0; j < x_cnt; j++) {
      int pin_cnt = _cong_grid->get_bin_list()[i * x_cnt + j]->get_pin_num();
      if (j == x_cnt - 1) {
        feed << pin_cnt;
      } else {
        feed << pin_cnt << ",";
      }
    }
    feed << std::endl;
  }
  plot << feed.str();
  feed.clear();
  plot.close();
  LOG_INFO << output_file_name + "_" + "PinNum" + ".csv"
           << " has been created in " << plot_path;
}

int CongestionEval::getBinPinNum(const int& index_x, const int& index_y)
{
  int index = index_x + index_y * _cong_grid->get_bin_cnt_x();
  auto bin = _cong_grid->get_bin_list()[index];
  return bin->get_pin_num();
}

double CongestionEval::getBinPinDens(const int& index_x, const int& index_y)
{
  int index = index_x + index_y * _cong_grid->get_bin_cnt_x();
  auto bin = _cong_grid->get_bin_list()[index];
  int pin_num = bin->get_pin_num();
  return (double) pin_num / bin->get_area();
}

void CongestionEval::evalInstDens()
{
  for (auto& bin : _cong_grid->get_bin_list()) {
    bin->reset();
  }
  for (auto& bin : _cong_grid->get_bin_list()) {
    double overlap_area = 0.0;
    double density = 0.0;
    for (auto& inst : bin->get_inst_list()) {
      overlap_area += getOverlapArea(bin, inst);
    }
    density = overlap_area / bin->get_area();
    bin->set_inst_density(density);
  }
}

std::vector<float> CongestionEval::getInstDens()
{
  evalInstDens();
  auto bin_list = _cong_grid->get_bin_list();
  std::vector<float> inst_density;
  inst_density.reserve(bin_list.size());
  for (size_t i = 0; i < bin_list.size(); ++i) {
    float density = bin_list[i]->get_inst_density();
    inst_density.emplace_back(density);
  }
  return inst_density;
}

void CongestionEval::reportInstDens()
{
  for (auto& bin : _cong_grid->get_bin_list()) {
    LOG_INFO << "Bin: (" << bin->get_lx() << "," << bin->get_ly() << "),(" << bin->get_ux() << "," << bin->get_uy()
             << "), density: " << bin->get_inst_density();
  }
}

void CongestionEval::plotInstDens(const std::string& plot_path, const std::string& output_file_name)
{
  std::ofstream plot(plot_path + output_file_name + "_" + "InstDens" + ".csv");
  if (!plot.good()) {
    std::cerr << "plot inst density:: cannot open " << output_file_name << "for writing" << std::endl;
    exit(1);
  }
  std::stringstream feed;
  feed.precision(5);
  int y_cnt = _cong_grid->get_bin_cnt_y();
  int x_cnt = _cong_grid->get_bin_cnt_x();
  for (int i = 0; i < x_cnt; i++) {
    if (i == x_cnt - 1) {
      feed << "clo_" << i;
    } else {
      feed << "col_" << i << ",";
    }
  }
  feed << std::endl;
  for (int i = y_cnt - 1; i >= 0; i--) {
    for (int j = 0; j < x_cnt; j++) {
      float density = _cong_grid->get_bin_list()[i * x_cnt + j]->get_inst_density();
      if (j == x_cnt - 1) {
        feed << density;
      } else {
        feed << density << ",";
      }
    }
    feed << std::endl;
  }
  plot << feed.str();
  feed.clear();
  plot.close();
  LOG_INFO << output_file_name + "_" + "InstDens" + ".csv"
           << " has been created in " << plot_path;
}

double CongestionEval::getBinInstDens(const int& index_x, const int& index_y)
{
  int index = index_x + index_y * _cong_grid->get_bin_cnt_x();
  auto bin = _cong_grid->get_bin_list()[index];
  return bin->get_inst_density();
}

/////////////////////////////////
/////////////////////////////////
/*----evaluate net congestion----*/
/////////////////////////////////
/////////////////////////////////

std::string CongestionEval::fixSlash(std::string raw_str)
{
  std::regex re(R"(\\)");
  return std::regex_replace(raw_str, re, "");
}

CongPin* CongestionEval::wrapCongPin(idb::IdbPin* idb_pin)
{
  CongPin* pin_ptr = nullptr;

  auto* idb_inst = idb_pin->get_instance();
  if (!idb_inst) {
    pin_ptr = new CongPin();
    pin_ptr->set_name(idb_pin->get_pin_name());
    pin_ptr->set_type(PIN_TYPE::kIOPort);
  } else {
    std::string pin_name = idb_inst->get_name() + ":" + idb_pin->get_pin_name();
    pin_ptr = new CongPin();
    pin_ptr->set_name(idb_pin->get_pin_name());
    pin_ptr->set_type(PIN_TYPE::kInstancePort);
    // set instance
    auto inst_iter = _name_to_inst_map.find(idb_inst->get_name());
    if (inst_iter != _name_to_inst_map.end()) {
      CongInst* inst = (*inst_iter).second;
      inst->add_pin(pin_ptr);
    } else {
      LOG_ERROR << idb_inst->get_name() << "is not found in cong_inst_map";
    }
  }

  pin_ptr->set_x(idb_pin->get_average_coordinate()->get_x());
  pin_ptr->set_y(idb_pin->get_average_coordinate()->get_y());

  return pin_ptr;
}

void CongestionEval::evalNetCong(const std::string& rudy_type)
{
}

std::vector<float> CongestionEval::getNetCong(const std::string& rudy_type)
{
  evalNetCong(rudy_type);
  auto bin_list = _cong_grid->get_bin_list();
  std::vector<float> net_cong;
  net_cong.reserve(bin_list.size());
  for (size_t i = 0; i < bin_list.size(); ++i) {
    float density = bin_list[i]->get_net_cong();
    net_cong.emplace_back(density);
  }
  return net_cong;
}

void CongestionEval::reportNetCong()
{
  for (auto& bin : _cong_grid->get_bin_list()) {
    LOG_INFO << "  Bin: (" << bin->get_lx() << "," << bin->get_ly() << "),(" << bin->get_ux() << "," << bin->get_uy()
             << "), net_cong: " << bin->get_net_cong();
  }
}

void CongestionEval::plotNetCong(const std::string& plot_path, const std::string& output_file_name, const std::string& type)
{
  std::ofstream plot(plot_path + output_file_name + "_" + type + ".csv");
  if (!plot.good()) {
    std::cerr << "plot NetCong:: cannot open " << output_file_name << "for writing" << std::endl;
    exit(1);
  }
  std::stringstream feed;
  feed.precision(5);
  int y_cnt = _cong_grid->get_bin_cnt_y();
  int x_cnt = _cong_grid->get_bin_cnt_x();
  for (int i = 0; i < x_cnt; i++) {
    if (i == x_cnt - 1) {
      feed << "clo_" << i;
    } else {
      feed << "col_" << i << ",";
    }
  }
  feed << std::endl;
  for (int i = y_cnt - 1; i >= 0; i--) {
    for (int j = 0; j < x_cnt; j++) {
      float net_cong = _cong_grid->get_bin_list()[i * x_cnt + j]->get_net_cong();
      if (j == x_cnt - 1) {
        feed << net_cong;
      } else {
        feed << net_cong << ",";
      }
    }
    feed << std::endl;
  }
  plot << feed.str();
  feed.clear();
  plot.close();
  LOG_INFO << output_file_name + "_" + type + ".csv"
           << " has been created in " << plot_path;
}

double CongestionEval::getBinNetCong(const int& index_x, const int& index_y, const std::string& rudy_type)
{
  int index = index_x + index_y * _cong_grid->get_bin_cnt_x();
  auto bin = _cong_grid->get_bin_list()[index];
  return bin->get_net_cong();
}

void CongestionEval::checkRUDYType(const std::string& rudy_type)
{
  std::set<std::string> rudy_set = {"RUDY", "RUDYDev", "PinRUDY", "PinSteinerRUDY", "SteinerRUDY", "TrueRUDY"};
  auto it = rudy_set.find(rudy_type);
  if (it == rudy_set.end()) {
    LOG_ERROR << rudy_type << " is not be supported in our evaluator";
    LOG_ERROR << "Only the following types are supported: RUDY, RUDYDev, PinRUDY, PinSteinerRUDY, SteinerRUDY, TrueRUDY ";
    LOG_ERROR << "EXIT";
    exit(1);
  } else {
    LOG_INFO << rudy_type << " is selected in Congestion Evaluator";
  }
}

/////////////////////////////////
/////////////////////////////////
/*----evaluate routing congestion----*/
/////////////////////////////////
/////////////////////////////////

std::vector<float> CongestionEval::evalRouteCong()
{
  std::vector<float> result_list;
  result_list.resize(3, 0.0f);

  // eval ACE
  std::vector<float> hor_edge_cong_list;
  std::vector<float> ver_edge_cong_list;
  for (auto& tile : _tile_grid->get_tiles()) {
    float ratio = getUsageCapacityRatio(tile);
    if (ratio >= 0.0f) {
      if (tile->is_horizontal()) {
        hor_edge_cong_list.push_back(ratio);
      } else {
        ver_edge_cong_list.push_back(ratio);
      }
    }
  }
  std::sort(hor_edge_cong_list.rbegin(), hor_edge_cong_list.rend());
  std::sort(ver_edge_cong_list.rbegin(), ver_edge_cong_list.rend());
  result_list[0] = evalACE(hor_edge_cong_list, ver_edge_cong_list);

  // eval TOF and MOF
  std::vector<int> overflow_list = evalOverflow();
  result_list[1] = static_cast<float>(overflow_list[0]);
  result_list[2] = static_cast<float>(overflow_list[1]);

  return result_list;
}

float CongestionEval::evalACE(const std::vector<float>& hor_edge_cong_list, const std::vector<float>& ver_edge_cong_list)
{
  int hor_list_size = hor_edge_cong_list.size();
  int ver_list_size = ver_edge_cong_list.size();

  float hor_avg_005_RC = 0;
  float hor_avg_010_RC = 0;
  float hor_avg_020_RC = 0;
  float hor_avg_050_RC = 0;
  for (int i = 0; i < hor_list_size; ++i) {
    if (i < 0.005 * hor_list_size) {
      hor_avg_005_RC += hor_edge_cong_list[i];
    }
    if (i < 0.01 * hor_list_size) {
      hor_avg_010_RC += hor_edge_cong_list[i];
    }
    if (i < 0.02 * hor_list_size) {
      hor_avg_020_RC += hor_edge_cong_list[i];
    }
    if (i < 0.05 * hor_list_size) {
      hor_avg_050_RC += hor_edge_cong_list[i];
    }
  }
  hor_avg_005_RC /= ceil(0.005 * hor_list_size);
  hor_avg_010_RC /= ceil(0.010 * hor_list_size);
  hor_avg_020_RC /= ceil(0.020 * hor_list_size);
  hor_avg_050_RC /= ceil(0.050 * hor_list_size);

  float ver_avg_005_RC = 0;
  float ver_avg_010_RC = 0;
  float ver_avg_020_RC = 0;
  float ver_avg_050_RC = 0;
  for (int i = 0; i < ver_list_size; ++i) {
    if (i < 0.005 * ver_list_size) {
      ver_avg_005_RC += ver_edge_cong_list[i];
    }
    if (i < 0.01 * ver_list_size) {
      ver_avg_010_RC += ver_edge_cong_list[i];
    }
    if (i < 0.02 * ver_list_size) {
      ver_avg_020_RC += ver_edge_cong_list[i];
    }
    if (i < 0.05 * ver_list_size) {
      ver_avg_050_RC += ver_edge_cong_list[i];
    }
  }
  ver_avg_005_RC /= ceil(0.005 * ver_list_size);
  ver_avg_010_RC /= ceil(0.010 * ver_list_size);
  ver_avg_020_RC /= ceil(0.020 * ver_list_size);
  ver_avg_050_RC /= ceil(0.050 * ver_list_size);

  float k1 = 1.0f;
  float k2 = 1.0f;
  float k3 = 1.0f;
  float k4 = 1.0f;

  float ACE = (k1 * fmax(hor_avg_005_RC, ver_avg_005_RC) + k2 * fmax(hor_avg_010_RC, ver_avg_010_RC)
               + k3 * fmax(hor_avg_020_RC, ver_avg_020_RC) + k4 * fmax(hor_avg_050_RC, ver_avg_050_RC))
              / (k1 + k2 + k3 + k4);
  LOG_INFO << "Average Congestion g-cell Edges(k1=k2=k3=k4=1): " << ACE;
  return ACE;
}

std::vector<int> CongestionEval::evalOverflow()
{
  std::vector<int> overflow_list;

  int all_layer_TOF = 0;
  int all_layer_MOF = 0;
  for (int i = 0; i < _tile_grid->get_num_routing_layers(); ++i) {
    std::vector<int> tmp_overflow_list = evalOverflow(i);
    all_layer_TOF += tmp_overflow_list[0];
    all_layer_MOF = std::max(tmp_overflow_list[1], all_layer_MOF);
  }
  LOG_INFO << "Total Overflow: " << all_layer_TOF;
  LOG_INFO << "Maximum Overflow: " << all_layer_MOF;
  overflow_list.emplace_back(all_layer_TOF);
  overflow_list.emplace_back(all_layer_MOF);

  return overflow_list;
}

std::vector<int> CongestionEval::evalOverflow(int layer_index)
{
  std::vector<int> overflow_list;

  int x_cnt = _tile_grid->get_tile_cnt_x();
  int y_cnt = _tile_grid->get_tile_cnt_y();
  int overflow = 0;
  int single_layer_MOF = 0;
  int single_layer_TOF = 0;
  int start_index = layer_index * x_cnt * y_cnt;

  if (_tile_grid->get_tiles()[start_index]->is_horizontal()) {
    for (int i = y_cnt - 1; i >= 0; i--) {
      for (int j = 0; j < x_cnt; j++) {
        auto tile = _tile_grid->get_tiles()[i * x_cnt + j + start_index];
        overflow = std::max((tile->get_east_use() - tile->get_east_cap()), (tile->get_west_use() - tile->get_west_cap()));
        if (overflow > 0) {
          single_layer_TOF += overflow;
          single_layer_MOF = std::max(overflow, single_layer_MOF);
        }
      }
    }
  } else {
    for (int i = y_cnt - 1; i >= 0; i--) {
      for (int j = 0; j < x_cnt; j++) {
        auto tile = _tile_grid->get_tiles()[i * x_cnt + j + start_index];
        overflow = std::max((tile->get_north_use() - tile->get_north_cap()), (tile->get_south_use() - tile->get_south_cap()));
        if (overflow > 0) {
          single_layer_TOF += overflow;
          single_layer_MOF = std::max(overflow, single_layer_MOF);
        }
      }
    }
  }

  overflow_list.emplace_back(single_layer_TOF);
  overflow_list.emplace_back(single_layer_MOF);

  return overflow_list;
}

std::vector<float> CongestionEval::getUseCapRatioList()
{
  std::vector<float> use_cap_ratio_list;

  int x_cnt = _tile_grid->get_tile_cnt_x();
  int y_cnt = _tile_grid->get_tile_cnt_y();
  int cnt = x_cnt * y_cnt;
  int layer_num = _tile_grid->get_num_routing_layers();
  auto tile_list = _tile_grid->get_tiles();
  use_cap_ratio_list.resize(cnt, 0.0f);

  for (int i = 0; i < layer_num * cnt; ++i) {
    int index = i % cnt;
    float ratio = getUsageCapacityRatio(tile_list[i]);
    if (ratio < 1.0f) {
      ratio = 1.0f;
    }
    use_cap_ratio_list[index] = std::max(ratio, use_cap_ratio_list[index]);
  }

  return use_cap_ratio_list;
}

void CongestionEval::plotGRCong(const string& plot_path, const string& output_file_name)
{
  assert(_tile_grid != nullptr);
  for (int layer_index = 0; layer_index < _tile_grid->get_num_routing_layers(); ++layer_index) {
    plotGRCongOneLayer(plot_path, output_file_name, layer_index);
  }
}

void CongestionEval::plotGRCongOneLayer(const string& plot_path, const string& output_file_name, int layer_index)
{
  // prepare for ploting
  std::ofstream plot(plot_path + output_file_name + std::to_string(layer_index) + ".csv");
  if (!plot.good()) {
    std::cerr << "plot GRCong:: cannot open " << output_file_name << "for writing" << std::endl;
    exit(1);
  }
  std::stringstream feed;
  feed.precision(5);
  int x_cnt = _tile_grid->get_tile_cnt_x();
  int y_cnt = _tile_grid->get_tile_cnt_y();
  for (int i = 0; i < x_cnt; i++) {
    if (i == x_cnt - 1) {
      feed << "col_" << i;
    } else {
      feed << "col_" << i << ",";
    }
  }
  feed << std::endl;

  // plot one layer congestion map
  if (_tile_grid->get_tiles()[layer_index * x_cnt * y_cnt]->is_horizontal()) {
    for (int i = y_cnt - 1; i >= 0; i--) {
      for (int j = 0; j < x_cnt; j++) {
        auto tile = _tile_grid->get_tiles()[i * x_cnt + j + layer_index * x_cnt * y_cnt];
        int congestion_overflow = std::max((tile->get_east_use() - tile->get_east_cap()), (tile->get_west_use() - tile->get_west_cap()));
        congestion_overflow = std::max(0, congestion_overflow);
        if (j == x_cnt - 1) {
          feed << congestion_overflow;
        } else {
          feed << congestion_overflow << ",";
        }
      }
      feed << std::endl;
    }
    LOG_INFO << std::to_string(layer_index) << " is horizontal" << std::endl;
  } else {
    for (int i = y_cnt - 1; i >= 0; i--) {
      for (int j = 0; j < x_cnt; j++) {
        auto tile = _tile_grid->get_tiles()[i * x_cnt + j + layer_index * x_cnt * y_cnt];
        int congestion_overflow
            = std::max((tile->get_north_use() - tile->get_north_cap()), (tile->get_south_use() - tile->get_south_cap()));
        congestion_overflow = std::max(0, congestion_overflow);
        if (j == x_cnt - 1) {
          feed << congestion_overflow;
        } else {
          feed << congestion_overflow << ",";
        }
      }
      feed << std::endl;
    }
    LOG_INFO << std::to_string(layer_index) << " is vertical" << std::endl;
  }

  plot << feed.str();
  feed.clear();
  plot.close();
  LOG_INFO << output_file_name + std::to_string(layer_index) + ".csv"
           << " has been created in " << plot_path;
}

void CongestionEval::plotOverflow(const std::string& plot_path, const std::string& output_file_name)
{
  int x_cnt = _tile_grid->get_tile_cnt_x();
  int y_cnt = _tile_grid->get_tile_cnt_y();
  int cnt = x_cnt * y_cnt;
  int layer_num = _tile_grid->get_num_routing_layers();
  auto tile_list = _tile_grid->get_tiles();

  std::vector<int> total_plane_grid;
  std::vector<int> max_plane_grid;
  total_plane_grid.resize(cnt, 0);
  max_plane_grid.resize(cnt, 0);

  for (int i = 0; i < layer_num * cnt; ++i) {
    int index = i % cnt;
    if (tile_list[i]->is_horizontal()) {
      int overflow = std::max((tile_list[i]->get_east_use() - tile_list[i]->get_east_cap()),
                              (tile_list[i]->get_west_use() - tile_list[i]->get_west_cap()));
      if (overflow > 0) {
        total_plane_grid[index] += overflow;
        max_plane_grid[index] = std::max(overflow, max_plane_grid[index]);
      }
    } else {
      int overflow = std::max((tile_list[i]->get_north_use() - tile_list[i]->get_north_cap()),
                              (tile_list[i]->get_south_use() - tile_list[i]->get_south_cap()));
      if (overflow > 0) {
        total_plane_grid[index] += overflow;
        max_plane_grid[index] = std::max(overflow, max_plane_grid[index]);
      }
    }
  }

  plotOverflow(plot_path, output_file_name, total_plane_grid, x_cnt, "TOF");
  plotOverflow(plot_path, output_file_name, max_plane_grid, x_cnt, "MOF");
}

void CongestionEval::plotOverflow(const std::string& plot_path, const std::string& output_file_name, const std::vector<int>& plane_grid,
                                  const int& x_cnt, const std::string& type)
{
  std::ofstream plot(plot_path + output_file_name + type + ".csv");
  if (!plot.good()) {
    std::cerr << "plot overflow:: cannot open " << output_file_name << "for writing" << std::endl;
    exit(1);
  }
  std::stringstream feed;
  feed.precision(5);
  for (int i = 0; i < x_cnt; i++) {
    if (i == x_cnt - 1) {
      feed << "col_" << i;
    } else {
      feed << "col_" << i << ",";
    }
  }
  feed << std::endl;

  int y_cnt = plane_grid.size() / x_cnt;
  for (int i = y_cnt - 1; i >= 0; i--) {
    for (int j = 0; j < x_cnt; j++) {
      int overflow = plane_grid[i * x_cnt + j];
      if (j == x_cnt - 1) {
        feed << overflow;
      } else {
        feed << overflow << ",";
      }
    }
    feed << std::endl;
  }

  plot << feed.str();
  feed.clear();
  plot.close();
  LOG_INFO << output_file_name + type + ".csv"
           << " has been created in " << plot_path;
}

void CongestionEval::reportCongMap()
{
  LOG_INFO << "Evaluator: Final congestion report: ";
  LOG_INFO << "Layer_ID         Resource        Demand        Usage (%)    Max H / Max V "
              "/ Total Overflow";
  LOG_INFO << "------------------------------------------------------------------------"
              "---------------";

  std::vector<int> cap_per_layer;
  std::vector<int> use_per_layer;
  std::vector<int> overflow_per_layer;
  std::vector<int> max_h_per_layer;
  std::vector<int> max_v_per_layer;

  cap_per_layer.resize(_tile_grid->get_num_routing_layers());
  use_per_layer.resize(_tile_grid->get_num_routing_layers());
  overflow_per_layer.resize(_tile_grid->get_num_routing_layers());
  max_h_per_layer.resize(_tile_grid->get_num_routing_layers());
  max_v_per_layer.resize(_tile_grid->get_num_routing_layers());

  for (int layer_index = 0; layer_index < _tile_grid->get_num_routing_layers(); layer_index++) {
    cap_per_layer[layer_index] = 0;
    use_per_layer[layer_index] = 0;
    overflow_per_layer[layer_index] = 0;
    max_h_per_layer[layer_index] = 0;
    max_v_per_layer[layer_index] = 0;

    for (int i = 0; i < _tile_grid->get_tile_cnt_y(); i++) {
      for (int j = 0; j < _tile_grid->get_tile_cnt_x(); j++) {
        auto tile = _tile_grid->get_tiles()[i + j];
        cap_per_layer[layer_index] += std::min(tile->get_east_cap(), tile->get_west_cap());
        use_per_layer[layer_index] += std::max(tile->get_east_use(), tile->get_west_use());

        const int overflow = std::max(tile->get_east_use(), tile->get_west_use()) - std::min(tile->get_east_cap(), tile->get_west_cap());
        if (overflow > 0) {
          overflow_per_layer[layer_index] += overflow;
          max_h_per_layer[layer_index] = std::max(max_h_per_layer[layer_index], overflow);
        }
      }
    }
    for (int i = 0; i < _tile_grid->get_tile_cnt_y(); i++) {
      for (int j = 0; j < _tile_grid->get_tile_cnt_x(); j++) {
        auto tile = _tile_grid->get_tiles()[i + j];
        cap_per_layer[layer_index] += std::min(tile->get_south_cap(), tile->get_north_cap());
        use_per_layer[layer_index] += std::max(tile->get_south_use(), tile->get_north_use());

        const int overflow
            = std::max(tile->get_south_use(), tile->get_north_use()) - std::min(tile->get_south_cap(), tile->get_north_cap());
        if (overflow > 0) {
          overflow_per_layer[layer_index] += overflow;
          max_v_per_layer[layer_index] = std::max(max_v_per_layer[layer_index], overflow);
        }
      }
    }
  }

  int total_cap = 0;
  int total_use = 0;
  int total_overflow = 0;
  int total_max_h = 0;
  int total_max_v = 0;

  for (int layer_index = 0; layer_index < _tile_grid->get_num_routing_layers(); layer_index++) {
    std::cout << layer_index << "         " << cap_per_layer[layer_index] << "        " << use_per_layer[layer_index] << "        "
              << (double) use_per_layer[layer_index] / cap_per_layer[layer_index] * 100 << "%    " << max_h_per_layer[layer_index] << " / "
              << max_v_per_layer[layer_index] << " / " << overflow_per_layer[layer_index] << std::endl;

    total_cap += cap_per_layer[layer_index];
    total_use += use_per_layer[layer_index];
    total_overflow += overflow_per_layer[layer_index];
    total_max_h += max_h_per_layer[layer_index];
    total_max_v += max_v_per_layer[layer_index];
  }
  // todo: each layer info
  std::cout << "------------------------------------------------------------------------"
               "---------------"
            << std::endl;
  std::cout << "Total"
            << "         " << total_cap << "        " << total_use << "        " << (double) total_use / total_cap * 100 << "%    "
            << total_max_h << " / " << total_max_v << " / " << total_overflow << std::endl;
}

/////////////////////////////////
/////////////////////////////////
/*----Common used----*/
/////////////////////////////////
/////////////////////////////////
void CongestionEval::set_tile_grid(const int& lx, const int& ly, const int& tileCntX, const int& tileCntY, const int& tileSizeX,
                                   const int& tileSizeY, const int& numRoutingLayers)
{
  _tile_grid = new TileGrid(lx, ly, tileCntX, tileCntY, tileSizeX, tileSizeY, numRoutingLayers);
  // _tile_grid->initTiles();
}

void CongestionEval::set_cong_grid(const int& lx, const int& ly, const int& binCntX, const int& binCntY, const int& binSizeX,
                                   const int& binSizeY)
{
  _cong_grid = new CongGrid(lx, ly, binCntX, binCntY, binSizeX, binSizeY);
  _cong_grid->initBins();
}

void CongestionEval::reportTileGrid()
{
  LOG_INFO << "tile_grid_lx: " << _tile_grid->get_lx();
  LOG_INFO << "tile_grid_ly: " << _tile_grid->get_ly();
  LOG_INFO << "tile_cnt_x: " << _tile_grid->get_tile_cnt_x();
  LOG_INFO << "tile_cnt_y: " << _tile_grid->get_tile_cnt_y();
  LOG_INFO << "tile_size_x: " << _tile_grid->get_tile_size_x();
  LOG_INFO << "tile_size_y: " << _tile_grid->get_tile_size_y();
  LOG_INFO << "numRoutingLayers: " << _tile_grid->get_num_routing_layers();
}

void CongestionEval::reportCongGrid()
{
  LOG_INFO << "bin_grid_lx: " << _cong_grid->get_lx();
  LOG_INFO << "bin_grid_ly: " << _cong_grid->get_ly();
  LOG_INFO << "bin_cnt_x: " << _cong_grid->get_bin_cnt_x();
  LOG_INFO << "bin_cnt_y: " << _cong_grid->get_bin_cnt_y();
  LOG_INFO << "bin_size_x: " << _cong_grid->get_bin_size_x();
  LOG_INFO << "bin_size_y: " << _cong_grid->get_bin_size_y();
  LOG_INFO << "number_routing_layers: " << _cong_grid->get_routing_layers_number();
}

// void CongestionEval::buildIDB(std::vector<std::string> lef_file_path_list, std::string def_file_path)
// {
//   idb::IdbBuilder* _idb_builder = new idb::IdbBuilder();
//   _idb_builder->buildLef(lef_file_path_list);
//   _idb_builder->buildDef(def_file_path);
// }

/////////////////////////////////
/////////////////////////////////
/*----private functions----*/
/////////////////////////////////
/////////////////////////////////
int32_t CongestionEval::getOverlapArea(CongBin* bin, CongInst* inst)
{
  int32_t rect_lx = std::max((int64_t) bin->get_lx(), inst->get_lx());
  int32_t rect_ly = std::max((int64_t) bin->get_ly(), inst->get_ly());
  int32_t rect_ux = std::min((int64_t) bin->get_ux(), inst->get_ux());
  int32_t rect_uy = std::min((int64_t) bin->get_uy(), inst->get_uy());

  if (rect_lx >= rect_ux || rect_ly >= rect_uy) {
    return 0;
  } else {
    return (rect_ux - rect_lx) * (rect_uy - rect_ly);
  }
}

int32_t CongestionEval::getOverlapArea(CongBin* bin, CongNet* net)
{
  int32_t rect_lx = std::max((int64_t) bin->get_lx(), net->get_lx());
  int32_t rect_ly = std::max((int64_t) bin->get_ly(), net->get_ly());
  int32_t rect_ux = std::min((int64_t) bin->get_ux(), net->get_ux());
  int32_t rect_uy = std::min((int64_t) bin->get_uy(), net->get_uy());

  if (rect_lx > rect_ux || rect_ly > rect_uy) {
    return 0;
  } else if (rect_lx == rect_ux) {
    return rect_uy - rect_ly;
  } else if (rect_ly == rect_uy) {
    return rect_ux - rect_lx;
  } else {
    return (rect_ux - rect_lx) * (rect_uy - rect_ly);
  }
}

// this idea is from the paper "Fast and Accurate Routing Demand Estimation for Efficient Routability-driven Placement"
double CongestionEval::getRudy(CongBin* bin, CongNet* net, DIRECTION direction)
{
  double horizontal = 0.0;
  double vertical = 0.0;

  if (net->get_height() == 0 || net->get_width() == 0) {
    return 1;
  }

  if (net->get_height() != 0) {
    horizontal = bin->get_average_wire_width() / static_cast<double>(net->get_height());
    if (direction == DIRECTION::kH) {
      return horizontal;
    }
  }
  if (net->get_width() != 0) {
    vertical = bin->get_average_wire_width() / static_cast<double>(net->get_width());
    if (direction == DIRECTION::kV) {
      return vertical;
    }
  }

  return horizontal + vertical;
}

// this idea is from the paper "Routability-Driven Analytical Placement by Net Overlapping Removal for
// Large-Scale Mixed-Size Designs"
double CongestionEval::getRudyDev(CongBin* bin, CongNet* net)
{
  double horizontal = 0.0;
  double vertical = 0.0;
  int64_t net_height = net->get_height();
  int64_t net_width = net->get_width();

  if (net_height == 0 || net_width == 0) {
    return 1;
  }

  if (net_height != 0) {
    int bin_size_y = _cong_grid->get_bin_size_y();
    int capacity_hor = bin->get_horizontal_capacity();
    horizontal = bin_size_y / capacity_hor / static_cast<double>(net_height);
  }

  if (net->get_width() != 0) {
    int bin_size_x = _cong_grid->get_bin_size_x();
    int capacity_ver = bin->get_vertical_capacity();
    vertical = bin_size_x / capacity_ver / static_cast<double>(net_width);
  }

  return horizontal + vertical;
}

// this idea is from the paper "Global Placement with Deep Learning-Enabled Explicit Routability Optimization"
double CongestionEval::getPinRudy(CongBin* bin, CongNet* net, DIRECTION direction)
{
  double horizontal = 0.0;
  double vertical = 0.0;
  int64_t net_height = net->get_height();
  int64_t net_width = net->get_width();

  for (auto& pin : net->get_pin_list()) {
    auto pin_x = pin->get_x();
    auto pin_y = pin->get_y();
    if (pin_x > bin->get_lx() && pin_x < bin->get_ux() && pin_y > bin->get_ly() && pin_y < bin->get_uy()) {
      if (net_height != 0) {
        horizontal += 1 / static_cast<double>(net_height);
      }
      if (net_width != 0) {
        vertical += 1 / static_cast<double>(net_width);
      }
    }
  }

  return horizontal + vertical;
}

double CongestionEval::getPinSteinerRudy(CongBin* bin, CongNet* net, const std::map<std::string, int64_t>& map)
{
  double result = 0.0;
  int64_t net_height = net->get_height();
  int64_t net_width = net->get_width();

  int64_t flute_wl = 0;
  auto net_pair = map.find(net->get_name());
  if (net_pair != map.end()) {
    flute_wl = net_pair->second;
  } else {
    flute_wl = 0;
  }

  for (auto& pin : net->get_pin_list()) {
    auto pin_x = pin->get_x();
    auto pin_y = pin->get_y();
    if (pin_x > bin->get_lx() && pin_x < bin->get_ux() && pin_y > bin->get_ly() && pin_y < bin->get_uy()) {
      if (net_height != 0 && net_width != 0) {
        result += flute_wl / static_cast<double>(net_height) / net_width;
      }
    }
  }

  return result;
}

double CongestionEval::getSteinerRudy(CongBin* bin, CongNet* net, const std::map<std::string, int64_t>& map)
{
  double result = 0.0;
  int64_t net_height = net->get_height();
  int64_t net_width = net->get_width();

  int64_t flute_wl = 0;
  auto net_pair = map.find(net->get_name());
  if (net_pair != map.end()) {
    flute_wl = net_pair->second;
  } else {
    flute_wl = 0;
  }

  if (net_height == 0 || net_width == 0) {
    result = 1;
  } else {
    result = flute_wl / static_cast<double>(net_height) / net_width * bin->get_average_wire_width();
  }
  std::cout << "steiner rudy :" << result << std::endl;
  return result;
}

double CongestionEval::getTrueRudy(CongBin* bin, CongNet* net, const std::map<std::string, int64_t>& map)
{
  double result = 0.0;
  int64_t net_height = net->get_height();
  int64_t net_width = net->get_width();

  int64_t true_wl = 0;
  auto net_pair = map.find(net->get_name());
  if (net_pair != map.end()) {
    true_wl = net_pair->second;
  } else {
    true_wl = 0;
  }

  if (net_height == 0 || net_width == 0) {
    result = 1;
  } else {
    result = true_wl / static_cast<double>(net_height) / net_width * bin->get_average_wire_width();
  }
  std::cout << "ture rudy :" << result << std::endl;
  return result;
}

float CongestionEval::getUsageCapacityRatio(Tile* tile)
{
  unsigned int cap_N = 0, cap_S = 0, cap_E = 0, cap_W = 0;
  unsigned int use_N = 0, use_S = 0, use_E = 0, use_W = 0;

  cap_N = tile->get_north_cap();
  cap_S = tile->get_south_cap();
  cap_E = tile->get_east_cap();
  cap_W = tile->get_west_cap();

  use_N = tile->get_north_use();
  use_S = tile->get_south_use();
  use_E = tile->get_east_use();
  use_W = tile->get_west_use();

  bool isHorizontal = tile->is_horizontal();
  unsigned int curCap = (isHorizontal) ? std::min(cap_E, cap_W) : std::min(cap_N, cap_S);  //取平均
  unsigned int curUse = (isHorizontal) ? std::max(use_E, use_W) : std::max(use_N, use_S);

  // escape tile ratio cals when capacity = 0
  if (curCap == 0) {
    return -1 * FLT_MAX;
  }

  // return usage (used routing track + blockage + via number) / total capacity
  return static_cast<float>(curUse) / curCap;
}

}  // namespace eval
