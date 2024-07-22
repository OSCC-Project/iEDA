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
 * @file		feature_builder.h
 * @date		13/05/2024
 * @version		0.1
 * @description


        build feature data
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "feature_builder.h"

#include <map>

#include "flow_config.h"
#include "idm.h"

namespace ieda_feature {

FeatureBuilder::FeatureBuilder()
{
}

FeatureBuilder::~FeatureBuilder()
{
}

DBSummary FeatureBuilder::buildDBSummary()
{
  DBSummary summary_db;
  summary_db.info = buildSummaryInfo();
  summary_db.layout = buildSummaryLayout();
  summary_db.statis = buildSummaryStatis();
  summary_db.instances = buildSummaryInstances();
  summary_db.nets = buildSummaryNets();
  summary_db.layers = buildSummaryLayers();
  summary_db.pins = buildSummaryPins();

  return summary_db;
}

SummaryInfo FeatureBuilder::buildSummaryInfo()
{
  SummaryInfo info;

  info.eda_tool = "iEDA";
  info.eda_version = iplf::flowConfigInst->get_env_info_software_version();
  info.design_name = dmInst->get_idb_design()->get_design_name();
  info.design_version = dmInst->get_idb_design()->get_version();
  info.flow_stage = iplf::flowConfigInst->get_status_stage();
  info.flow_runtime = iplf::flowConfigInst->get_status_runtime_string();
  info.flow_memory = iplf::flowConfigInst->get_status_memmory_string();

  return info;
}

SummaryLayout FeatureBuilder::buildSummaryLayout()
{
  SummaryLayout layout;
  std::memset(&layout, 0, sizeof(SummaryLayout));

  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = dmInst->get_idb_layout();

  int dbu = idb_design->get_units()->get_micron_dbu() < 0 ? idb_layout->get_units()->get_micron_dbu()
                                                          : idb_design->get_units()->get_micron_dbu();

  layout.design_dbu = dbu;
  layout.die_area = dmInst->dieAreaUm();
  layout.die_usage = dmInst->dieUtilization();
  layout.die_bounding_width = ((double) idb_layout->get_die()->get_width()) / dbu;
  layout.die_bounding_height = ((double) idb_layout->get_die()->get_height()) / dbu;
  layout.core_area = dmInst->coreAreaUm();
  layout.core_usage = dmInst->coreUtilization();
  layout.core_bounding_width = ((double) idb_layout->get_core()->get_bounding_box()->get_width()) / dbu;
  layout.core_bounding_height = ((double) idb_layout->get_core()->get_bounding_box()->get_height()) / dbu;

  return layout;
}

SummaryStatis FeatureBuilder::buildSummaryStatis()
{
  SummaryStatis summary_statis;
  std::memset(&summary_statis, 0, sizeof(SummaryStatis));

  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = dmInst->get_idb_layout();

  summary_statis.num_layers = idb_layout->get_layers()->get_layers_num();
  summary_statis.num_layers_routing = idb_layout->get_layers()->get_routing_layers_number();
  summary_statis.num_layers_cut = idb_layout->get_layers()->get_cut_layers_number();
  summary_statis.num_iopins = idb_design->get_io_pin_list()->get_pin_num();
  summary_statis.num_instances = idb_design->get_instance_list()->get_num();
  summary_statis.num_nets = idb_design->get_net_list()->get_num();
  summary_statis.num_pdn = idb_design->get_special_net_list()->get_num();

  return summary_statis;
}

SummaryInstances FeatureBuilder::buildSummaryInstances()
{
  SummaryInstances summary_insts;
  std::memset(&summary_insts, 0, sizeof(SummaryInstances));

  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = dmInst->get_idb_layout();

  int dbu = idb_design->get_units()->get_micron_dbu() < 0 ? idb_layout->get_units()->get_micron_dbu()
                                                          : idb_design->get_units()->get_micron_dbu();

  double die_area = ((double) idb_layout->get_die()->get_area()) / dbu / dbu;
  double core_area = ((double) idb_layout->get_core()->get_bounding_box()->get_area()) / dbu / dbu;

  /// all instances
  summary_insts.total.num = idb_design->get_instance_list()->get_num();
  summary_insts.total.num_ratio = (double) (1);  /// all instance
  summary_insts.total.area = idb_design->get_instance_list()->get_area() / dbu / dbu;
  summary_insts.total.area_ratio = (double) (1);
  summary_insts.total.die_usage = (double) (summary_insts.total.area / die_area);
  summary_insts.total.core_usage = (double) (summary_insts.total.area / core_area);
  summary_insts.total.pin_num = idb_design->get_instance_list()->get_connected_pin_num();
  summary_insts.total.pin_ratio = (double) (1);

  /// all iopads
  summary_insts.iopads.num = idb_design->get_instance_list()->get_num_pad();
  summary_insts.iopads.num_ratio = ((double) summary_insts.iopads.num) / summary_insts.total.num;
  summary_insts.iopads.area = ((double) idb_design->get_instance_list()->get_area_pad()) / dbu / dbu;
  summary_insts.iopads.area_ratio = ((double) summary_insts.iopads.area) / summary_insts.total.area;
  summary_insts.iopads.die_usage = (double) (summary_insts.iopads.area / die_area);
  summary_insts.iopads.core_usage = (double) (summary_insts.iopads.area / core_area);
  summary_insts.iopads.pin_num = idb_design->get_instance_list()->get_iopads_pin_num();
  summary_insts.iopads.pin_ratio = ((double) summary_insts.iopads.pin_num) / summary_insts.total.pin_num;

  /// all macros
  summary_insts.macros.num = idb_design->get_instance_list()->get_num_block();
  summary_insts.macros.num_ratio = ((double) summary_insts.macros.num) / summary_insts.total.num;
  summary_insts.macros.area = ((double) idb_design->get_instance_list()->get_area_block()) / dbu / dbu;
  summary_insts.macros.area_ratio = ((double) summary_insts.macros.area) / summary_insts.total.area;
  summary_insts.macros.die_usage = (double) (summary_insts.macros.area / die_area);
  summary_insts.macros.core_usage = (double) (summary_insts.macros.area / core_area);
  summary_insts.macros.pin_num = idb_design->get_instance_list()->get_macro_pin_num();
  summary_insts.macros.pin_ratio = ((double) summary_insts.macros.pin_num) / summary_insts.total.pin_num;

  /// all logic cell including connected with signal net, clock net
  summary_insts.logic.num = idb_design->get_instance_list()->get_num_core_logic();
  summary_insts.logic.num_ratio = ((double) summary_insts.logic.num) / summary_insts.total.num;
  summary_insts.logic.area = ((double) idb_design->get_instance_list()->get_area_core_logic()) / dbu / dbu;
  summary_insts.logic.area_ratio = ((double) summary_insts.logic.area) / summary_insts.total.area;
  summary_insts.logic.die_usage = (double) (summary_insts.logic.area / die_area);
  summary_insts.logic.core_usage = (double) (summary_insts.logic.area / core_area);
  summary_insts.logic.pin_num = idb_design->get_instance_list()->get_logic_pin_num();
  summary_insts.logic.pin_ratio = ((double) summary_insts.logic.pin_num) / summary_insts.total.pin_num;

  /// all clock tree cell
  summary_insts.clock.num = idb_design->get_instance_list()->get_num_clockcell();
  summary_insts.clock.num_ratio = ((double) summary_insts.clock.num) / summary_insts.total.num;
  summary_insts.clock.area = ((double) idb_design->get_instance_list()->get_area_clock()) / dbu / dbu;
  summary_insts.clock.area_ratio = ((double) summary_insts.clock.area) / summary_insts.total.area;
  summary_insts.clock.die_usage = (double) (summary_insts.clock.area / die_area);
  summary_insts.clock.core_usage = (double) (summary_insts.clock.area / core_area);
  summary_insts.clock.pin_num = idb_design->get_instance_list()->get_clock_pin_num();
  summary_insts.clock.pin_ratio = ((double) summary_insts.clock.pin_num) / summary_insts.total.pin_num;

  return summary_insts;
}

SummaryNets FeatureBuilder::buildSummaryNets()
{
  SummaryNets summary_nets;
  std::memset(&summary_nets, 0, sizeof(SummaryNets));

  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = dmInst->get_idb_layout();

  int dbu = idb_design->get_units()->get_micron_dbu() < 0 ? idb_layout->get_units()->get_micron_dbu()
                                                          : idb_design->get_units()->get_micron_dbu();

  summary_nets.num_total = idb_design->get_net_list()->get_num();
  summary_nets.num_signal = idb_design->get_net_list()->get_num_signal();
  summary_nets.num_clock = idb_design->get_net_list()->get_num_clock();
  summary_nets.num_pins = idb_design->get_net_list()->get_pin_num();
  summary_nets.num_segment = idb_design->get_net_list()->get_segment_num();
  summary_nets.num_via = idb_design->get_net_list()->get_via_num();
  summary_nets.num_wire = idb_design->get_net_list()->get_segment_wire_num();
  summary_nets.num_patch = idb_design->get_net_list()->get_patch_num();

  summary_nets.wire_len = ((double) dmInst->allNetLength()) / dbu;
  summary_nets.wire_len_signal = ((double) dmInst->getSignalNetListLength()) / dbu;
  summary_nets.ratio_signal = ((double) summary_nets.wire_len_signal) / summary_nets.wire_len;
  summary_nets.wire_len_clock = ((double) dmInst->getClockNetListLength()) / dbu;
  summary_nets.ratio_clock = ((double) summary_nets.wire_len_clock) / summary_nets.wire_len;

  return summary_nets;
}

SummaryLayers FeatureBuilder::buildSummaryLayers()
{
  SummaryLayers summary_layers;

  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = dmInst->get_idb_layout();

  int dbu = idb_design->get_units()->get_micron_dbu() < 0 ? idb_layout->get_units()->get_micron_dbu()
                                                          : idb_design->get_units()->get_micron_dbu();

  std::map<int, SummaryLayerRouting> routing_layer_map;
  std::map<int, SummaryLayerCut> cut_layer_map;

  bool find_routing_layer = false;
  for (auto layer : idb_layout->get_layers()->get_layers()) {
    if (layer->is_routing()) {
      find_routing_layer = true;

      SummaryLayerRouting layer_routing;
      layer_routing.layer_name = layer->get_name();
      layer_routing.layer_order = layer->get_order();
      layer_routing.wire_len = 0;
      layer_routing.wire_num = 0;
      layer_routing.patch_num = 0;

      routing_layer_map.insert(std::make_pair(layer->get_order(), layer_routing));
    }

    if (layer->is_cut() && find_routing_layer) {
      SummaryLayerCut layer_cut;
      layer_cut.layer_name = layer->get_name();
      layer_cut.layer_order = layer->get_order();
      layer_cut.via_num = 0;

      cut_layer_map.insert(std::make_pair(layer->get_order(), layer_cut));
    }
  }

  double wire_total = 0;
  uint64_t via_total = 0;
  for (auto net : idb_design->get_net_list()->get_net_list()) {
    for (auto wire : net->get_wire_list()->get_wire_list()) {
      for (auto segment : wire->get_segment_list()) {
        if (segment->is_wire()) {
          int order = segment->get_layer()->get_order();
          routing_layer_map[order].wire_num += 1;
          routing_layer_map[order].wire_len += ((double) segment->length()) / dbu;

          wire_total += ((double) segment->length()) / dbu;
        }

        if (segment->is_rect()) {
          int order = segment->get_layer()->get_order();
          routing_layer_map[order].patch_num += 1;
          routing_layer_map[order].wire_len += ((double) segment->length()) / dbu;

          wire_total += ((double) segment->length()) / dbu;
        }

        if (segment->is_via()) {
          auto via_list = segment->get_via_list();
          for (auto via : via_list) {
            auto layer_shape = via->get_cut_layer_shape();

            int order = layer_shape.get_layer()->get_order();
            cut_layer_map[order].via_num += 1;

            via_total += 1;
          }
        }
      }
    }
  }

  for (auto special_net : idb_design->get_special_net_list()->get_net_list()) {
    for (auto special_wire : special_net->get_wire_list()->get_wire_list()) {
      for (auto special_segment : special_wire->get_segment_list()) {
        if (special_segment->is_via()) {
          auto via = special_segment->get_via();
          auto layer_shape = via->get_cut_layer_shape();
          int order = layer_shape.get_layer()->get_order();
          cut_layer_map[order].via_num += 1;

          via_total += 1;
        }
      }
    }
  }

  for (auto layer_iter = routing_layer_map.begin(); layer_iter != routing_layer_map.end(); layer_iter++) {
    auto layer = (*layer_iter).second;
    layer.wire_ratio = wire_total == 0 ? 0 : ((double) layer.wire_len) / wire_total;
    summary_layers.routing_layers.push_back(layer);
  }

  for (auto layer_iter = cut_layer_map.begin(); layer_iter != cut_layer_map.end(); layer_iter++) {
    auto layer = (*layer_iter).second;
    layer.via_ratio = via_total == 0 ? 0 : ((double) layer.via_num) / via_total;

    summary_layers.cut_layers.push_back(layer);
  }

  summary_layers.num_layers = idb_layout->get_layers()->get_layers_num();
  summary_layers.num_layers_routing = idb_layout->get_layers()->get_routing_layers_number();
  summary_layers.num_layers_cut = idb_layout->get_layers()->get_cut_layers_number();

  return summary_layers;
}

SummaryPins FeatureBuilder::buildSummaryPins()
{
  SummaryPins summary_pins;

  auto idb_design = dmInst->get_idb_design();

  summary_pins.max_fanout = 32;
  for (int i = 0; i <= summary_pins.max_fanout + 1; i++) {
    SummaryPin item;
    memset(&item, 0, sizeof(SummaryPin));
    item.pin_num = i;
    summary_pins.pin_distribution.push_back(item);
  }

  int net_total = idb_design->get_net_list()->get_net_list().size();

  for (auto net : idb_design->get_net_list()->get_net_list()) {
    auto pin_num = net->get_pin_number();
    if (pin_num >= 0 && pin_num <= summary_pins.max_fanout) {
      summary_pins.pin_distribution[pin_num].net_num += 1;
    } else {
      /// add to last item
      summary_pins.pin_distribution[summary_pins.max_fanout + 1].net_num += 1;
    }
  }

  for (int i = 0; i < (int) summary_pins.pin_distribution.size(); i++) {
    summary_pins.pin_distribution[i].net_ratio = ((double) summary_pins.pin_distribution[i].net_num) / net_total;
  }

  int inst_total = idb_design->get_instance_list()->get_instance_list().size();

  for (auto inst : idb_design->get_instance_list()->get_instance_list()) {
    auto pin_num = inst->get_connected_pin_number();
    if (pin_num >= 0 && pin_num <= summary_pins.max_fanout) {
      summary_pins.pin_distribution[pin_num].inst_num += 1;
    } else {
      /// add to last item
      summary_pins.pin_distribution[summary_pins.max_fanout + 1].inst_num += 1;
    }
  }

  for (int i = 0; i < (int) summary_pins.pin_distribution.size(); i++) {
    summary_pins.pin_distribution[i].inst_ratio = ((double) summary_pins.pin_distribution[i].inst_num) / inst_total;
  }

  return summary_pins;
}

}  // namespace ieda_feature