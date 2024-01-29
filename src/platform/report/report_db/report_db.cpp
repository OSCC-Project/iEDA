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

#include "report_db.h"

#include "ReportTable.hh"
#include "flow_config.h"
#include "idm.h"

namespace iplf {

std::ostream& operator<<(std::ostream& os, ReportDanglingNet& report)
{
  auto* netlist = dmInst->get_idb_design()->get_net_list();
  for (auto* net : netlist->get_net_list()) {
    if (net->get_pin_number() == 0) {
      report.count();
      os << "Net " << net->get_net_name() << std::endl;
    }
  }
  return os;
}

std::string ReportDB::title()
{
  //   auto design = dmInst->get_idb_design();
  //   std::string title;
  //   std::string name = design->get_design_name();
  //   std::string version = design->get_version();

  //   std::vector<std::string> header_list = {"iEDA", flowConfigInst->get_env_info_software_version()};
  //   auto tbl = std::make_shared<ieda::ReportTable>("Design Info", header_list, static_cast<int>(ReportDBType::kTitle));

  //   *tbl << "Flow Stage" << flowConfigInst->get_status_stage() << TABLE_ENDLINE;

  //   *tbl << TABLE_SKIP << TABLE_SKIP << TABLE_ENDLINE;

  //   *tbl << "Flow Runtime" << flowConfigInst->get_status_runtime_string() << TABLE_ENDLINE;
  //   *tbl << "Memmory" << flowConfigInst->get_status_memmory_string() << TABLE_ENDLINE;

  //   *tbl << TABLE_SKIP << TABLE_SKIP << TABLE_ENDLINE;

  //   *tbl << "Design Name" << name << TABLE_ENDLINE;
  //   *tbl << "DEF&LEF Version" << version << TABLE_ENDLINE;
  //   *tbl << "DBU" << design->get_units()->get_micron_dbu() << TABLE_ENDLINE;

  //   return tbl->to_string();
  return ReportBase::title();
}

std::shared_ptr<ieda::ReportTable> ReportDB::createSummaryTable()
{
  std::vector<std::string> header_list = {"Module", "Value"};
  auto tbl = std::make_shared<ieda::ReportTable>("Summary", header_list, static_cast<int>(ReportDBType::kSummary));
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();

  int dbu = idb_design->get_units()->get_micron_dbu() < 0 ? idb_layout->get_units()->get_micron_dbu()
                                                          : idb_design->get_units()->get_micron_dbu();

  auto* idb_die = idb_layout->get_die();
  auto die_width = ((double) idb_die->get_width()) / dbu;
  auto die_height = ((double) idb_die->get_height()) / dbu;
  *tbl << "DIE Area ( um^2 )" << ieda::Str::printf("%f = %03f * %03f", die_width * die_height, die_width, die_height) << TABLE_ENDLINE;
  *tbl << "DIE Usage" << ieda::Str::printf("%f", dmInst->dieUtilization()) << TABLE_ENDLINE;

  /// Core
  auto idb_core_box = idb_layout->get_core()->get_bounding_box();
  auto core_width = ((double) idb_core_box->get_width()) / dbu;
  auto core_height = ((double) idb_core_box->get_height()) / dbu;
  *tbl << "CORE Area ( um^2 )" << ieda::Str::printf("%f = %03f * %03f", core_width * core_height, core_width, core_height) << TABLE_ENDLINE;
  *tbl << "CORE Usage" << ieda::Str::printf("%f", dmInst->coreUtilization()) << TABLE_ENDLINE;

  *tbl << TABLE_SKIP << TABLE_SKIP << TABLE_ENDLINE;
  /// site
  *tbl << "Number - Site" << idb_layout->get_sites()->get_sites_num() << TABLE_ENDLINE;

  /// row
  *tbl << "Number - Row" << idb_layout->get_rows()->get_row_num() << TABLE_ENDLINE;

  /// track
  *tbl << "Number - Track" << idb_layout->get_track_grid_list()->get_track_grid_num() << TABLE_ENDLINE;

  /// layer
  *tbl << "Number - Layer" << idb_layout->get_layers()->get_layers_num() << TABLE_ENDLINE;
  *tbl << "Number - Routing Layer" << idb_layout->get_layers()->get_routing_layers_number() << TABLE_ENDLINE;
  *tbl << "Number - Cut Layer" << idb_layout->get_layers()->get_cut_layers_number() << TABLE_ENDLINE;

  /// Gcell
  *tbl << "Number - GCell Grid" << idb_layout->get_gcell_grid_list()->get_gcell_grid_num() << TABLE_ENDLINE;

  /// cell master
  *tbl << "Number - Cell Master" << idb_layout->get_cell_master_list()->get_cell_master_num() << TABLE_ENDLINE;

  /// Via Rules
  *tbl << "Number - Via Rule"
       << idb_layout->get_via_rule_list()->get_num_via_rule_generate() + idb_layout->get_via_list()->get_num_via()
              + idb_design->get_via_list()->get_num_via()
       << TABLE_ENDLINE;

  *tbl << TABLE_SKIP << TABLE_SKIP << TABLE_ENDLINE;

  /// IO Pin
  *tbl << "Number - IO Pin" << idb_design->get_io_pin_list()->get_pin_num() << TABLE_ENDLINE;

  /// instance
  *tbl << "Number - Instance" << idb_design->get_instance_list()->get_num() << TABLE_ENDLINE;

  /// Blockage
  *tbl << "Number - Blockage" << idb_design->get_blockage_list()->get_num() << TABLE_ENDLINE;

  /// Filler
  *tbl << "Number - Filler" << idb_design->get_fill_list()->get_num_fill() << TABLE_ENDLINE;

  /// net
  *tbl << "Number - Net" << idb_design->get_net_list()->get_num() << TABLE_ENDLINE;

  /// special net
  *tbl << "Number - Special Net" << idb_design->get_special_net_list()->get_num() << TABLE_ENDLINE;

  return tbl;
}

std::shared_ptr<ieda::ReportTable> ReportDB::createSummaryInstances()
{
  std::vector<std::string> header_list = {"Type", "Number", "Number Ratio", "Area", "Area Ratio"};
  auto tbl = std::make_shared<ieda::ReportTable>("Summary - Instance", header_list, static_cast<int>(ReportDBType::kSummaryInstance));

  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = dmInst->get_idb_layout();
  auto inst_list = idb_design->get_instance_list();

  int dbu = idb_design->get_units()->get_micron_dbu() < 0 ? idb_layout->get_units()->get_micron_dbu()
                                                          : idb_design->get_units()->get_micron_dbu();

  int num_max = idb_design->get_instance_list()->get_num();
  int num_netlist = idb_design->get_instance_list()->get_num(IdbInstanceType::kNetlist);
  int num_dist = idb_design->get_instance_list()->get_num(IdbInstanceType::kDist);
  int num_timing = idb_design->get_instance_list()->get_num(IdbInstanceType::kTiming);

  double area_max = dmInst->instanceArea(IdbInstanceType::kMax);
  *tbl << "All Instances" << num_max << (double) (1) << area_max << (double) (1) << TABLE_ENDLINE;

  *tbl << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_ENDLINE;

  double area_netlist = dmInst->netlistInstArea();
  double area_dist = dmInst->distInstArea();
  double area_timing = dmInst->timingInstArea();
  *tbl << "Netlist" << num_netlist << ((double) num_netlist) / num_max << area_netlist << ((double) area_netlist) / area_max
       << TABLE_ENDLINE;
  *tbl << "Physical" << num_dist << ((double) num_dist) / num_max << area_dist << ((double) area_dist) / area_max << TABLE_ENDLINE;
  *tbl << "Timing" << num_timing << ((double) num_timing) / num_max << area_timing << ((double) area_timing) / area_max << TABLE_ENDLINE;

  *tbl << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_ENDLINE;

  int num_core = inst_list->get_num_core();
  int num_core_logic = inst_list->get_num_core_logic();
  int num_pad = inst_list->get_num_pad();
  int num_block = inst_list->get_num_block();
  int num_endcap = inst_list->get_num_endcap();
  int num_cover = inst_list->get_num_cover();
  int num_ring = inst_list->get_num_ring();

  double area_core = ((double) inst_list->get_area_core()) / dbu / dbu;
  double area_core_logic = ((double) inst_list->get_area_core_logic()) / dbu / dbu;
  double area_pad = ((double) inst_list->get_area_pad()) / dbu / dbu;
  double area_block = ((double) inst_list->get_area_block()) / dbu / dbu;
  double area_endcap = ((double) inst_list->get_area_endcap()) / dbu / dbu;
  double area_cover = ((double) inst_list->get_area_cover()) / dbu / dbu;
  double area_ring = ((double) inst_list->get_area_ring()) / dbu / dbu;
  *tbl << "Core" << num_core << ((double) num_core) / num_max << area_core << ((double) area_core) / area_max << TABLE_ENDLINE;
  *tbl << "Core - logic" << num_core_logic << ((double) num_core_logic) / num_max << area_core_logic
       << ((double) area_core_logic) / area_max << TABLE_ENDLINE;
  *tbl << "Pad" << num_pad << ((double) num_pad) / num_max << area_pad << ((double) area_pad) / area_max << TABLE_ENDLINE;
  *tbl << "Block" << num_block << ((double) num_block) / num_max << area_block << ((double) area_block) / area_max << TABLE_ENDLINE;
  *tbl << "Endcap" << num_endcap << ((double) num_endcap) / num_max << area_endcap << ((double) area_endcap) / area_max << TABLE_ENDLINE;
  *tbl << "Cover" << num_cover << ((double) num_cover) / num_max << area_cover << ((double) area_cover) / area_max << TABLE_ENDLINE;
  *tbl << "Ring" << num_ring << ((double) num_ring) / num_max << area_ring << ((double) area_ring) / area_max << TABLE_ENDLINE;

  return tbl;
}

std::shared_ptr<ieda::ReportTable> ReportDB::createSummaryNets()
{
  std::vector<std::string> header_list = {"Net Type", "Number", "Number Ratio", "Length", "Length Ratio"};
  auto tbl = std::make_shared<ieda::ReportTable>("Summary - Net", header_list, static_cast<int>(ReportDBType::kSummaryNet));

  auto idb_design = dmInst->get_idb_design();

  auto idb_net_list = idb_design->get_net_list();

  int32_t num_all = idb_net_list->get_num();
  int32_t num_signal = idb_net_list->get_num_signal();
  int32_t num_clock = idb_net_list->get_num_clock();
  int32_t num_pdn = idb_net_list->get_num_pdn();

  uint64_t len_all = dmInst->allNetLength();
  uint64_t len_signal = dmInst->getSignalNetListLength();
  uint64_t len_clock = dmInst->getClockNetListLength();
  uint64_t len_pdn = dmInst->getPdnNetListLength();
  /// instance
  *tbl << "All Nets" << num_all << (double) 1 << len_all << (len_all == 0 ? 0 : 1) << TABLE_ENDLINE;
  *tbl << "Signal" << num_signal << ((double) num_signal) / num_all << len_signal << (len_all == 0 ? 0 : ((double) len_signal) / len_all)
       << TABLE_ENDLINE;
  *tbl << "Clock" << num_clock << ((double) num_clock) / num_all << len_clock << (len_all == 0 ? 0 : ((double) len_clock) / len_all)
       << TABLE_ENDLINE;
  *tbl << "Power & Ground" << num_pdn << ((double) num_pdn) / num_all << len_pdn << (len_all == 0 ? 0 : ((double) len_pdn) / len_all)
       << TABLE_ENDLINE;

  return tbl;
}

std::shared_ptr<ieda::ReportTable> ReportDB::createSummaryLayers()
{
  std::vector<std::string> header_list = {"Layer",
                                          "Net - Wire Length",
                                          "Net - Wire Length Ratio",
                                          "Net - Wire Number",
                                          "Net - Via Number",
                                          "Net - Patch Number",
                                          "Special Net - Wire Length",
                                          "Special Net - Wire Number",
                                          "Special Net - Via Number"};
  auto tbl = std::make_shared<ieda::ReportTable>("Summary - Layer", header_list, static_cast<int>(ReportDBType::kSummaryLayer));

  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();

  std::vector<SummaryLayerValue> layer_net_value_list;
  std::vector<SummaryLayerValue> layer_specialnet_value_list;
  for (auto layer : idb_layout->get_layers()->get_layers()) {
    SummaryLayerValue layer_value;
    layer_value.layer_name = layer->get_name();
    layer_value.layer_order = layer->get_order();
    layer_value.wire_len = 0;
    layer_value.seg_num = 0;
    layer_value.wire_num = 0;
    layer_value.via_num = 0;
    layer_value.patch_num = 0;

    layer_net_value_list.push_back(layer_value);
    layer_specialnet_value_list.push_back(layer_value);
  }

  for (auto net : idb_design->get_net_list()->get_net_list()) {
    for (auto wire : net->get_wire_list()->get_wire_list()) {
      for (auto segment : wire->get_segment_list()) {
        size_t order = segment->get_layer()->get_order();
        if (order >= layer_net_value_list.size()) {
          continue;
        }

        if (segment->is_wire()) {
          layer_net_value_list[order].seg_num += 1;
          layer_net_value_list[order].wire_num += 1;
          layer_net_value_list[order].wire_len += segment->length();
        }

        if (segment->is_rect()) {
          layer_net_value_list[order].seg_num += 1;
          layer_net_value_list[order].patch_num += 1;
          layer_net_value_list[order].wire_len += segment->length();
        }

        if (segment->is_via()) {
          auto via_list = segment->get_via_list();
          for (auto via : via_list) {
            auto layer_shape = via->get_cut_layer_shape();

            order = layer_shape.get_layer()->get_order();
            layer_net_value_list[order].seg_num += 1;
            layer_net_value_list[order].via_num += 1;
          }
        }
      }
    }
  }

  for (auto special_net : idb_design->get_special_net_list()->get_net_list()) {
    for (auto special_wire : special_net->get_wire_list()->get_wire_list()) {
      for (auto special_segment : special_wire->get_segment_list()) {
        size_t order = special_segment->get_layer()->get_order();
        if (order < 0 || order >= layer_specialnet_value_list.size()) {
          continue;
        }

        if (special_segment->is_line()) {
          layer_specialnet_value_list[order].seg_num += 1;
          layer_specialnet_value_list[order].wire_num += 1;
          layer_specialnet_value_list[order].wire_len += special_segment->length();
        }

        if (special_segment->is_via()) {
          auto via = special_segment->get_via();
          auto layer_shape = via->get_cut_layer_shape();
          order = layer_shape.get_layer()->get_order();
          layer_specialnet_value_list[order].seg_num += 1;
          layer_specialnet_value_list[order].via_num += 1;
        }
      }
    }
  }

  uint64_t nets_len_all = dmInst->allNetLength();
  int layer_num = idb_layout->get_layers()->get_layers().size();
  for (int i = 0; i < layer_num; i++) {
    auto net_value = layer_net_value_list[i];
    auto special_net_value = layer_specialnet_value_list[i];
    *tbl << net_value.layer_name << net_value.wire_len << ((double) net_value.wire_len) / nets_len_all << net_value.wire_num
         << net_value.via_num << net_value.patch_num << special_net_value.wire_len << special_net_value.wire_num
         << special_net_value.via_num << TABLE_ENDLINE;
  }
  return tbl;
}

std::shared_ptr<ieda::ReportTable> ReportDB::createSummaryPins()
{
  std::vector<std::string> header_list = {"Pin Number", "Net Number", "Net Ratio", "Instance Number", "Instance Ratio"};
  auto tbl = std::make_shared<ieda::ReportTable>("Summary - Pin Distribution", header_list, static_cast<int>(ReportDBType::kSummaryPin));

  auto idb_design = dmInst->get_idb_design();
  // auto idb_layout = idb_design->get_layout();

  int instance_total = idb_design->get_instance_list()->get_instance_list().size();
  int net_total = idb_design->get_net_list()->get_net_list().size();

  if (instance_total <= 0 || net_total <= 0) {
    return tbl;
  }

  std::vector<int> net_array(max_num, 0);
  for (auto net : idb_design->get_net_list()->get_net_list()) {
    auto pin_num = net->get_pin_number();
    if (pin_num >= 0 && pin_num <= max_fanout) {
      net_array[pin_num] += 1;
    } else {
      net_array[max_num - 1] += 1;
    }
  }

  std::vector<int> inst_array(max_num, 0);
  for (auto instance : idb_design->get_instance_list()->get_instance_list()) {
    auto pin_num = instance->get_logic_pin_num();
    if (pin_num >= 0 && pin_num <= max_fanout) {
      inst_array[pin_num] += 1;
    } else {
      inst_array[max_num - 1] += 1;
    }
  }

  for (int i = 0; i <= max_fanout; i++) {
    *tbl << i << net_array[i] << ieda::Str::printf("%f", ((float) net_array[i]) / net_total) << inst_array[i]
         << ieda::Str::printf("%f", ((float) inst_array[i]) / instance_total) << TABLE_ENDLINE;
  }

  *tbl << ieda::Str::printf(">= %d ", max_fanout) << net_array[max_num - 1]
       << ieda::Str::printf("%f", ((float) net_array[max_num - 1]) / net_total) << inst_array[max_num - 1]
       << ieda::Str::printf("%f", ((float) inst_array[max_num - 1]) / instance_total) << TABLE_ENDLINE;

  return tbl;
}

}  // namespace iplf
