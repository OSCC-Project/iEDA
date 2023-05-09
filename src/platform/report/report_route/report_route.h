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
#include "idm.h"
#include "report_basic.h"

namespace iplf {

class ReportRoute : public ReportBase
{
 public:
  explicit ReportRoute(const std::string& report_name) : ReportBase(report_name) {}

  void createNetReport(IdbNet* net);
  void createSummaryReport();

 private:
  std::shared_ptr<ieda::ReportTable> getDesignStatsTable(int64_t pins_number);
  std::shared_ptr<ieda::ReportTable> getPinStatsTable(vector<int64_t>& pin_net_count, int64_t nets);
  std::shared_ptr<ieda::ReportTable> getWireLengthStatsTable(const std::vector<int64_t>& routing_layer_length);
  std::shared_ptr<ieda::ReportTable> getViaCutStatsTable(const std::vector<int64_t>& via_cut_nums);
  std::shared_ptr<ieda::ReportTable> getLengthRangeTable(std::vector<int64_t>& lengths, int64_t d);
};

struct NetStatistics
{
  string Name;
  int64_t HPWL = 0;
  int64_t TotalLength = 0;
  int64_t TotalVias = 0;
  std::vector<int64_t> RouteLayerLength;
  std::vector<int64_t> CutLayerVias;
  std::map<std::string, int64_t> Vias;
  int64_t FanIn = 0;
  int64_t FanOut = 0;
  int64_t PinNum = 0;
  NetStatistics()
      : RouteLayerLength(dmInst->get_idb_layout()->get_layers()->get_routing_layers_number(), 0),
        CutLayerVias(dmInst->get_idb_layout()->get_layers()->get_cut_layers_number(), 0)
  {
  }
  NetStatistics& operator+=(const NetStatistics& ref);
  static NetStatistics extractNetInfo(IdbNet* net);
};

}  // namespace iplf