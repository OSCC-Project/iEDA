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
 * @file RealTechPinCapProbe.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Representative real-tech pin capacitance probe discovery for tests.
 */

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "Flow.hh"
#include "IdbCellMaster.h"
#include "IdbDesign.h"
#include "IdbEnum.h"
#include "IdbInstance.h"
#include "IdbNet.h"
#include "IdbPins.h"
#include "IdbTerm.h"
#include "common/CTSTestRuntime.hh"
#include "common/realtech/asset/RealTechAssetLoader.hh"
#include "common/realtech/setup/RealTechDesignSetup.hh"
#include "database/design/Inst.hh"
#include "database/design/Pin.hh"
#include "database/io/Wrapper.hh"
#include "database/spatial/Point.hh"
#include "idm.h"

namespace icts_test::common::realtech::asset {
namespace {

auto TryMakeRealPinCapProbe(idb::IdbPin* idb_pin, const std::string& net_name, bool is_clock_net) -> std::optional<RealPinCapProbe>
{
  if (idb_pin == nullptr) {
    return std::nullopt;
  }

  auto* idb_term = idb_pin->get_term();
  if (idb_term == nullptr) {
    return std::nullopt;
  }
  const auto direction = idb_term->get_direction();
  if (direction != idb::IdbConnectDirection::kInput && direction != idb::IdbConnectDirection::kInOut) {
    return std::nullopt;
  }

  auto* idb_inst = idb_pin->get_instance();
  if (idb_inst == nullptr) {
    return std::nullopt;
  }
  auto* cell_master = idb_inst->get_cell_master();
  if (cell_master == nullptr) {
    return std::nullopt;
  }

  icts::Inst probe_inst(idb_inst->get_name(), cell_master->get_name(), icts::InstType::kUnknown, icts::Point<int>(-1, -1));
  icts::Pin probe_pin(idb_pin->get_pin_name(), icts::PinType::kIn, icts::Point<int>(-1, -1), &probe_inst);
  const double pin_cap_pf = std::max(0.0, icts_test::runtime::CurrentRuntime().wrapper.queryPinCapacitance(&probe_pin));
  if (pin_cap_pf <= 0.0) {
    return std::nullopt;
  }

  return RealPinCapProbe{
      .net_name = net_name,
      .inst_name = idb_inst->get_name(),
      .cell_master = cell_master->get_name(),
      .pin_name = idb_pin->get_pin_name(),
      .is_clock_net = is_clock_net,
      .pre_timing_cap_pf = pin_cap_pf,
  };
}

}  // namespace

auto TryFindRepresentativeRealPinCapProbe() -> std::optional<RealPinCapProbe>
{
  auto* idb_design = dmInst->get_idb_design();
  auto* net_list = idb_design != nullptr ? idb_design->get_net_list() : nullptr;
  if (net_list == nullptr) {
    return std::nullopt;
  }

  const auto try_scan = [&net_list](bool clock_only) -> std::optional<RealPinCapProbe> {
    for (auto* idb_net : net_list->get_net_list()) {
      if (idb_net == nullptr) {
        continue;
      }
      const bool is_clock_net = idb_net->is_clock() != 0U;
      if (clock_only && !is_clock_net) {
        continue;
      }
      for (auto* idb_pin : idb_net->get_load_pins()) {
        if (auto probe = TryMakeRealPinCapProbe(idb_pin, idb_net->get_net_name(), is_clock_net); probe.has_value()) {
          return probe;
        }
      }
    }
    return std::nullopt;
  };

  if (auto probe = try_scan(true); probe.has_value()) {
    return probe;
  }
  return try_scan(false);
}

}  // namespace icts_test::common::realtech::asset
