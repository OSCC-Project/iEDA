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
#include "compare/ResultSorter.hh"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ircx {
namespace compare_spef {
namespace {

auto netOrder(const Data& data, const std::string& net_name) -> std::size_t
{
  return data.index.orderOf(net_name);
}

void sortRows(Result& result)
{
  auto relative_value = [](const auto& row) { return row.relative_delta.value_or(std::numeric_limits<double>::infinity()); };
  auto rounded_percent_value = [](const auto& row) {
    if (!row.relative_delta.has_value()) {
      return std::numeric_limits<double>::infinity();
    }
    return std::round(*row.relative_delta * 100000.0) / 1000.0;
  };
  std::sort(result.tcap_rows.begin(), result.tcap_rows.end(), [&](const ValueRow& lhs, const ValueRow& rhs) {
    const double lhs_rel = relative_value(lhs);
    const double rhs_rel = relative_value(rhs);
    if (lhs_rel != rhs_rel) {
      return lhs_rel < rhs_rel;
    }
    return lhs.net < rhs.net;
  });
  std::sort(result.gcap_rows.begin(), result.gcap_rows.end(), [&](const GcapRow& lhs, const GcapRow& rhs) {
    const double lhs_rel = relative_value(lhs);
    const double rhs_rel = relative_value(rhs);
    if (lhs_rel != rhs_rel) {
      return lhs_rel < rhs_rel;
    }
    return lhs.net < rhs.net;
  });
  std::sort(result.ccap_rows.begin(), result.ccap_rows.end(), [](const CcapRow& lhs, const CcapRow& rhs) {
    const double lhs_rel = lhs.relative_delta.value_or(std::numeric_limits<double>::infinity());
    const double rhs_rel = rhs.relative_delta.value_or(std::numeric_limits<double>::infinity());
    if (lhs_rel != rhs_rel) {
      return lhs_rel < rhs_rel;
    }
    if (lhs.victim != rhs.victim) {
      return lhs.victim < rhs.victim;
    }
    return lhs.aggressor < rhs.aggressor;
  });
  std::sort(result.p2p_rows.begin(), result.p2p_rows.end(), [&](const ResistanceRow& lhs, const ResistanceRow& rhs) {
    const double lhs_rel = rounded_percent_value(lhs);
    const double rhs_rel = rounded_percent_value(rhs);
    if (lhs_rel != rhs_rel) {
      return lhs_rel < rhs_rel;
    }
    const double lhs_report_test = std::round(lhs.test * 1000.0) / 1000.0;
    const double rhs_report_test = std::round(rhs.test * 1000.0) / 1000.0;
    const double lhs_report_reference = std::round(lhs.reference * 1000.0) / 1000.0;
    const double rhs_report_reference = std::round(rhs.reference * 1000.0) / 1000.0;
    if (lhs_report_test == rhs_report_test && lhs_report_reference == rhs_report_reference) {
      if (lhs.net != rhs.net) {
        return lhs.net < rhs.net;
      }
    }
    const double lhs_raw_rel = relative_value(lhs);
    const double rhs_raw_rel = relative_value(rhs);
    if (lhs_raw_rel != rhs_raw_rel) {
      return lhs_raw_rel < rhs_raw_rel;
    }
    if (lhs.net != rhs.net) {
      return lhs.net < rhs.net;
    }
    if (lhs.from_pin != rhs.from_pin) {
      return lhs.from_pin < rhs.from_pin;
    }
    return lhs.to_pin < rhs.to_pin;
  });
}

void sortMismatchedNets(Result& result, const Data& test, const Data& reference)
{
  auto reverse_reference_order
      = [&](const std::string& lhs, const std::string& rhs) { return netOrder(reference, lhs) > netOrder(reference, rhs); };
  auto reverse_test_order = [&](const std::string& lhs, const std::string& rhs) { return netOrder(test, lhs) > netOrder(test, rhs); };
  std::sort(result.reference_only_nets.begin(), result.reference_only_nets.end(), reverse_reference_order);
  std::sort(result.test_only_nets.begin(), result.test_only_nets.end(), reverse_test_order);
}

auto reverseCouplingOrder(const CcapMismatch& lhs, const CcapMismatch& rhs) -> bool
{
  if (lhs.first_external != rhs.first_external) {
    return !lhs.first_external;
  }
  if (lhs.first_order != rhs.first_order) {
    return lhs.first_external ? lhs.first_order < rhs.first_order : lhs.first_order > rhs.first_order;
  }
  if (lhs.first_external && lhs.second_external != rhs.second_external) {
    return !lhs.second_external;
  }
  if (lhs.second_order != rhs.second_order) {
    return lhs.first_external && lhs.second_external ? lhs.second_order < rhs.second_order : lhs.second_order > rhs.second_order;
  }
  if (lhs.report_nets.first != rhs.report_nets.first) {
    return lhs.report_nets.first > rhs.report_nets.first;
  }
  if (lhs.report_nets.second != rhs.report_nets.second) {
    return lhs.report_nets.second > rhs.report_nets.second;
  }
  return lhs.capacitance < rhs.capacitance;
}

void sortMismatchedCouplings(Result& result)
{
  std::sort(result.reference_only_couplings.begin(), result.reference_only_couplings.end(), reverseCouplingOrder);
  std::sort(result.test_only_couplings.begin(), result.test_only_couplings.end(), reverseCouplingOrder);
}

}  // namespace

void ResultSorter::sort(Result& result, const Data& test, const Data& reference) const
{
  sortMismatchedNets(result, test, reference);
  sortMismatchedCouplings(result);
  sortRows(result);
}

}  // namespace compare_spef
}  // namespace ircx
