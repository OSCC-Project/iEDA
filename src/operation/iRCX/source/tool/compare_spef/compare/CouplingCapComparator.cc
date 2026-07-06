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
#include "compare/CouplingCapComparator.hh"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils/CompareMath.hh"
#include "utils/CompareParallel.hh"

namespace ircx {
namespace compare_spef {
namespace {

struct NetMeta
{
  const Net* net = nullptr;
  std::size_t order = 0;
  bool external = false;
  bool selected = false;
};

class NetMetaIndex
{
 public:
  NetMetaIndex(const Data& data, const NetSelector& selector)
  {
    _meta.reserve(data.nets.size());
    for (const Net& net : data.nets) {
      _meta.emplace(net.name,
                    NetMeta{.net = &net,
                            .order = data.index.orderOf(net.name),
                            .external = std::any_of(net.pins.begin(), net.pins.end(), [](const Pin& pin) { return pin.is_external; }),
                            .selected = selector.selected(net)});
    }
  }

  auto find(const std::string& net_name) const -> const NetMeta*
  {
    const auto meta_it = _meta.find(net_name);
    return meta_it == _meta.end() ? nullptr : &meta_it->second;
  }

  auto isExternal(const std::string& net_name) const -> bool
  {
    const NetMeta* net_meta = find(net_name);
    return net_meta != nullptr && net_meta->external;
  }

  auto orderOf(const std::string& net_name) const -> std::size_t
  {
    const NetMeta* net_meta = find(net_name);
    return net_meta == nullptr ? 0 : net_meta->order;
  }

 private:
  std::unordered_map<std::string, NetMeta> _meta;
};

auto makeReportCouplingPair(const NetMetaIndex& meta, const NodePair& key) -> NodePair
{
  const bool first_external = meta.isExternal(key.first);
  const bool second_external = meta.isExternal(key.second);
  if (first_external != second_external) {
    return first_external ? key : NodePair{key.second, key.first};
  }

  const NetMeta* first_meta = meta.find(key.first);
  const NetMeta* second_meta = meta.find(key.second);
  if (first_meta == nullptr || second_meta == nullptr) {
    return key;
  }

  if (first_external && second_external && first_meta->order < second_meta->order) {
    return NodePair{key.second, key.first};
  }
  return key;
}

auto makeCcapMismatch(const NetMetaIndex& meta, const NodePair& key, double capacitance) -> CcapMismatch
{
  CcapMismatch mismatch;
  mismatch.nets = key;
  mismatch.report_nets = makeReportCouplingPair(meta, key);
  mismatch.first_order = meta.orderOf(mismatch.report_nets.first);
  mismatch.second_order = meta.orderOf(mismatch.report_nets.second);
  mismatch.first_external = meta.isExternal(mismatch.report_nets.first);
  mismatch.second_external = meta.isExternal(mismatch.report_nets.second);
  mismatch.capacitance = capacitance;
  return mismatch;
}

void appendRows(Result& result, Result&& thread_result)
{
  result.ccap_rows.insert(result.ccap_rows.end(), std::make_move_iterator(thread_result.ccap_rows.begin()),
                          std::make_move_iterator(thread_result.ccap_rows.end()));
  result.reference_only_couplings.insert(result.reference_only_couplings.end(),
                                         std::make_move_iterator(thread_result.reference_only_couplings.begin()),
                                         std::make_move_iterator(thread_result.reference_only_couplings.end()));
  result.test_only_couplings.insert(result.test_only_couplings.end(), std::make_move_iterator(thread_result.test_only_couplings.begin()),
                                    std::make_move_iterator(thread_result.test_only_couplings.end()));
}

void reserveRows(Result& result, const std::vector<Result>& thread_results)
{
  std::size_t ccap_count = result.ccap_rows.size();
  std::size_t reference_only_count = result.reference_only_couplings.size();
  std::size_t test_only_count = result.test_only_couplings.size();
  for (const auto& thread_result : thread_results) {
    ccap_count += thread_result.ccap_rows.size();
    reference_only_count += thread_result.reference_only_couplings.size();
    test_only_count += thread_result.test_only_couplings.size();
  }
  result.ccap_rows.reserve(ccap_count);
  result.reference_only_couplings.reserve(reference_only_count);
  result.test_only_couplings.reserve(test_only_count);
}

void addRow(const Config& config, const NetMetaIndex& reference_meta, const std::string& victim, const std::string& aggressor,
            double reference_cap, double test_cap, Result& result)
{
  const NetMeta* victim_meta = reference_meta.find(victim);
  if (victim_meta == nullptr || victim_meta->net == nullptr || !victim_meta->selected) {
    return;
  }

  const double reference_victim_tcap = victim_meta->net->total_cap;
  const double reference_rel = reference_victim_tcap <= math::kEpsilon ? 0.0 : std::abs(reference_cap) / reference_victim_tcap;
  if (std::abs(reference_cap) < config.ccap_abs_threshold || reference_rel < config.ccap_rel_threshold) {
    return;
  }

  CcapRow row;
  row.victim = victim;
  row.aggressor = aggressor;
  row.reference = reference_cap;
  row.test = test_cap;
  row.delta = row.test - row.reference;
  row.reference_victim_total_cap = reference_victim_tcap;
  row.relative_delta = math::couplingRelativeDelta(row.test, row.reference, row.reference);
  result.ccap_rows.push_back(std::move(row));
}

}  // namespace

CouplingCapComparator::CouplingCapComparator(const Config& config) : _config(config), _net_selector(config)
{
}

void CouplingCapComparator::compare(const Data& test, const Data& reference, Result& result) const
{
  const NetMetaIndex reference_meta(reference, _net_selector);
  const NetMetaIndex test_meta(test, _net_selector);

  if (!reference.coupling_caps.empty()) {
    const int reference_thread_count = parallel::threadCount(_config, reference.coupling_caps.size());
    std::vector<Result> reference_thread_results(reference_thread_count);
#pragma omp parallel for schedule(dynamic, 256) num_threads(reference_thread_count)
    for (std::size_t entry_index = 0; entry_index < reference.coupling_caps.entries.size(); ++entry_index) {
      const auto& entry = reference.coupling_caps.entries[entry_index];
      const auto* test_cap = test.coupling_caps.find(entry.nets);
      if (test_cap == nullptr) {
        reference_thread_results[omp_get_thread_num()].reference_only_couplings.push_back(
            makeCcapMismatch(reference_meta, entry.nets, entry.capacitance));
        continue;
      }

      addRow(_config, reference_meta, entry.nets.first, entry.nets.second, entry.capacitance, test_cap->capacitance,
             reference_thread_results[omp_get_thread_num()]);
      addRow(_config, reference_meta, entry.nets.second, entry.nets.first, entry.capacitance, test_cap->capacitance,
             reference_thread_results[omp_get_thread_num()]);
    }

    reserveRows(result, reference_thread_results);
    for (auto& thread_result : reference_thread_results) {
      appendRows(result, std::move(thread_result));
    }
  }

  if (!test.coupling_caps.empty()) {
    const int test_thread_count = parallel::threadCount(_config, test.coupling_caps.size());
    std::vector<Result> test_thread_results(test_thread_count);
#pragma omp parallel for schedule(dynamic, 256) num_threads(test_thread_count)
    for (std::size_t entry_index = 0; entry_index < test.coupling_caps.entries.size(); ++entry_index) {
      const auto& entry = test.coupling_caps.entries[entry_index];
      if (!reference.coupling_caps.contains(entry.nets)) {
        test_thread_results[omp_get_thread_num()].test_only_couplings.push_back(
            makeCcapMismatch(test_meta, entry.nets, entry.capacitance));
      }
    }

    reserveRows(result, test_thread_results);
    for (auto& thread_result : test_thread_results) {
      appendRows(result, std::move(thread_result));
    }
  }
}

}  // namespace compare_spef
}  // namespace ircx
