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
#include "compare/PathPairGenerator.hh"

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_set>
#include <utility>

namespace ircx {
namespace compare_spef {
namespace {

constexpr std::size_t kMaxPathPairs = 250;
constexpr std::size_t kPathPairPinWindow = 50;
constexpr std::size_t kDenseNetAnchorPins = 5;

// compare_parasitics reaches this code with SPEF data only. Default p2p selection is
// intentionally inferred from *CONN role and direction; it does not rely on
// DEF/LEF/Liberty macro class or pin class metadata.

struct PinCounts
{
  std::size_t sources = 0;
  std::size_t sinks = 0;
};

using PinList = std::vector<const Pin*>;

enum class PairUniqueness
{
  kDirected,
  kUndirected
};

class PathPairCollector
{
 public:
  PathPairCollector(std::size_t reserve_count, PairUniqueness uniqueness) : _uniqueness(uniqueness)
  {
    _seen.reserve(reserve_count);
    _pairs.reserve(reserve_count);
  }

  auto add(NodePair pair) -> bool
  {
    if (pair.first.empty() || pair.second.empty() || pair.first == pair.second) {
      return false;
    }

    NodePair seen_pair = _uniqueness == PairUniqueness::kUndirected ? NodePair::ordered(pair.first, pair.second) : pair;
    if (!_seen.insert(std::move(seen_pair)).second) {
      return false;
    }

    _pairs.push_back(std::move(pair));
    return true;
  }

  auto full() const -> bool { return _pairs.size() >= kMaxPathPairs; }
  auto size() const -> std::size_t { return _pairs.size(); }

  auto sortedPairs() && -> std::vector<NodePair>
  {
    std::sort(_pairs.begin(), _pairs.end());
    return std::move(_pairs);
  }

  auto pairs() && -> std::vector<NodePair> { return std::move(_pairs); }

 private:
  PairUniqueness _uniqueness;
  std::unordered_set<NodePair, NodePairHash> _seen;
  std::vector<NodePair> _pairs;
};

auto isSourcePin(const Pin& pin) -> bool
{
  if (pin.direction == "B") {
    return true;
  }
  if (pin.is_external) {
    return pin.direction == "I";
  }
  return pin.direction == "O";
}

auto isSinkPin(const Pin& pin) -> bool
{
  if (pin.direction == "B") {
    return true;
  }
  if (pin.is_external) {
    return pin.direction == "O";
  }
  return pin.direction == "I";
}

auto orderedReportPair(const Pin& pin1, const Pin& pin2) -> NodePair
{
  if (pin1.connection_order != pin2.connection_order) {
    return pin1.connection_order < pin2.connection_order ? NodePair{pin1.name, pin2.name} : NodePair{pin2.name, pin1.name};
  }
  return pin1.name < pin2.name ? NodePair{pin1.name, pin2.name} : NodePair{pin2.name, pin1.name};
}

auto pinNames(const Net& net) -> std::vector<std::string>
{
  std::vector<std::string> names;
  names.reserve(net.pins.size());
  for (const auto& pin : net.pins) {
    if (pin.direction == "N") {
      continue;
    }
    names.push_back(pin.name);
  }
  std::sort(names.begin(), names.end());
  names.erase(std::unique(names.begin(), names.end()), names.end());
  return names;
}

auto pinList(const Net& net) -> PinList
{
  PinList pins;
  pins.reserve(net.pins.size());
  for (const auto& pin : net.pins) {
    if (pin.direction == "N" || pin.name.empty()) {
      continue;
    }
    pins.push_back(&pin);
  }
  std::stable_sort(pins.begin(), pins.end(), [](const Pin* lhs, const Pin* rhs) { return lhs->name < rhs->name; });
  pins.erase(std::unique(pins.begin(), pins.end(), [](const Pin* lhs, const Pin* rhs) { return lhs->name == rhs->name; }), pins.end());
  return pins;
}

auto countPins(const PinList& pins) -> PinCounts
{
  PinCounts counts;
  for (const Pin* pin : pins) {
    if (isSourcePin(*pin)) {
      counts.sources++;
    }
    if (isSinkPin(*pin)) {
      counts.sinks++;
    }
  }
  return counts;
}

auto shouldCompareWindowPair(std::size_t lhs_index, std::size_t rhs_index, bool dense_net) -> bool
{
  return !dense_net || lhs_index < kDenseNetAnchorPins || rhs_index < kDenseNetAnchorPins;
}

auto appendSourceSinkPairs(const PinList& pins, PathPairCollector& collector) -> bool
{
  const std::size_t pin_count = std::min(kPathPairPinWindow, pins.size());
  const bool dense_net = pins.size() > kDenseNetAnchorPins;

  for (std::size_t source_index = 0; source_index < pin_count; ++source_index) {
    const Pin* source_pin = pins[source_index];
    if (!isSourcePin(*source_pin)) {
      continue;
    }
    for (std::size_t sink_index = 0; sink_index < pin_count; ++sink_index) {
      const Pin* sink_pin = pins[sink_index];
      if (source_index == sink_index || !isSinkPin(*sink_pin) || !shouldCompareWindowPair(source_index, sink_index, dense_net)) {
        continue;
      }

      if (collector.add(orderedReportPair(*source_pin, *sink_pin))) {
        if (collector.full()) {
          return true;
        }
      }
    }
  }
  return false;
}

auto appendExternalInternalPairs(const PinList& pins, PathPairCollector& collector) -> bool
{
  const std::size_t pin_count = std::min(kPathPairPinWindow, pins.size());
  const bool dense_net = pins.size() > kDenseNetAnchorPins;

  for (std::size_t external_index = 0; external_index < pin_count; ++external_index) {
    const Pin* external_pin = pins[external_index];
    if (!external_pin->is_external || external_pin->direction == "N") {
      continue;
    }
    for (std::size_t internal_index = 0; internal_index < pin_count; ++internal_index) {
      const Pin* internal_pin = pins[internal_index];
      if (external_index == internal_index || internal_pin->is_external || internal_pin->direction == "N"
          || external_pin->direction == internal_pin->direction || !shouldCompareWindowPair(external_index, internal_index, dense_net)) {
        continue;
      }
      if (collector.add(orderedReportPair(*external_pin, *internal_pin)) && collector.full()) {
        return true;
      }
    }
  }
  return false;
}

auto configuredPathPairs(const Net& net, const Config& config) -> std::vector<NodePair>
{
  const auto pins = pinNames(net);
  const std::unordered_set<std::string> pin_set(pins.begin(), pins.end());
  PathPairCollector collector(kMaxPathPairs, PairUniqueness::kDirected);

  auto add_from_to = [&](const std::string& from_pin, const std::string& to_pin) {
    if (pin_set.contains(from_pin) && pin_set.contains(to_pin)) {
      collector.add(NodePair{from_pin, to_pin});
    }
  };

  if (!config.from_pin.empty() && !config.to_pin.empty()) {
    add_from_to(config.from_pin, config.to_pin);
  }
  for (const auto& [from_pin, to_pin] : config.from_to_pins) {
    add_from_to(from_pin, to_pin);
  }

  std::vector<std::string> from_pins = config.from_pins;
  std::vector<std::string> to_pins = config.to_pins;
  if (!config.from_pin.empty()) {
    from_pins.push_back(config.from_pin);
  }
  if (!config.to_pin.empty()) {
    to_pins.push_back(config.to_pin);
  }

  if (!from_pins.empty() && !to_pins.empty()) {
    for (const auto& from_pin : from_pins) {
      for (const auto& to_pin : to_pins) {
        add_from_to(from_pin, to_pin);
      }
    }
  } else if (!from_pins.empty()) {
    for (const auto& from_pin : from_pins) {
      if (!pin_set.contains(from_pin)) {
        continue;
      }
      for (const auto& to_pin : pins) {
        collector.add(NodePair{from_pin, to_pin});
      }
    }
  } else if (!to_pins.empty()) {
    for (const auto& to_pin : to_pins) {
      if (!pin_set.contains(to_pin)) {
        continue;
      }
      for (const auto& from_pin : pins) {
        collector.add(NodePair{from_pin, to_pin});
      }
    }
  }

  return std::move(collector).sortedPairs();
}

auto defaultPathPairs(const Net& net) -> std::vector<NodePair>
{
  const auto pins = pinList(net);
  const PinCounts counts = countPins(pins);

  PathPairCollector collector(std::min(kMaxPathPairs, counts.sources * std::max<std::size_t>(counts.sinks, 1)),
                              PairUniqueness::kUndirected);
  if (appendSourceSinkPairs(pins, collector)) {
    return std::move(collector).pairs();
  }
  if (appendExternalInternalPairs(pins, collector)) {
    return std::move(collector).pairs();
  }
  return std::move(collector).pairs();
}

}  // namespace

PathPairGenerator::PathPairGenerator(const Config& config) : _config(config), _net_selector(config)
{
}

auto PathPairGenerator::generate(const Net& net) const -> std::vector<NodePair>
{
  if (_net_selector.hasPathFilter()) {
    return configuredPathPairs(net, _config);
  }
  return defaultPathPairs(net);
}

}  // namespace compare_spef
}  // namespace ircx
