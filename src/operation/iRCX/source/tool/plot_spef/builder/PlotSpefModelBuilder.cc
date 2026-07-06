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
#include "builder/PlotSpefModelBuilder.hh"

#include <omp.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "StringUtils.hh"
#include "Types.hh"
#include "log/Log.hh"
#include "resolver/PlotSpefCapResolver.hh"

namespace ircx::plot_spef {
namespace {

enum class ScanSection
{
  kNone,
  kConn,
  kCap,
  kRes
};

struct ScanGeometry
{
  double length = 0.0;
  double width = 0.0;
  int layer = 0;
  int direction = -1;
  int llx = 0;
  int lly = 0;
  int urx = 0;
  int ury = 0;
  bool has_length = false;
  bool has_width = false;
  bool has_layer = false;
  bool has_direction = false;
  bool has_box = false;
};

auto threadCount(const Config& config, std::size_t work_items) -> int
{
  if (work_items == 0) {
    return 1;
  }
  const int requested = config.cores > 0 ? config.cores : 1;
  return std::min<int>({requested, omp_get_max_threads(), static_cast<int>(work_items)});
}

auto hasCoord(const spef::Coord& coord) -> bool
{
  return coord.x >= 0.0 && coord.y >= 0.0;
}

auto expandScanName(const spef::Exchange& exchange, std::string_view name) -> std::string
{
  return spef::removeEscapes(spef::stripQuotes(spef::expandName(exchange, std::string{name})));
}

auto extractGeometry(std::string_view text, int dbu) -> ScanGeometry
{
  ScanGeometry geometry;
  bool has_llx = false;
  bool has_lly = false;
  bool has_urx = false;
  bool has_ury = false;

  while (true) {
    const auto token = string::take_token(text);
    if (token.empty()) {
      break;
    }

    if (auto llx = string::parse_double_after_prefix(token, "$llx=")) {
      geometry.llx = unit::to_dbu(*llx, dbu);
      has_llx = true;
    } else if (auto lly = string::parse_double_after_prefix(token, "$lly=")) {
      geometry.lly = unit::to_dbu(*lly, dbu);
      has_lly = true;
    } else if (auto urx = string::parse_double_after_prefix(token, "$urx=")) {
      geometry.urx = unit::to_dbu(*urx, dbu);
      has_urx = true;
    } else if (auto ury = string::parse_double_after_prefix(token, "$ury=")) {
      geometry.ury = unit::to_dbu(*ury, dbu);
      has_ury = true;
    } else if (auto layer = string::parse_int_after_prefix(token, "$lvl=")) {
      geometry.layer = *layer;
      geometry.has_layer = true;
    } else if (auto length = string::parse_double_after_prefix(token, "$l=")) {
      geometry.length = *length;
      geometry.has_length = true;
    } else if (auto width = string::parse_double_after_prefix(token, "$w=")) {
      geometry.width = *width;
      geometry.has_width = true;
    } else if (auto direction = string::parse_int_after_prefix(token, "$dir=")) {
      geometry.direction = *direction;
      geometry.has_direction = true;
    }
  }

  geometry.has_box = has_llx && has_lly && has_urx && has_ury && geometry.llx != geometry.urx && geometry.lly != geometry.ury;
  return geometry;
}

auto getDesignName(const spef::Exchange& exchange, const std::string& spef_file) -> std::string
{
  for (const auto& header : exchange.header) {
    if (header.key == "*DESIGN" && !header.value.empty()) {
      return header.value;
    }
  }

  const auto stem = std::filesystem::path(spef_file).stem().string();
  return stem.empty() ? std::string{"plot_spef"} : stem;
}

auto getHeaderValue(const spef::Exchange& exchange, const std::string& key) -> std::string
{
  for (const auto& header : exchange.header) {
    if (header.key == key) {
      return header.value;
    }
  }
  return {};
}

auto normalizeNetFilterName(const spef::Exchange& exchange, const std::string& name) -> std::string
{
  return spef::removeEscapes(spef::stripQuotes(spef::expandName(exchange, name)));
}

auto owningNetName(const std::string& node_name) -> std::string
{
  const auto delimiter = node_name.find(':');
  return delimiter == std::string::npos ? node_name : node_name.substr(0, delimiter);
}

auto fallbackBox(Node& node) -> void
{
  constexpr int kHalfSize = 2;
  node.llx = node.x - kHalfSize;
  node.lly = node.y - kHalfSize;
  node.urx = node.x + kHalfSize;
  node.ury = node.y + kHalfSize;
  node.has_box = true;
}

auto buildNode(const spef::ConnEntry& conn, int dbu) -> Node
{
  Node node;
  node.name = conn.pin_port_name;
  node.layer = std::max(conn.layer, 0);
  if (hasCoord(conn.coordinate)) {
    node.x = unit::to_dbu(conn.coordinate.x, dbu);
    node.y = unit::to_dbu(conn.coordinate.y, dbu);
    node.has_point = true;
  }
  if (hasCoord(conn.ll_coordinate) && hasCoord(conn.ur_coordinate)) {
    node.llx = unit::to_dbu(conn.ll_coordinate.x, dbu);
    node.lly = unit::to_dbu(conn.ll_coordinate.y, dbu);
    node.urx = unit::to_dbu(conn.ur_coordinate.x, dbu);
    node.ury = unit::to_dbu(conn.ur_coordinate.y, dbu);
    node.has_box = node.llx != node.urx && node.lly != node.ury;
  }
  if (!node.has_box && node.has_point) {
    fallbackBox(node);
  }
  return node;
}

auto applyGeometry(Node& node, const ScanGeometry& geometry) -> void
{
  if (geometry.has_layer) {
    node.layer = geometry.layer;
  }
  if (geometry.has_box) {
    node.llx = geometry.llx;
    node.lly = geometry.lly;
    node.urx = geometry.urx;
    node.ury = geometry.ury;
    node.has_box = true;
  }
}

auto applyGeometry(Resistor& resistor, const ScanGeometry& geometry) -> void
{
  if (geometry.has_length) {
    resistor.length = geometry.length;
    resistor.has_length = true;
  }
  if (geometry.has_width) {
    resistor.width = geometry.width;
    resistor.has_width = true;
  }
  if (geometry.has_layer) {
    resistor.layer = geometry.layer;
    resistor.has_layer = true;
  }
  if (geometry.has_direction) {
    resistor.direction = geometry.direction;
    resistor.has_direction = true;
  }
  if (geometry.has_box) {
    resistor.llx = geometry.llx;
    resistor.lly = geometry.lly;
    resistor.urx = geometry.urx;
    resistor.ury = geometry.ury;
    resistor.has_box = true;
  }
}

auto applyNodeGeometry(Model& model, Net& net, const std::string& node_name, const ScanGeometry& geometry) -> void
{
  auto local_node_it = net.nodes_by_name.find(node_name);
  if (local_node_it != net.nodes_by_name.end()) {
    applyGeometry(*local_node_it->second, geometry);
    return;
  }

  auto node_it = model.nodes_by_name.find(node_name);
  if (node_it != model.nodes_by_name.end()) {
    applyGeometry(*node_it->second, geometry);
  }
}

auto parseLayerMapLine(std::string_view line, Model& model) -> void
{
  auto payload = string::trim_view(line.substr(2));
  const auto index = string::take_token(payload);
  const auto name = string::take_token(payload);
  auto layer_id = string::parse_prefixed_index<int>(index);
  if (!layer_id.has_value() || name.empty()) {
    return;
  }
  model.layer_names[*layer_id] = spef::removeEscapes(spef::stripQuotes(std::string{name}));
}

auto buildNetMap(Model& model) -> std::unordered_map<std::string, Net*>
{
  std::unordered_map<std::string, Net*> net_map;
  net_map.reserve(model.nets.size());
  for (auto& net : model.nets) {
    net_map[net.name] = &net;
  }
  return net_map;
}

auto shouldBuildNet(const std::optional<std::unordered_set<std::string>>& visible_net_names, const std::string& net_name) -> bool
{
  return !visible_net_names.has_value() || visible_net_names->contains(net_name);
}

auto initModelMetadata(const spef::Exchange& exchange, const Config& config) -> Model
{
  Model model;
  model.design_name = getDesignName(exchange, config.spef_file);
  model.vendor_name = getHeaderValue(exchange, "*VENDOR");
  model.program_name = getHeaderValue(exchange, "*PROGRAM");
  model.cap_unit = spef::getSpefCapUnit(exchange);
  model.res_unit = spef::getSpefResUnit(exchange);
  model.dbu = config.dbu;
  return model;
}

// SPEF comments carry geometry and layer-map details that the parser
// intentionally keeps out of the normalized Exchange model.
class TextGeometryAugmenter
{
 public:
  TextGeometryAugmenter(const spef::Exchange& exchange, Model& model, const Config& config)
      : exchange_(exchange), model_(model), config_(config), net_map_(buildNetMap(model))
  {
  }

  auto run() -> void
  {
    std::ifstream file(config_.spef_file);
    if (!file.is_open()) {
      return;
    }

    std::string line;
    while (std::getline(file, line)) {
      scanLine(line);
    }
  }

 private:
  auto scanLine(const std::string& line) -> void
  {
    const auto stripped = string::trim_view(line);
    if (stripped.empty() || scanStandaloneComment(stripped)) {
      return;
    }
    in_layer_map_ = false;

    const auto comment_pos = line.find("//");
    const auto content
        = string::trim_view(comment_pos == std::string::npos ? std::string_view{line} : std::string_view{line}.substr(0, comment_pos));
    const auto comment = comment_pos == std::string::npos ? std::string_view{} : std::string_view{line}.substr(comment_pos + 2);
    if (content.empty() || scanSectionHeader(content)) {
      return;
    }
    if (current_net_ == nullptr) {
      return;
    }

    scanSectionPayload(content, comment);
  }

  auto scanStandaloneComment(std::string_view stripped) -> bool
  {
    if (string::starts_with(stripped, "//")) {
      if (string::starts_with(stripped, "// *LAYER_MAP")) {
        in_layer_map_ = true;
        return true;
      }
      if (in_layer_map_) {
        parseLayerMapLine(stripped, model_);
      }
      return true;
    }
    return false;
  }

  auto scanSectionHeader(std::string_view content) -> bool
  {
    if (string::starts_with(content, "*D_NET")) {
      beginNet(content);
      return true;
    }
    if (string::starts_with(content, "*END")) {
      current_net_ = nullptr;
      section_ = ScanSection::kNone;
      return true;
    }
    if (string::starts_with(content, "*CONN")) {
      section_ = ScanSection::kConn;
      return true;
    }
    if (string::starts_with(content, "*CAP")) {
      section_ = ScanSection::kCap;
      return true;
    }
    if (string::starts_with(content, "*RES")) {
      section_ = ScanSection::kRes;
      res_index_ = 0;
      return true;
    }
    return false;
  }

  auto beginNet(std::string_view content) -> void
  {
    auto content_tail = content;
    static_cast<void>(string::take_token(content_tail));
    auto net_name_token = string::take_token(content_tail);
    if (net_name_token.empty()) {
      current_net_ = nullptr;
      section_ = ScanSection::kNone;
      return;
    }

    std::string net_name = expandScanName(exchange_, net_name_token);
    const auto net_it = net_map_.find(net_name);
    current_net_ = net_it == net_map_.end() ? nullptr : net_it->second;
    section_ = ScanSection::kNone;
    res_index_ = 0;
  }

  auto scanSectionPayload(std::string_view content, std::string_view comment) -> void
  {
    if (section_ == ScanSection::kConn
        && (string::starts_with(content, "*I") || string::starts_with(content, "*P") || string::starts_with(content, "*N"))) {
      scanConnectionGeometry(content, comment);
      return;
    }

    if (section_ == ScanSection::kRes && !content.empty() && std::isdigit(static_cast<unsigned char>(content.front()))) {
      scanResistanceGeometry(comment);
    }
  }

  auto scanConnectionGeometry(std::string_view content, std::string_view comment) -> void
  {
    auto content_tail = content;
    static_cast<void>(string::take_token(content_tail));
    const auto node_name_token = string::take_token(content_tail);
    if (node_name_token.empty()) {
      return;
    }

    const std::string node_name = expandScanName(exchange_, node_name_token);
    applyNodeGeometry(model_, *current_net_, node_name, extractGeometry(comment, config_.dbu));
  }

  auto scanResistanceGeometry(std::string_view comment) -> void
  {
    if (res_index_ < current_net_->resistors.size()) {
      applyGeometry(current_net_->resistors[res_index_], extractGeometry(comment, config_.dbu));
    }
    ++res_index_;
  }

  const spef::Exchange& exchange_;
  Model& model_;
  const Config& config_;
  std::unordered_map<std::string, Net*> net_map_;
  Net* current_net_ = nullptr;
  ScanSection section_ = ScanSection::kNone;
  bool in_layer_map_ = false;
  std::size_t res_index_ = 0;
};

// A focused net plot still needs directly coupled neighbors so Cc shapes can be
// resolved and drawn with useful context.
class PlotScope
{
 public:
  PlotScope(const spef::Exchange& exchange, const Config& config) : exchange_(exchange), config_(config) {}

  auto netNamesToBuild() const -> std::optional<std::unordered_set<std::string>>
  {
    if (!config_.hasNetFilter()) {
      return std::nullopt;
    }

    const std::string target_net = targetNetName();
    std::unordered_set<std::string> visible_nets;
    visible_nets.insert(target_net);

    for (const auto& net : exchange_.nets) {
      for (const auto& cap : net.caps) {
        if (cap.node2.empty()) {
          continue;
        }
        const std::string net1 = owningNetName(cap.node1);
        const std::string net2 = owningNetName(cap.node2);
        if (net1 == target_net) {
          visible_nets.insert(net2);
        }
        if (net2 == target_net) {
          visible_nets.insert(net1);
        }
      }
    }
    return visible_nets;
  }

  auto apply(Model& model) const -> void
  {
    if (!config_.hasNetFilter()) {
      showAll(model);
      return;
    }

    const std::string target_net = targetNetName();
    auto net_map = buildNetMap(model);
    const bool found_target = showOnlyTarget(model, target_net);
    showCoupledContext(model, net_map, target_net);
    if (!found_target) {
      LOG_ERROR << "plot_spef warning: target net not found: " << config_.net_name;
    }
  }

 private:
  auto targetNetName() const -> std::string { return normalizeNetFilterName(exchange_, config_.net_name); }

  static auto showAll(Model& model) -> void
  {
    for (auto& net : model.nets) {
      net.visible = true;
      net.context_only = false;
      for (auto& node : net.nodes) {
        node.visible = true;
      }
      for (auto& resistor : net.resistors) {
        resistor.visible = true;
      }
    }
  }

  static auto showOnlyTarget(Model& model, const std::string& target_net) -> bool
  {
    bool found_target = false;
    for (auto& net : model.nets) {
      const bool is_target_net = net.name == target_net;
      net.visible = is_target_net;
      net.context_only = !is_target_net;
      found_target = found_target || is_target_net;
      for (auto& node : net.nodes) {
        node.visible = is_target_net;
      }
      for (auto& resistor : net.resistors) {
        resistor.visible = is_target_net;
      }
    }
    return found_target;
  }

  auto showCoupledContext(Model& model, std::unordered_map<std::string, Net*>& net_map, const std::string& target_net) const -> void
  {
    for (const auto& net : model.nets) {
      for (const auto& cap : net.coupling_caps) {
        const std::string net1 = owningNetName(cap.node1);
        const std::string net2 = owningNetName(cap.node2);
        if (net1 == target_net || net2 == target_net) {
          showNode(model, net_map, cap.node1);
          showNode(model, net_map, cap.node2);
          showEdge(model, net_map, cap.edge1);
          showEdge(model, net_map, cap.edge2);
        }
      }
    }
  }

  static auto showNode(Model& model, std::unordered_map<std::string, Net*>& net_map, const std::string& node_name) -> void
  {
    const auto node_it = model.nodes_by_name.find(node_name);
    if (node_it != model.nodes_by_name.end()) {
      node_it->second->visible = true;
    }
    const auto net_it = net_map.find(owningNetName(node_name));
    if (net_it != net_map.end()) {
      net_it->second->visible = true;
    }
  }

  static auto showEdge(Model& model, std::unordered_map<std::string, Net*>& net_map, const EdgeRef& edge_ref) -> void
  {
    if (!edge_ref.valid || edge_ref.net_index >= model.nets.size()) {
      return;
    }
    auto& net = model.nets[edge_ref.net_index];
    net.visible = true;
    if (edge_ref.resistor_index >= net.resistors.size()) {
      return;
    }
    auto& resistor = net.resistors[edge_ref.resistor_index];
    resistor.visible = true;
    showNode(model, net_map, resistor.node1);
    showNode(model, net_map, resistor.node2);
  }

  const spef::Exchange& exchange_;
  const Config& config_;
};

class ModelAssembler
{
 public:
  ModelAssembler(const spef::Exchange& exchange, const Config& config,
                 const std::optional<std::unordered_set<std::string>>& visible_net_names)
      : exchange_(exchange), config_(config), visible_net_names_(visible_net_names)
  {
  }

  auto build() const -> Model
  {
    Model model = initModelMetadata(exchange_, config_);
    const auto jobs = collectJobs();
    model.nets.reserve(jobs.size());
    model.nodes_by_name.reserve(countNodes(jobs));

    // Each thread builds one independent Net. The final Model vector and node
    // pointer indexes are populated serially to keep pointer ownership stable.
    std::vector<Net> built_nets(jobs.size());
    const int threads = threadCount(config_, jobs.size());
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads)
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(jobs.size()); ++index) {
      built_nets[index] = buildNet(*jobs[index]);
    }

    for (auto& net : built_nets) {
      model.nets.push_back(std::move(net));
      indexNetNodes(model, model.nets.back());
    }
    return model;
  }

 private:
  using NetJobs = std::vector<const spef::Net*>;

  auto shouldBuild(const std::string& net_name) const -> bool { return shouldBuildNet(visible_net_names_, net_name); }

  auto collectJobs() const -> NetJobs
  {
    NetJobs jobs;
    jobs.reserve(visible_net_names_.has_value() ? std::min(exchange_.nets.size(), visible_net_names_->size()) : exchange_.nets.size());
    for (const auto& spef_net : exchange_.nets) {
      if (shouldBuild(spef_net.name)) {
        jobs.push_back(&spef_net);
      }
    }
    return jobs;
  }

  static auto countNodes(const NetJobs& jobs) -> std::size_t
  {
    std::size_t node_count = 0;
    for (const auto* spef_net : jobs) {
      node_count += spef_net->conns.size();
    }
    return node_count;
  }

  auto buildNet(const spef::Net& spef_net) const -> Net
  {
    Net net;
    net.name = spef_net.name;
    net.nodes.reserve(spef_net.conns.size());
    net.nodes_by_name.reserve(spef_net.conns.size());

    const bool need_resistors = config_.plotResistance() || config_.plotCouplingCap() || config_.plotGroundCap();
    const bool need_coupling_caps = config_.plotCouplingCap() || config_.plotGroundCap() || config_.hasNetFilter();
    if (need_resistors) {
      net.resistors.reserve(spef_net.ress.size());
    }
    reserveCapacitors(net, spef_net, need_coupling_caps);

    appendNodes(net, spef_net);
    if (need_resistors) {
      appendResistors(net, spef_net);
    }
    appendCapacitors(net, spef_net, need_coupling_caps);
    return net;
  }

  auto reserveCapacitors(Net& net, const spef::Net& spef_net, bool need_coupling_caps) const -> void
  {
    if (!need_coupling_caps && !config_.plotGroundCap()) {
      return;
    }

    std::size_t coupling_cap_count = 0;
    std::size_t ground_cap_count = 0;
    for (const auto& cap : spef_net.caps) {
      if (cap.node2.empty()) {
        ground_cap_count++;
      } else {
        coupling_cap_count++;
      }
    }
    if (need_coupling_caps) {
      net.coupling_caps.reserve(coupling_cap_count);
    }
    if (config_.plotGroundCap()) {
      net.ground_caps.reserve(ground_cap_count);
    }
  }

  auto appendCapacitors(Net& net, const spef::Net& spef_net, bool need_coupling_caps) const -> void
  {
    if (!need_coupling_caps && !config_.plotGroundCap()) {
      return;
    }

    for (const auto& cap : spef_net.caps) {
      Capacitor capacitor{.node1 = cap.node1, .node2 = cap.node2, .value = cap.res_or_cap};
      if (cap.node2.empty()) {
        if (config_.plotGroundCap()) {
          net.ground_caps.push_back(std::move(capacitor));
        }
      } else if (need_coupling_caps) {
        net.coupling_caps.push_back(std::move(capacitor));
      }
    }
  }

  auto appendNodes(Net& net, const spef::Net& spef_net) const -> void
  {
    for (const auto& conn : spef_net.conns) {
      net.nodes.push_back(buildNode(conn, config_.dbu));
    }
  }

  static auto indexNetNodes(Model& model, Net& net) -> void
  {
    for (auto& node : net.nodes) {
      model.nodes_by_name[node.name] = &node;
      net.nodes_by_name[node.name] = &node;
    }
  }


  static auto appendResistors(Net& net, const spef::Net& spef_net) -> void
  {
    for (std::size_t res_index = 0; res_index < spef_net.ress.size(); ++res_index) {
      const auto& res = spef_net.ress[res_index];
      net.resistors.push_back(Resistor{.node1 = res.node1, .node2 = res.node2, .value = res.res_or_cap, .index = res_index});
    }
  }

  const spef::Exchange& exchange_;
  const Config& config_;
  const std::optional<std::unordered_set<std::string>>& visible_net_names_;
};

}  // namespace

auto ModelBuilder::build(const spef::Exchange& exchange, const Config& config) const -> Model
{
  const PlotScope scope(exchange, config);
  const auto visible_net_names = scope.netNamesToBuild();
  Model model = ModelAssembler(exchange, config, visible_net_names).build();
  TextGeometryAugmenter(exchange, model, config).run();
  resolveCapacitorEdges(model, config);
  scope.apply(model);
  return model;
}

}  // namespace ircx::plot_spef
