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
#include "gds/PlotSpefGdsWriter.hh"

#include <omp.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "FormatUtils.hh"
#include "GTWriter.hpp"
#include "GdsBoundary.hpp"
#include "GdsData.hpp"
#include "GdsPath.hpp"
#include "GdsSref.hpp"
#include "GdsStruct.hpp"
#include "GdsText.hpp"
#include "PathUtils.hh"
#include "StringUtils.hh"
#include "log/Log.hh"
#include "model/PlotSpefGdsType.hh"

namespace ircx::plot_spef {
namespace {

constexpr int kEdgeWidth = 1;
constexpr int kCcWidth = 1;

struct PlotPoint
{
  int x = 0;
  int y = 0;
  int layer = 0;
  bool valid = false;
};

struct GroundCapPlot
{
  std::string key;
  EdgeRef edge;
  std::string node;
  double value = 0.0;
};

struct CouplingCapPlot
{
  std::string key;
  EdgeRef edge1;
  EdgeRef edge2;
  std::string node1;
  std::string node2;
  double value = 0.0;
};

struct NetGdsJob
{
  const Net* net = nullptr;
  std::string struct_name;
};

auto threadCount(const Config& config, std::size_t work_items) -> int
{
  if (work_items == 0) {
    return 1;
  }
  const int requested = config.cores > 0 ? config.cores : 1;
  return std::min<int>({requested, omp_get_max_threads(), static_cast<int>(work_items)});
}

auto findNode(const Model& model, const std::string& name) -> const Node*
{
  const auto it = model.nodes_by_name.find(name);
  return it == model.nodes_by_name.end() ? nullptr : it->second;
}

auto findEdge(const Model& model, const EdgeRef& ref) -> const Resistor*
{
  if (!ref.valid || ref.net_index >= model.nets.size()) {
    return nullptr;
  }
  const auto& net = model.nets[ref.net_index];
  if (ref.resistor_index >= net.resistors.size()) {
    return nullptr;
  }
  return &net.resistors[ref.resistor_index];
}

auto addBoxBoundary(idb::GdsStruct& gds_net, int layer, int data_type, int llx, int lly, int urx, int ury) -> void
{
  idb::GdsBoundary boundary;
  boundary.layer = layer;
  boundary.data_type = data_type;
  boundary.add_coord(llx, lly);
  boundary.add_coord(urx, lly);
  boundary.add_coord(urx, ury);
  boundary.add_coord(llx, ury);
  boundary.add_coord(llx, lly);
  gds_net.add_element(boundary);
}

auto addRect(idb::GdsStruct& gds_net, const Node& node) -> void
{
  if (node.has_box) {
    addBoxBoundary(gds_net, node.layer, kNode, node.llx, node.lly, node.urx, node.ury);
  }
}

auto addText(idb::GdsStruct& gds_net, int layer, int data_type, int x, int y, const std::string& text,
             idb::GdsPresentation presentation = idb::GdsPresentation::kBottomLeft) -> void
{
  idb::GdsText gds_text;
  gds_text.layer = layer;
  gds_text.text_type = data_type;
  gds_text.add_coord(x, y);
  gds_text.str = text;
  gds_text.presentation = presentation;
  gds_net.add_element(gds_text);
}

auto addPath(idb::GdsStruct& gds_net, int layer, int data_type, int width, const Node& node1, const Node& node2) -> void
{
  if (!node1.has_point || !node2.has_point) {
    return;
  }

  idb::GdsPath path;
  path.layer = layer;
  path.data_type = data_type;
  path.width = width;
  path.add_coord(node1.x, node1.y);
  path.add_coord(node2.x, node2.y);
  gds_net.add_element(path);
}

auto addPath(idb::GdsStruct& gds_net, int layer, int data_type, int width, const PlotPoint& point1, const PlotPoint& point2) -> void
{
  if (!point1.valid || !point2.valid) {
    return;
  }

  idb::GdsPath path;
  path.layer = layer;
  path.data_type = data_type;
  path.width = width;
  path.add_coord(point1.x, point1.y);
  path.add_coord(point2.x, point2.y);
  gds_net.add_element(path);
}

auto addRect(idb::GdsStruct& gds_net, const Resistor& resistor, int layer) -> void
{
  if (resistor.has_box) {
    addBoxBoundary(gds_net, layer, kEdge, resistor.llx, resistor.lly, resistor.urx, resistor.ury);
  }
}

auto edgeCenter(const Model& model, const Resistor& edge) -> PlotPoint
{
  PlotPoint point;
  if (edge.has_layer) {
    point.layer = edge.layer;
  }
  if (edge.has_box) {
    point.x = (edge.llx + edge.urx) / 2;
    point.y = (edge.lly + edge.ury) / 2;
    point.valid = true;
    return point;
  }

  const Node* node1 = findNode(model, edge.node1);
  const Node* node2 = findNode(model, edge.node2);
  if (node1 == nullptr || node2 == nullptr || !node1->has_point || !node2->has_point) {
    return {};
  }

  point.x = (node1->x + node2->x) / 2;
  point.y = (node1->y + node2->y) / 2;
  if (!edge.has_layer) {
    point.layer = node1->layer;
  }
  point.valid = true;
  return point;
}

auto capEdgeCenter(const Model& model, const EdgeRef& ref) -> PlotPoint
{
  const Resistor* edge = findEdge(model, ref);
  return edge == nullptr ? PlotPoint{} : edgeCenter(model, *edge);
}

auto nodePoint(const Model& model, const std::string& node_name) -> PlotPoint
{
  const Node* node = findNode(model, node_name);
  if (node == nullptr || !node->has_point) {
    return {};
  }
  return PlotPoint{.x = node->x, .y = node->y, .layer = node->layer, .valid = true};
}

auto capPoint(const Model& model, const EdgeRef& edge, const std::string& node_name) -> PlotPoint
{
  PlotPoint point = capEdgeCenter(model, edge);
  return point.valid ? point : nodePoint(model, node_name);
}

auto isVisibleEdge(const Model& model, const EdgeRef& ref) -> bool
{
  if (!ref.valid) {
    return true;
  }
  if (ref.net_index >= model.nets.size()) {
    return false;
  }
  const auto& net = model.nets[ref.net_index];
  return net.visible && ref.resistor_index < net.resistors.size() && net.resistors[ref.resistor_index].visible;
}

auto couplingKey(const Capacitor& cap) -> std::string
{
  if (cap.edge1.valid && cap.edge2.valid) {
    std::string key1 = edgeRefKey(cap.edge1);
    std::string key2 = edgeRefKey(cap.edge2);
    if (key2 < key1) {
      std::swap(key1, key2);
    }
    return "E:" + key1 + "\n" + key2;
  }

  std::string node1 = cap.node1;
  std::string node2 = cap.node2;
  if (node2 < node1) {
    std::swap(node1, node2);
  }
  return "N:" + node1 + "\n" + node2;
}

auto groundKey(const Capacitor& cap) -> std::string
{
  return cap.edge1.valid ? "E:" + edgeRefKey(cap.edge1) : "N:" + cap.node1;
}

auto collectGroundCapPlots(const Model& model, const Net& net) -> std::vector<GroundCapPlot>
{
  std::unordered_map<std::string, std::size_t> index_by_key;
  std::vector<GroundCapPlot> plots;
  for (const auto& cap : net.ground_caps) {
    if (!isVisibleEdge(model, cap.edge1)) {
      continue;
    }
    const std::string key = groundKey(cap);
    const auto [it, inserted] = index_by_key.emplace(key, plots.size());
    if (inserted) {
      plots.push_back(GroundCapPlot{.key = key, .edge = cap.edge1, .node = cap.node1});
    }
    plots[it->second].value += cap.value;
  }
  std::sort(plots.begin(), plots.end(), [](const GroundCapPlot& lhs, const GroundCapPlot& rhs) { return lhs.key < rhs.key; });
  return plots;
}

auto collectCouplingCapPlots(const Model& model, const Net& net) -> std::vector<CouplingCapPlot>
{
  std::unordered_map<std::string, std::size_t> index_by_key;
  std::vector<CouplingCapPlot> plots;
  for (const auto& cap : net.coupling_caps) {
    if (!isVisibleEdge(model, cap.edge1) || !isVisibleEdge(model, cap.edge2)) {
      continue;
    }
    const std::string key = couplingKey(cap);
    const auto [it, inserted] = index_by_key.emplace(key, plots.size());
    if (inserted) {
      plots.push_back(CouplingCapPlot{.key = key, .edge1 = cap.edge1, .edge2 = cap.edge2, .node1 = cap.node1, .node2 = cap.node2});
    }
    plots[it->second].value += cap.value;
  }
  std::sort(plots.begin(), plots.end(), [](const CouplingCapPlot& lhs, const CouplingCapPlot& rhs) { return lhs.key < rhs.key; });
  return plots;
}

auto addTopReference(idb::GdsData& gds_data, const std::string& name) -> void
{
  idb::GdsSref sref;
  sref.sname = name;
  sref.add_coord(0, 0);
  gds_data.get_top_struct()->add_element(sref);
}

auto formatResistance(double value) -> std::string
{
  return "R=" + format::significant(value, 3) + "Ω";
}

auto formatCapacitance(double value, const std::string& unit) -> std::string
{
  return format::with_unit(value, format::unit_symbol(unit), 3);
}

auto collectNetGdsJobs(const Model& model) -> std::vector<NetGdsJob>
{
  std::unordered_map<std::string, int> struct_name_count;
  struct_name_count.reserve(model.nets.size());

  std::vector<NetGdsJob> jobs;
  jobs.reserve(model.nets.size());
  for (const auto& net : model.nets) {
    if (!net.visible) {
      continue;
    }
    std::string struct_name = string::identifier(net.name, "net");
    const int count = struct_name_count[struct_name]++;
    if (count > 0) {
      struct_name += "_" + std::to_string(count);
    }
    jobs.push_back({.net = &net, .struct_name = std::move(struct_name)});
  }
  return jobs;
}

class NetGdsWriter
{
 public:
  NetGdsWriter(const Model& model, const Config& config) : model_(model), config_(config) {}

  auto write(idb::GdsStruct& gds_net, const Net& net) const -> void
  {
    writeNodes(gds_net, net);
    writeResistors(gds_net, net);
    writeCouplingCaps(gds_net, net);
    writeGroundCaps(gds_net, net);
  }

 private:
  auto writeNodes(idb::GdsStruct& gds_net, const Net& net) const -> void
  {
    for (const auto& node : net.nodes) {
      if (!node.visible) {
        continue;
      }
      addRect(gds_net, node);
      if (node.has_point) {
        addText(gds_net, node.layer, kTextNode, node.x, node.y, node.name);
      }
    }
  }

  auto writeResistors(idb::GdsStruct& gds_net, const Net& net) const -> void
  {
    if (!config_.plotResistance() && !net.context_only) {
      return;
    }

    for (const auto& resistor : net.resistors) {
      if (!resistor.visible) {
        continue;
      }
      const Node* node1 = findNode(model_, resistor.node1);
      const Node* node2 = findNode(model_, resistor.node2);
      if (node1 == nullptr || node2 == nullptr || !node1->has_point || !node2->has_point) {
        continue;
      }

      const int layer = resistor.has_layer ? resistor.layer : node1->layer;
      if (resistor.has_box) {
        addRect(gds_net, resistor, layer);
      } else {
        addPath(gds_net, layer, kEdge, kEdgeWidth, *node1, *node2);
      }
      if (config_.plotResistance()) {
        addText(gds_net, layer, kTextRes, (node1->x + node2->x) / 2, (node1->y + node2->y) / 2, formatResistance(resistor.value),
                idb::GdsPresentation::kTopRight);
      }
    }
  }

  auto writeCouplingCaps(idb::GdsStruct& gds_net, const Net& net) const -> void
  {
    if (net.context_only || !config_.plotCouplingCap()) {
      return;
    }

    for (const auto& cap : collectCouplingCapPlots(model_, net)) {
      const PlotPoint point1 = capPoint(model_, cap.edge1, cap.node1);
      const PlotPoint point2 = capPoint(model_, cap.edge2, cap.node2);
      if (!point1.valid || !point2.valid) {
        continue;
      }

      const int layer = point1.layer;
      addPath(gds_net, layer, kCc, kCcWidth, point1, point2);
      addText(gds_net, layer, kTextCc, (point1.x + point2.x) / 2, (point1.y + point2.y) / 2,
              formatCapacitance(cap.value, model_.cap_unit), idb::GdsPresentation::kTopRight);
    }
  }

  auto writeGroundCaps(idb::GdsStruct& gds_net, const Net& net) const -> void
  {
    if (net.context_only || !config_.plotGroundCap()) {
      return;
    }

    for (const auto& cap : collectGroundCapPlots(model_, net)) {
      PlotPoint point = capPoint(model_, cap.edge, cap.node);
      if (!point.valid) {
        continue;
      }
      addText(gds_net, point.layer, kTextCg, point.x, point.y, formatCapacitance(cap.value, model_.cap_unit),
              idb::GdsPresentation::kTopRight);
    }
  }

  const Model& model_;
  const Config& config_;
};

auto buildNetGdsStructs(const Model& model, const Config& config, const std::vector<NetGdsJob>& jobs)
    -> std::vector<std::unique_ptr<idb::GdsStruct>>
{
  // Threads build private structs only. GdsData and the file writer are updated
  // later on the main thread because their ownership model is not thread-safe.
  std::vector<std::unique_ptr<idb::GdsStruct>> structs(jobs.size());
  const NetGdsWriter net_writer(model, config);
  const int threads = threadCount(config, jobs.size());
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads)
  for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(jobs.size()); ++index) {
    const auto& job = jobs[index];
    auto gds_net = std::make_unique<idb::GdsStruct>(job.struct_name);
    net_writer.write(*gds_net, *job.net);
    structs[index] = std::move(gds_net);
  }
  return structs;
}

}  // namespace

auto GdsWriter::formatValue(double value, const std::string& unit) -> std::string
{
  return formatCapacitance(value, unit);
}

auto GdsWriter::write(const Model& model, const Config& config) const -> bool
{
  const auto gds_name = string::identifier(model.design_name, "plot_spef");
  const auto gds_file = path::file_under_dir(config.output_dir, gds_name, ".gds");

  idb::GdsData gds_data;
  gds_data.set_lib_name(model.design_name);
  gds_data.set_unit(1.0 / model.dbu, 1e-6 / model.dbu);
  gds_data.set_top_struct(new idb::GdsStruct(gds_name));

  const auto jobs = collectNetGdsJobs(model);
  auto net_structs = buildNetGdsStructs(model, config, jobs);

  idb::GdsiiTextWriter writer;
  if (!writer.init(gds_file.string(), &gds_data)) {
    LOG_ERROR << "plot_spef failed: cannot open GDS output file " << gds_file.string();
    return false;
  }
  if (!writer.begin()) {
    LOG_ERROR << "plot_spef failed: cannot begin GDS output file " << gds_file.string();
    return false;
  }

  for (std::size_t index = 0; index < jobs.size(); ++index) {
    addTopReference(gds_data, jobs[index].struct_name);
    gds_data.add_struct(net_structs[index].release());
  }
  if (!writer.finish()) {
    return false;
  }
  LOG_INFO << "plot_spef wrote GDS to " << gds_file.string();
  return true;
}

}  // namespace ircx::plot_spef
