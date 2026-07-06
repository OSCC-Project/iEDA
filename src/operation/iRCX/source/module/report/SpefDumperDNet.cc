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
#include "SpefDumper.hh"

#include <algorithm>
#include <iomanip>
#include <iterator>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Geometry.hh"
#include "LayoutData.hh"
#include "RCTable.hh"
#include "TopoPool.hh"

namespace ircx {

namespace {

auto pinIo(const Pin& pin) -> char
{
  if (pin.is_input && pin.is_output) {
    return 'B';
  }
  if (pin.is_input) {
    return 'I';
  }
  if (pin.is_output) {
    return 'O';
  }
  return 'B';
}

}  // namespace

void SpefDumper::writeDNet(std::ostream& os, Size corner_idx, Size net_idx) const
{
  const Net& net = layout_data_->net_vec[net_idx];
  const Str& net_spef_name = net_spef_names_[net_idx];
  const auto& node_spef_name = node_spef_names_;
  const auto& coupling_refs = net_coupling_refs_[net_idx];
  const auto& edge_pool = topo_pool_->edge_pool();

  std::unordered_map<Str, char> inst_pin_io;
  inst_pin_io.reserve(net.pins.size());
  for (const Pin& pin : net.pins) {
    inst_pin_io[pin.name] = pinIo(pin);
  }

  const auto nodes = topo_pool_->net_nodes(net_idx);
  const auto edges = topo_pool_->net_edges(net_idx);
  const Size node_offset = topo_pool_->net_node_range(net_idx).first;

  const Size node_num = nodes.size();
  if (node_num == 0) {
    return;
  }

  const Micron micron_per_dbu = unit::to_micron(1, layout_data_->dbu_per_micron);

  auto gcap_pool = rc_table_->corner_net_gcap_pool({corner_idx, net_idx});
  std::vector<double> node_gnd(node_num, 0.0);

  for (Size edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
    const TopoEdge& edge = edges[edge_idx];
    if (edge.is_via()) {
      continue;
    }
    const double ground_cap = gcap_pool[edge_idx];
    if (ground_cap <= 0.0) {
      continue;
    }

    node_gnd[edge.u() - node_offset] += ground_cap / 2.0;
    node_gnd[edge.v() - node_offset] += ground_cap / 2.0;
  }

  // key: (local node index on this net, SPEF node name on the other net)
  std::map<std::pair<Size, Str>, double> node_cc_map;

  for (const CouplingRef& coupling : coupling_refs) {
    const TopoEdge& e_self = edge_pool[coupling.self_edge_id];
    const TopoEdge& e_other = edge_pool[coupling.other_edge_id];

    if (e_self.u() == kMaxSize || e_self.v() == kMaxSize || e_other.u() == kMaxSize || e_other.v() == kMaxSize) {
      continue;
    }

    const GtlPointI& pa_u = topo_pool_->node_at(e_self.u()).point();
    const GtlPointI& pa_v = topo_pool_->node_at(e_self.v()).point();
    const GtlPointI& pb_u = topo_pool_->node_at(e_other.u()).point();
    const GtlPointI& pb_v = topo_pool_->node_at(e_other.v()).point();

    struct Candidate
    {
      Size self_global;
      Size other_global;
      Dbu dist;
    };

    const Candidate candidates[4] = {
        {e_self.u(), e_other.u(), geom::manhattan_distance(pa_u, pb_u)},
        {e_self.u(), e_other.v(), geom::manhattan_distance(pa_u, pb_v)},
        {e_self.v(), e_other.u(), geom::manhattan_distance(pa_v, pb_u)},
        {e_self.v(), e_other.v(), geom::manhattan_distance(pa_v, pb_v)},
    };
    const auto& best = *std::min_element(
        std::begin(candidates), std::end(candidates), [](const Candidate& lhs, const Candidate& rhs) { return lhs.dist < rhs.dist; });

    const Size self_local = best.self_global - node_offset;
    const Str& other_node = node_spef_name[best.other_global];
    node_cc_map[{self_local, other_node}] += coupling.cap_ff;
  }

  double tcap = 0.0;
  for (double ground_cap : node_gnd) {
    tcap += ground_cap;
  }
  for (const auto& [_, coupling_cap] : node_cc_map) {
    tcap += coupling_cap;
  }

  os << "\n*D_NET " << net_spef_name << " " << std::fixed << std::setprecision(6) << tcap << "\n\n";

  os << "*CONN\n";
  for (const TopoNode& node : nodes) {
    if (!node.is_pin_node() || node.pin_name().find(':') != Str::npos) {
      continue;
    }

    const Micron x = geom::x(node.point()) * micron_per_dbu;
    const Micron y = geom::y(node.point()) * micron_per_dbu;
    os << "*P " << node_spef_name[topo_pool_->node_index(net_idx, node.id())] << " " << port_io_.at(node.pin_name()) << " *C " << std::fixed
       << std::setprecision(3) << x << " " << y;
    writeNodeGeometry(os, node, micron_per_dbu);
    os << "\n";
  }

  for (const TopoNode& node : nodes) {
    if (!node.is_pin_node() || node.pin_name().find(':') == Str::npos) {
      continue;
    }

    const Micron x = geom::x(node.point()) * micron_per_dbu;
    const Micron y = geom::y(node.point()) * micron_per_dbu;
    const auto io_it = inst_pin_io.find(node.pin_name());
    const char io = (io_it != inst_pin_io.end()) ? io_it->second : 'B';
    os << "*I " << node_spef_name[topo_pool_->node_index(net_idx, node.id())] << " " << io << " *C " << std::fixed << std::setprecision(3)
       << x << " " << y;
    writeNodeGeometry(os, node, micron_per_dbu);
    os << "\n";
  }

  for (const TopoNode& node : nodes) {
    if (node.is_pin_node()) {
      continue;
    }

    const Micron x = geom::x(node.point()) * micron_per_dbu;
    const Micron y = geom::y(node.point()) * micron_per_dbu;
    os << "*N " << node_spef_name[topo_pool_->node_index(net_idx, node.id())] << " *C " << std::fixed << std::setprecision(3) << x << " "
       << y;
    writeNodeGeometry(os, node, micron_per_dbu);
    os << "\n";
  }

  os << "\n*CAP\n";
  int cap_id = 1;
  for (const auto& [key, coupling_cap] : node_cc_map) {
    if (coupling_cap <= 0.0) {
      continue;
    }
    os << cap_id++ << " " << node_spef_name[topo_pool_->node_index(net_idx, key.first)] << " " << key.second << " " << std::setprecision(6)
       << coupling_cap << "\n";
  }
  for (Size node_idx = 0; node_idx < node_num; ++node_idx) {
    if (node_gnd[node_idx] <= 0.0) {
      continue;
    }
    os << cap_id++ << " " << node_spef_name[topo_pool_->node_index(net_idx, node_idx)] << " " << std::setprecision(6) << node_gnd[node_idx]
       << "\n";
  }

  os << "\n*RES\n";
  int res_id = 1;
  auto res_pool = rc_table_->corner_net_res_pool({corner_idx, net_idx});
  for (Size edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
    const TopoEdge& edge = edges[edge_idx];
    if (edge.u() == kMaxSize || edge.v() == kMaxSize) {
      continue;
    }

    const double resistance = res_pool[edge_idx];
    os << res_id++ << " " << node_spef_name[edge.u()] << " " << node_spef_name[edge.v()] << " " << std::setprecision(6) << resistance;
    writeResistanceGeometry(os, corner_idx, edge, micron_per_dbu);
    os << "\n";
  }

  os << "*END\n";
}

}  // namespace ircx
