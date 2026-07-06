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
#include <iterator>

#include "TopoPool.hh"
#include "log/Log.hh"
namespace ircx {

// TopoEdge

void TopoEdge::set_shape(const GtlRectI& v) {
  shape_ = v;

  line_seg_.is_horz = geom::is_horizontal_dominant(shape_);

  width_ = line_seg_.is_horz ? geom::delta_y(shape_) : geom::delta_x(shape_);
  half_width_ = width_ / 2;
  length_ = line_seg_.is_horz ? geom::delta_x(shape_) : geom::delta_y(shape_);
  center_ = geom::center(shape_);

  line_seg_.coord = line_seg_.is_horz ? geom::center_y(shape_) : geom::center_x(shape_);
  line_seg_.lo    = line_seg_.is_horz ? geom::min_x(shape_)    : geom::min_y(shape_);
  line_seg_.hi    = line_seg_.is_horz ? geom::max_x(shape_)    : geom::max_y(shape_);
}

// TopoPool

void TopoPool::reserve(Size net_count, Size total_nodes, Size total_edges) {
  net_node_ranges_.reserve(net_count);
  net_edge_ranges_.reserve(net_count);
  node_pool_.reserve(total_nodes);
  edge_pool_.reserve(total_edges);
}

void TopoPool::clear()
{
  node_pool_.clear();
  net_node_ranges_.clear();
  edge_pool_.clear();
  net_edge_ranges_.clear();
  special_edge_pool_.clear();
}

std::span<const TopoNode> TopoPool::net_nodes(Size net_id) const {
  LOG_FATAL_IF(net_id >= net_node_ranges_.size()) << "net_id out of range.";
  auto [offset, count] = net_node_ranges_[net_id];
  return std::span<const TopoNode>(node_pool_).subspan(offset, count);
}

std::span<const TopoEdge> TopoPool::net_edges(Size net_id) const {
  LOG_FATAL_IF(net_id >= net_edge_ranges_.size()) << "net_id out of range.";
  auto [offset, count] = net_edge_ranges_[net_id];
  return std::span<const TopoEdge>(edge_pool_).subspan(offset, count);
}

std::span<TopoNode> TopoPool::net_nodes(Size net_id) {
  LOG_FATAL_IF(net_id >= net_node_ranges_.size()) << "net_id out of range.";
  auto [offset, count] = net_node_ranges_[net_id];
  return std::span<TopoNode>(node_pool_).subspan(offset, count);
}

std::span<TopoEdge> TopoPool::net_edges(Size net_id) {
  LOG_FATAL_IF(net_id >= net_edge_ranges_.size()) << "net_id out of range.";
  auto [offset, count] = net_edge_ranges_[net_id];
  return std::span<TopoEdge>(edge_pool_).subspan(offset, count);
}

std::pair<Size, Size> TopoPool::net_node_range(Size net_id) const {
  LOG_FATAL_IF(net_id >= net_node_ranges_.size()) << "net_id out of range.";
  return net_node_ranges_[net_id];
}

std::pair<Size, Size> TopoPool::net_edge_range(Size net_id) const {
  LOG_FATAL_IF(net_id >= net_edge_ranges_.size()) << "net_id out of range.";
  return net_edge_ranges_[net_id];
}

void TopoPool::addNet(std::vector<TopoNode> nodes,
                      std::vector<TopoEdge> edges)
{
  const Size node_off = node_pool_.size();
  const Size node_cnt = nodes.size();
  const Size edge_off = edge_pool_.size();
  const Size edge_cnt = edges.size();

  // Assign LOCAL ids (0..count-1) for the id() accessor.
  for (Size node_idx = 0; node_idx < nodes.size(); ++node_idx) nodes[node_idx].set_id(node_idx);
  for (Size edge_idx = 0; edge_idx < edges.size(); ++edge_idx) edges[edge_idx].set_id(edge_idx);

  net_node_ranges_.emplace_back(node_off, node_cnt);
  net_edge_ranges_.emplace_back(edge_off, edge_cnt);

  node_pool_.reserve(node_off + node_cnt);
  edge_pool_.reserve(edge_off + edge_cnt);

  node_pool_.insert(node_pool_.end(),
                    std::make_move_iterator(nodes.begin()),
                    std::make_move_iterator(nodes.end()));
  edge_pool_.insert(edge_pool_.end(),
                    std::make_move_iterator(edges.begin()),
                    std::make_move_iterator(edges.end()));

}

void TopoPool::addSpecialEdges(std::vector<TopoEdge> edges)
{
  for (Size edge_idx = 0; edge_idx < edges.size(); ++edge_idx) edges[edge_idx].set_id(edge_idx);

  special_edge_pool_.reserve(special_edge_pool_.size() + edges.size());
  special_edge_pool_.insert(special_edge_pool_.end(),
                            std::make_move_iterator(edges.begin()),
                            std::make_move_iterator(edges.end()));
}

}  // namespace ircx
