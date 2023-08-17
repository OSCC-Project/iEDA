#pragma once
#include <boost/serialization/base_object.hpp>
#include <stdexcept>
#include <type_traits>

#include "AccessPoint.hpp"
#include "ConnectType.hpp"
#include "EXTLayerRect.hpp"
#include "EXTPlanarCoord.hpp"
#include "EXTPlanarRect.hpp"
#include "MTree.hpp"
#include "Net.hpp"
#include "PHYNode.hpp"
#include "PinNode.hpp"
#include "PlanarCoord.hpp"
#include "PlanarRect.hpp"
#include "TNode.hpp"
#include "WireNode.hpp"
#include "serialize.hpp"

namespace irt {

// ----------------------------------------------------------------------------
// Save and Load functions for irt::Net
// ----------------------------------------------------------------------------
template <typename Archive>
void save(Archive& ar, irt::Net& net, const unsigned int version)
{
  int32_t net_idx = net.get_net_idx();
  auto connect_type = net.get_connect_type();

  auto& pin_list = net.get_pin_list();
  iplf::Archive(ar, net_idx, net.get_net_name(), connect_type, pin_list, net.get_driving_pin(), net.get_bounding_box());
  iplf::Archive(ar, net.get_gr_result_tree(), net.get_ta_result_tree(), net.get_dr_result_tree(), net.get_vr_result_tree());
}

template <typename Archive>
void load(Archive& ar, irt::Net& net, const unsigned int version)
{
  int32_t net_idx;
  irt::ConnectType connect_type = net.get_connect_type();
  std::string net_name;
  iplf::Archive(ar, net_idx, net_name, connect_type, net.get_pin_list(), net.get_driving_pin(), net.get_bounding_box());
  iplf::Archive(ar, net.get_gr_result_tree(), net.get_ta_result_tree(), net.get_dr_result_tree(), net.get_vr_result_tree());
  net.set_connect_type(connect_type);
  net.set_net_idx(net_idx);
}

// ----------------------------------------------------------------------------
// Save and Load functions for irt::Pin
// ----------------------------------------------------------------------------
template <typename Archive>
void save(Archive& ar, irt::Pin& pin, const unsigned int version)
{
  iplf::Archive(ar, pin.get_pin_name(), pin.get_routing_shape_list(), pin.get_cut_shape_list(), pin.get_access_point_list());
}

template <typename Archive>
void load(Archive& ar, irt::Pin& pin, const unsigned int version)
{
  iplf::Archive(ar, pin.get_pin_name(), pin.get_routing_shape_list(), pin.get_cut_shape_list(), pin.get_access_point_list());
}

// ----------------------------------------------------------------------------
// Save and Load functions for irt::EXTLayerRect
// ----------------------------------------------------------------------------
template <typename Archive>
void save(Archive& ar, irt::EXTLayerRect& rect, const unsigned int version)
{
  int layer_idx = rect.get_layer_idx();
  iplf::Archive(ar, layer_idx, rect.get_grid_rect(), rect.get_real_rect());
}
template <typename Archive>
void load(Archive& ar, irt::EXTLayerRect& rect, const unsigned int version)
{
  int layer_idx;
  iplf::Archive(ar, layer_idx, rect.get_grid_rect(), rect.get_real_rect());
  rect.set_layer_idx(layer_idx);
}

// ----------------------------------------------------------------------------
// Serialize functions for irt::PlanarRect
// ----------------------------------------------------------------------------
template <typename Archive>
void serialize(Archive& ar, irt::PlanarRect& rect, const unsigned int version)
{
  iplf::Archive(ar, rect.get_lb(), rect.get_rt());
}

// ----------------------------------------------------------------------------
// Serialize functions for irt::PlanarCoord
// ----------------------------------------------------------------------------
template <typename Archive>
void serialize(Archive& ar, irt::PlanarCoord& coord, const unsigned int version)
{
  int x = coord.get_x();
  int y = coord.get_y();
  iplf::Archive(ar, x, y);
  if constexpr (Archive::is_loading::value) {
    coord.set_coord(x, y);
  }
}

// ----------------------------------------------------------------------------
// Serialize functions for irt::AccessPoint
// ----------------------------------------------------------------------------
template <typename Archive>
void serialize(Archive& ar, irt::AccessPoint& ap, const unsigned int version)
{
  auto type = ap.get_type();
  int layer_idx = ap.get_layer_idx();
  iplf::Archive(ar, type, layer_idx, ap.get_grid_coord(), ap.get_real_coord());
  if constexpr (Archive::is_loading::value) {
    ap.set_type(type);
    ap.set_layer_idx(layer_idx);
  }
}
// ----------------------------------------------------------------------------
// Serialize functions for irt::EXTPlanarCoord
// ----------------------------------------------------------------------------
template <typename Archive>
void serialize(Archive& ar, irt::EXTPlanarCoord& coord, const unsigned int version)
{
  iplf::Archive(ar, coord.get_grid_coord(), coord.get_real_coord());
}

// ----------------------------------------------------------------------------
// Serialize functions for irt::EXTPlanarRect
// ----------------------------------------------------------------------------
template <typename Archive>
void serialize(Archive& ar, irt::EXTPlanarRect& rect, const unsigned int version)
{
  iplf::Archive(ar, rect.get_grid_rect(), rect.get_real_rect());
}

// ----------------------------------------------------------------------------
// Serialize functions for irt::BoundingBox
// ----------------------------------------------------------------------------
template <typename Archive>
void serialize(Archive& ar, irt::BoundingBox& box, const unsigned int version)
{
  iplf::Archive(ar, boost::serialization::base_object<EXTPlanarRect>(box));
}

// ----------------------------------------------------------------------------
// Save and Load functions for irt::GridMap<double>
// ----------------------------------------------------------------------------
template <typename Archive>
void save(Archive& ar, irt::GridMap<double>& costmap, const unsigned int version)
{
  int xsize = costmap.get_x_size();
  int ysize = costmap.get_y_size();
  ar << xsize << ysize;
  for (int x = 0; x < xsize; ++x) {
    for (int y = 0; y < ysize; ++y) {
      ar << costmap[x][y];
    }
  }
}

template <typename Archive>
void load(Archive& ar, irt::GridMap<double>& costmap, const unsigned int version)
{
  int xsize;
  int ysize;
  ar >> xsize >> ysize;
  irt::GridMap<double> tmp(xsize, ysize);
  for (int x = 0; x < xsize; ++x) {
    for (int y = 0; y < ysize; ++y) {
      ar >> tmp[x][y];
    }
  }
  costmap = std::move(tmp);
}

// ----------------------------------------------------------------------------
// Serialize functions for irt::TNode<T>
// ----------------------------------------------------------------------------
template <typename Archive, typename T>
void serialize(Archive& ar, irt::TNode<T>& tnode, const unsigned int version)
{
  iplf::Archive(ar, tnode.value(), tnode.get_child_list());
}

// ----------------------------------------------------------------------------
// Save and Load functions for irt::MTree<T>
// ----------------------------------------------------------------------------
template <typename Archive, typename T>
void save(Archive& ar, irt::MTree<T>& mtree, const unsigned int version)
{
  ar << mtree.get_root();
}

template <typename Archive, typename T>
void load(Archive& ar, irt::MTree<T>& mtree, const unsigned int version)
{
  decltype(mtree.get_root()) tnode = nullptr;
  ar >> tnode;
  mtree.clear();
  mtree.set_root(tnode);
}

// ----------------------------------------------------------------------------
// Serialize functions for irt::LayerRect
// ----------------------------------------------------------------------------
template <typename Archive>
void serialize(Archive& ar, irt::LayerRect& rect, const unsigned int version)
{
  ar& boost::serialization::base_object<irt::PlanarRect>(rect);
  int layer_id = rect.get_layer_idx();
  ar& layer_id;
  if constexpr (Archive::is_loading::value) {
    rect.set_layer_idx(layer_id);
  }
}

// ----------------------------------------------------------------------------
// Serialize functions for irt::LayerCoord
// ----------------------------------------------------------------------------
template <typename Archive>
void serialize(Archive& ar, irt::LayerCoord& coord, const unsigned int version)
{
  ar& boost::serialization::base_object<irt::PlanarCoord>(coord);
  int layer_id = coord.get_layer_idx();
  ar& layer_id;
  if constexpr (Archive::is_loading::value) {
    coord.set_layer_idx(layer_id);
  }
}

// ----------------------------------------------------------------------------
// Serialize functions for irt::Guide
// ----------------------------------------------------------------------------
template <typename Archive>
void serialize(Archive& ar, irt::Guide& guide, const unsigned int version)
{
  ar& boost::serialization::base_object<irt::LayerRect>(guide);
  ar& guide.get_grid_coord();
}

// ----------------------------------------------------------------------------
// Serialize functions for irt::Segment<T>
// ----------------------------------------------------------------------------
template <typename Archive, typename T>
void serialize(Archive& ar, irt::Segment<T>& segment, const unsigned int version)
{
  ar& segment.get_first();
  ar& segment.get_second();
}

// ----------------------------------------------------------------------------
// Serialize functions for irt::RTNode
// ----------------------------------------------------------------------------
template <typename Archive>
void serialize(Archive& ar, irt::RTNode& rt, const unsigned int version)
{
  iplf::Archive(ar, rt.get_first_guide(), rt.get_second_guide(), rt.get_pin_idx_set(), rt.get_routing_tree());
}

enum class NodeType : int
{
  monostate = 0,
  PinNode,
  WireNode,
  ViaNode
};
static NodeType PHYNodeType(irt::PHYNode& node)
{
  if (node.isEmpty()) {
    return NodeType::monostate;
  }
  if (node.isType<irt::PinNode>()) {
    return NodeType::PinNode;
  }
  if (node.isType<irt::WireNode>()) {
    return NodeType::WireNode;
  }
  if (node.isType<irt::ViaNode>()) {
    return NodeType::ViaNode;
  }
  assert(false);
  return NodeType::monostate;
}

template <typename Archive>
void serialize(Archive& ar, irt::PHYNode& node, const unsigned int version)
{
  NodeType type = PHYNodeType(node);
  ar& type;
  switch (type) {
    case NodeType::monostate: {
      break;
    }
    case NodeType::PinNode: {
      ar& node.getNode<irt::PinNode>();
      break;
    }
    case NodeType::WireNode: {
      ar& node.getNode<irt::WireNode>();
      break;
    }
    case NodeType::ViaNode: {
      ar& node.getNode<irt::ViaNode>();
      break;
    }
  }
}

template <typename Archive>
void serialize(Archive& ar, irt::PinNode& node, const unsigned int version)
{
  int net_idx = node.get_net_idx();
  int pin_idx = node.get_pin_idx();
  int layer_idx = node.get_layer_idx();

  iplf::Archive(ar, net_idx, pin_idx, layer_idx);
  iplf::Archive(ar, node.get_planar_coord());
  if constexpr (Archive::is_loading::value) {
    node.set_net_idx(net_idx);
    node.set_pin_idx(pin_idx);
    node.set_layer_idx(layer_idx);
  }
}

template <typename Archive>
void serialize(Archive& ar, irt::WireNode& node, const unsigned int version)
{
  int net_idx = node.get_net_idx();
  int layer_idx = node.get_layer_idx();
  int wire_width = node.get_wire_width();
  iplf::Archive(ar, net_idx, layer_idx, wire_width);
  iplf::Archive(ar, node.get_first(), node.get_second());
  if constexpr (Archive::is_loading::value) {
    node.set_net_idx(net_idx);
    node.set_layer_idx(layer_idx);
    node.set_wire_width(wire_width);
  }
}

template <typename Archive>
void serialize(Archive& ar, irt::ViaNode& node, const unsigned int version)
{
  int net_idx = node.get_net_idx();
  int below_layer_idx = node.get_via_master_idx().get_below_layer_idx();
  int via_idx = node.get_via_master_idx().get_via_idx();
  iplf::Archive(ar, net_idx, below_layer_idx, via_idx, boost::serialization::base_object<irt::PlanarCoord>(node));
  if constexpr (Archive::is_loading::value) {
    node.set_net_idx(net_idx);
    node.get_via_master_idx().set_below_layer_idx(below_layer_idx);
    node.get_via_master_idx().set_via_idx(via_idx);
  }
}
template <typename Archive>
void save(Archive& ar, const std::vector<irt::Net>& net_list, const unsigned int version)
{
  size_t sz = net_list.size();
  ar << sz;
  for (size_t i = 0; i < sz; ++i) {
    ar << net_list[i];
  }
}
template <typename Archive>
void load(Archive& ar, std::vector<irt::Net>& net_list, const unsigned int version)
{
  size_t sz;
  ar >> sz;
  if (sz != net_list.size()) {
    throw std::runtime_error("serialized netlist size doesnt match");
  }
  for (size_t i = 0; i < sz; ++i) {
    ar >> net_list[i];
  }
}

}  // namespace irt