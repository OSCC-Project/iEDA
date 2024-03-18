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
#include "PlanarCoord.hpp"
#include "PlanarRect.hpp"
#include "TNode.hpp"
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
  iplf::Archive(ar, net_idx, net.get_net_name(), connect_type, pin_list, net.get_bounding_box());
  iplf::Archive(ar, net.get_ir_result_tree(), net.get_gr_result_tree());
}

template <typename Archive>
void load(Archive& ar, irt::Net& net, const unsigned int version)
{
  int32_t net_idx;
  irt::ConnectType connect_type = net.get_connect_type();
  std::string net_name;
  iplf::Archive(ar, net_idx, net_name, connect_type, net.get_pin_list(), net.get_bounding_box());
  iplf::Archive(ar, net.get_ir_result_tree(), net.get_gr_result_tree());
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
  ar & layer_id;
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
  ar & layer_id;
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
  ar & guide.get_grid_coord();
}

// ----------------------------------------------------------------------------
// Serialize functions for irt::Segment<T>
// ----------------------------------------------------------------------------
template <typename Archive, typename T>
void serialize(Archive& ar, irt::Segment<T>& segment, const unsigned int version)
{
  ar & segment.get_first();
  ar & segment.get_second();
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