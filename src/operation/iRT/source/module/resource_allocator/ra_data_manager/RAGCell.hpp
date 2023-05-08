#pragma once

#include "PlanarRect.hpp"
#include "RANet.hpp"
#include "RANetNode.hpp"
#include "RTU.hpp"

namespace irt {

class RAGCell
{
 public:
  RAGCell() = default;
  ~RAGCell() = default;
  // getter
  PlanarRect& get_real_rect() { return _real_rect; }
  std::map<irt_int, std::vector<PlanarRect>>& get_layer_blockage_map() { return _layer_blockage_map; }
  irt_int get_public_track_supply() const { return _public_track_supply; }
  std::vector<RANetNode>& get_ra_net_node_list() { return _ra_net_node_list; }
  // setter
  void set_real_rect(const PlanarRect& real_rect) { _real_rect = real_rect; }
  void set_public_track_supply(const irt_int public_track_supply) { _public_track_supply = public_track_supply; }
  void set_ra_net_node_list(const std::vector<RANetNode>& ra_net_node_list) { _ra_net_node_list = ra_net_node_list; }
  // function

 private:
  PlanarRect _real_rect;
  std::map<irt_int, std::vector<PlanarRect>> _layer_blockage_map;
  irt_int _public_track_supply = 0;
  std::vector<RANetNode> _ra_net_node_list;
};

}  // namespace irt
