#ifndef IDRC_SRC_DB_TECH_H_
#define IDRC_SRC_DB_TECH_H_

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "DrcLayer.h"
#include "DrcVia.h"

namespace idrc {
class Tech
{
 public:
  Tech() {}
  ~Tech()
  {
    clear_drc_routing_layer_list();
    clear_drc_cut_layer_list();
    clear_via_lib();
  }
  // setter
  // DrcRoutingLayer* add_routing_layer();
  // DrcCutLayer* add_cut_layer();
  // DrcVia* add_via();
  // getter
  std::vector<DrcRoutingLayer*>& get_drc_routing_layer_list() { return _drc_routing_layer_list; }
  std::vector<DrcCutLayer*>& get_drc_cut_layer_list() { return _drc_cut_layer_list; }
  std::vector<DrcVia*>& get_via_lib() { return _via_lib; }
  // function
  int getRoutingWidth(int routingLayerId);
  int getRoutingSpacing(int routingLayerId, int width);
  int getRoutingMinWidth(int routingLayerId);
  // int getRoutingMinArea(int routingLayerId);
  int getRoutingMinEnclosedArea(int routingLayerId);
  int getRoutingMaxRequireSpacing(int routingLayerId, DrcRect* target_rect);
  DrcVia* findViaByIdx(int idx) { return idx >= 0 && idx < (int) _via_lib.size() ? (_via_lib[idx]) : nullptr; }
  // LayerDirection getLayerDirection(int routingLayerId);
  int getCutSpacing(int cutLayerId);
  // DrcCutLayer* getCutLayerById(int layer_id);

  std::string getCutLayerNameById(int layer_id);
  std::string getRoutingLayerNameById(int layer_id);
  ///////////
  std::pair<bool, int> getLayerInfoByLayerName(const std::string& name);

  int getLayerIdByLayerName(const std::string& name);
  // clear
  void clear_drc_routing_layer_list();
  void clear_drc_cut_layer_list();
  void clear_via_lib();

 private:
  std::vector<DrcRoutingLayer*> _drc_routing_layer_list;
  std::vector<DrcCutLayer*> _drc_cut_layer_list;
  std::vector<DrcVia*> _via_lib;
};
}  // namespace idrc

#endif