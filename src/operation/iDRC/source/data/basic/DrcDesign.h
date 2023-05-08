#ifndef IDRC_SRC_DB_DRC_DESIGN_H_
#define IDRC_SRC_DB_DRC_DESIGN_H_

#include <memory>
#include <utility>

#include "BoostType.h"
#include "DrcNet.h"
#include "DrcPolygon.h"
#include "DrcSpot.h"
#include "DrcViolationSpot.h"
#include "Tech.h"

namespace idrc {
class DrcDesign
{
 public:
  DrcDesign() {}
  ~DrcDesign()
  {
    clear_drc_net_list();
    clear_blockage_list();
  }

  // setter
  void add_drc_net(DrcNet* drc_net) { _drc_net_list.push_back(drc_net); }
  void add_blockage(int routingLayerId, DrcRect* block_rect) { _layer_to_blockage_list[routingLayerId].push_back(block_rect); }
  void add_blockage(int routingLayerId, const BoostRect& block_rect) { _blockage_polygon_set_list[routingLayerId] += block_rect; }
  DrcPolygon* add_blockage_polygon(int routingLayerId, const PolygonWithHoles& drc_poly)
  {
    std::unique_ptr<DrcPolygon> drcPolygon = std::make_unique<DrcPolygon>(routingLayerId, drc_poly);
    DrcPolygon* drc_polygon = drcPolygon.get();
    _layer_to_drc_block_polygon_list[routingLayerId].push_back(std::move(drcPolygon));
    return drc_polygon;
  }
  // getter
  std::vector<DrcNet*>& get_drc_net_list() { return _drc_net_list; }
  std::map<int, std::vector<DrcRect*>>& get_layer_to_blockage_list() { return _layer_to_blockage_list; }
  std::map<int, bp::polygon_90_set_data<int>>& get_blockage_polygon_set_list() { return _blockage_polygon_set_list; }
  // function
  int get_net_num() { return _drc_net_list.size(); }
  DrcNet* add_drc_net()
  {
    DrcNet* drc_net = new DrcNet();
    _drc_net_list.push_back(drc_net);
    return drc_net;
  }
  // DrcRect* add_blockage(int routingLayerId, DrcRect* drcRect)
  // {
  //   DrcRect* block_rect = new DrcRect();
  //   _blockage_list.push_back(block_rect);
  //   return block_rect;
  // }
  void clear_drc_net_list()
  {
    for (auto net : _drc_net_list) {
      if (net != nullptr) {
        delete net;
        net = nullptr;
      }
    }
    _drc_net_list.clear();
  }

  void clear_blockage_list()
  {
    for (auto& [layer_id, blockage_list] : _layer_to_blockage_list) {
      for (auto blockage : blockage_list) {
        if (blockage != nullptr) {
          delete blockage;
          blockage = nullptr;
        }
      }
    }
    _layer_to_blockage_list.clear();
  }

  void clear_blockage_polygon_set_list() { _blockage_polygon_set_list.clear(); }

 private:
  std::vector<DrcNet*> _drc_net_list;
  std::map<int, std::vector<DrcRect*>> _layer_to_blockage_list;
  ////////////////////////////////////////////////////
  //下面两个目前工程上没用，multi-patterning相关
  std::map<int, bp::polygon_90_set_data<int>> _blockage_polygon_set_list;
  std::map<int, std::vector<std::unique_ptr<DrcPolygon>>> _layer_to_drc_block_polygon_list;
  // std::vector<DrcRect*> _instance_list;
};
}  // namespace idrc

#endif