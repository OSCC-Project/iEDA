#pragma once

#include "DRCUtil.h"
#include "DrcConfig.h"
#include "DrcDesign.h"
#include "RegionQuery.h"
#include "Tech.h"
namespace idrc {
class RoutingAreaCheck
{
 public:
  // static RoutingAreaCheck* getInstance(DrcConfig* config = nullptr, Tech* tech = nullptr)
  // {
  //   static RoutingAreaCheck instance(config, tech);
  //   return &instance;
  // }
  //-----------------------
  RoutingAreaCheck(DrcConfig* config, Tech* tech, RegionQuery* region_query) { init(config, tech, region_query); }
  RoutingAreaCheck(Tech* tech, RegionQuery* region_query) : _tech(tech), _region_query(region_query) { _interact_with_op = true; }
  RoutingAreaCheck(const RoutingAreaCheck& other) = delete;
  RoutingAreaCheck(RoutingAreaCheck&& other) = delete;
  ~RoutingAreaCheck() {}
  RoutingAreaCheck& operator=(const RoutingAreaCheck& other) = delete;
  RoutingAreaCheck& operator=(RoutingAreaCheck&& other) = delete;

  // getter
  // operation api
  bool check(DrcNet* target_net);
  bool check(DrcPoly* target_poly);

  // function
  void checkArea(DrcNet* target_net);
  void checkArea(DrcPoly* target_poly);

  void checkRoutingArea(DrcNet* net);
  void checkRoutingArea(const LayerNameToRTreeMap& layer_to_rects_tree_map);

  std::map<int, std::vector<DrcSpot>>& get_routing_layer_to_spots_map() { return _routing_layer_to_spots_map; }
  int get_area_violation_num();
  void reset();

 private:
  int _rule_index;
  DrcConfig* _config;
  Tech* _tech;
  RegionQuery* _region_query;
  std::vector<std::shared_ptr<idb::routinglayer::Lef58Area>> _lef58_rules;
  std::map<int, std::vector<BoostPolygon>> _layer_to_polygons_map;
  std::map<int, std::vector<DrcSpot>> _routing_layer_to_spots_map;

  //----------interact with other operations----------
  bool _interact_with_op = false;
  bool _check_result = true;

  // function
  void init(DrcConfig* config, Tech* tech, RegionQuery* region_query);

  void initLayerPolygonSet(DrcNet* target_net);
  void initLayerToPolygonSetFromRoutingRects(DrcNet* target_net);
  void initLayerToPolygonSetFromPinRects(DrcNet* target_net);
  void checkRoutingArea();
  void add_spot(int layerId, const DrcRectangle<int>& vialation_box, ViolationType type);

  void checkRoutingArea(int netId);
  bool skipCheck(int layerId, int netId, const BoostPolygon& target_polygon, const RTreeBox& query_box);

  /////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////interact with iRT
  void initLayerPolygonSet(const LayerNameToRTreeMap& layer_to_rects_tree_map);

  /////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////another way to check
  void initLayerPolygonSet();
  void checkRoutingAreaLayerByLayer();
  bool checkLef58Area(DrcPoly* target_poly);
  bool checkMaxEdge(DrcPoly* target_poly);
  bool checkMinEdge(DrcPoly* target_poly);
  bool checkMinSize(DrcPoly* target_poly);
  bool checkMinimumArea(DrcPoly* target_poly);
  void add_spot();
  void addSpot(DrcPoly* target_poly);
};
}  // namespace idrc
