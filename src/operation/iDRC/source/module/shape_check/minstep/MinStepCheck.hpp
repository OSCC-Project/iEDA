#pragma once

#include "DRCUtil.h"
#include "DrcConfig.h"
#include "DrcDesign.h"
#include "RegionQuery.h"
#include "Tech.h"
namespace idrc {
class MinStepCheck
{
 public:
  // static MinStepCheck* getInstance(DrcConfig* config = nullptr, Tech* tech = nullptr)
  // {
  //   static MinStepCheck instance(config, tech);
  //   return &instance;
  // }
  //-----------------------
  MinStepCheck(DrcConfig* config, Tech* tech) { init(config, tech); }
  MinStepCheck(Tech* tech, RegionQuery* region_query) : _tech(tech), _region_query(region_query) { _interact_with_op = true; }
  MinStepCheck(const MinStepCheck& other) = delete;
  MinStepCheck(MinStepCheck&& other) = delete;
  ~MinStepCheck() {}
  MinStepCheck& operator=(const MinStepCheck& other) = delete;
  MinStepCheck& operator=(MinStepCheck&& other) = delete;

  // api
  bool check(DrcNet* target_net);
  bool check(DrcPoly* target_poly);

  // getter

  // function
  void checkMinStep(DrcNet* target_net);

 private:
  int _rule_index;
  DrcEdge* _begin_edge;
  DrcEdge* _end_edge;
  DrcEdge* _trigger_edge;
  std::set<DrcEdge*> _vio_edge_set;
  DrcConfig* _config;
  Tech* _tech;
  RegionQuery* _region_query;
  std::vector<std::shared_ptr<idb::routinglayer::Lef58MinStep>> _lef58_rule;
  std::shared_ptr<idb::IdbMinStep> _rule;
  std::map<int, std::vector<BoostPolygon>> _layer_to_polygons_map;
  std::map<int, std::vector<DrcSpot>> _routing_layer_to_spots_map;

  //----------interact with other operations----------
  bool _interact_with_op = false;
  bool _check_result = true;

  // function
  void init(DrcConfig* config, Tech* tech);

  void checkMinStep(DrcPoly* target_poly);
  bool isTriggerEdge_lef58(DrcEdge* edge);
  void checkAdjEdgeLength(DrcEdge* edge, DrcEdge* adj_edge);
  void checkNextEdge_lef58(DrcEdge* edge);
  void checkPrevEdge_lef58(DrcEdge* edge);
  void checkMinStep_lef58(DrcEdge* edge);
  void checkMinStep(DrcEdge* edge);
  void refresh(DrcEdge* edge);
  bool checkBeginCorner();
  bool checkEndCorner();
  bool checkCorner();
  bool isTriggerEdge(DrcEdge* edge);

  void addSpot(DrcEdge* begin_edge, DrcEdge* end_edge);
  void addSpot_lef58(DrcEdge* edge, DrcEdge* adj_edge);
};
}  // namespace idrc
