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
#pragma once

#include <set>

#include "BoostType.h"
#include "DRCCOMUtil.h"
#include "DrcConfig.h"
#include "DrcDesign.h"
#include "DrcPoly.hpp"
#include "DrcRules.hpp"
#include "RegionQuery.h"
#include "Tech.h"
#include "builder.h"
#include "def_service.h"
#include "lef_service.h"

namespace idrc {

class DrcIDBWrapper
{
 public:
  explicit DrcIDBWrapper() {}
  DrcIDBWrapper(DrcConfig* config, Tech* tech) : _config(config), _tech(tech) {}
  DrcIDBWrapper(DrcConfig* config, Tech* tech, DrcDesign* design, RegionQuery* region_query)
      : _config(config), _tech(tech), _design(design), _region_query(region_query)
  {
  }
  DrcIDBWrapper(Tech* tech, DrcDesign* design, idb::IdbBuilder* in, RegionQuery* region_query)
      : _tech(tech), _design(design), _region_query(region_query), _db_builder(in)
  {
  }

  DrcIDBWrapper(Tech* tech, idb::IdbBuilder* in) : _tech(tech), _db_builder(in) {}
  DrcIDBWrapper(RegionQuery* region_query) : _region_query(region_query) {}
  DrcIDBWrapper(const DrcIDBWrapper& other) = delete;
  DrcIDBWrapper(DrcIDBWrapper&& other) = delete;
  ~DrcIDBWrapper()
  {
    // if (_db_builder != nullptr) {
    //   delete _db_builder;
    // }
  }
  DrcIDBWrapper& operator=(const DrcIDBWrapper& other) = delete;
  DrcIDBWrapper& operator=(DrcIDBWrapper&& other) = delete;
  // getter
  DrcConfig* get_config() { return _config; }
  Tech* get_tech() { return _tech; }
  DrcDesign* get_design() { return _design; }
  // setter

  // function
  void input(idb::IdbBuilder* idb_builder = nullptr);
  void initTech(idb::IdbBuilder* db_builder = nullptr);
  void wrapTech();
  void wrapDesign();

  // init polys
  void initPolyPolygon(DrcNet* net);
  void initPolyEdges(DrcNet* net);
  void initPolyCorners(DrcNet* net);
  void initPolyMaxRects(DrcNet* net);

  void getAllPolygonEdges(DrcNet* net, std::vector<std::set<std::pair<DrcCoordinate<int>, DrcCoordinate<int>>>>& polygon_edges);
  void initPolyOuterEdges(DrcNet* net, DrcPoly* poly, DrcPolygon* polygon, int layer_id);
  void initPolyInnerEdges(DrcNet* net, DrcPoly* poly, const BoostPolygon& hole_poly, int layer_id);
  void initPolyCornerMain(DrcNet* net, DrcPoly* poly);

 protected:
  // parser
  void initIDB(idb::IdbBuilder* db_builder);
  void initIDBLayout();

  void wrap();
  void wrapRoutingLayerList();
  void wrapCutLayerList();
  void wrapViaLib();
  void wrapBlockageList();
  void wrapNetList();
  void wrapNetPolyList();

  DrcNet* get_drc_net(int netId);

 private:
  DrcConfig* _config = nullptr;
  Tech* _tech = nullptr;
  DrcDesign* _design = nullptr;
  RegionQuery* _region_query = nullptr;
  idb::IdbBuilder* _db_builder = nullptr;
  std::map<int, DrcNet*> _id_to_net;

  /// basic rect wrap function
  //////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////
  /// tech
  void wrapRect(DrcRectangle<int>& rect, idb::IdbRect* idb_rect)
  {
    if (idb_rect == nullptr) {
      return;
    }

    rect.set_lb(idb_rect->get_low_x(), idb_rect->get_low_y());
    rect.set_rt(idb_rect->get_high_x(), idb_rect->get_high_y());
  }

  // getter
  idb::IdbBuilder* get_idb_builder() { return _db_builder; }
  idb::IdbDesign* get_idb_design() { return _db_builder->get_def_service()->get_design(); }
  idb::IdbLayout* get_idb_layout() { return _db_builder->get_lef_service()->get_layout(); }

  void wrapRoutingLayer(idb::IdbLayerRouting* idb_routing_layer, DrcRoutingLayer* layer);
  void wrapCutLayer(idb::IdbLayerCut* idb_cut_layer, DrcCutLayer* layer);
  void wrapVia(idb::IdbVia* idb_via, std::vector<DrcVia*>& via_list);
  ////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////
  /// design
  // rect convert
  void wrapRect(DrcRect* drc_rect, idb::IdbRect* idb_rect);
  BoostRect getBoostRectFromIdbRect(idb::IdbRect* idb_rect);

  // blockage list in def
  void wrapBlockageListInDef();
  void addPlacementBlockage(idb::IdbPlacementBlockage* idb_blockage, DrcDesign* design);
  void addRoutingBlockage(idb::IdbRoutingBlockage* idb_blockage, DrcDesign* design);
  void addIdbBlockToDrcDesign(idb::IdbBlockage* idb_blockage, DrcDesign* design);
  // instance as blockage
  void wrapInstanceListBlockage();
  void addInstanceBlockage(idb::IdbInstance* idb_instance, DrcDesign* design);
  void addInstanceObsBlockage(idb::IdbInstance* idb_instance, DrcDesign* design);
  void addInstancePinBlockage(idb::IdbInstance* idb_instance, DrcDesign* design);
  // specialnets as blokage
  void wrapSpecialNetListBlockage();
  void addSpecialNetBlockage(idb::IdbSpecialNet* idb_net, DrcDesign* design);
  void addSpecialNetViaBlockage(idb::IdbSpecialWireSegment* idb_segment, DrcDesign* design);
  void addSpecialNetWireBlockage(idb::IdbSpecialWireSegment* idb_segment, DrcDesign* design);
  /// basic blockage wrap function
  void wrapBlockageFromLayerShape(idb::IdbLayerShape* layer_shape, DrcDesign* design);
  // net
  bool addNetToDrcDesign(idb::IdbNet* idb_net, DrcDesign* design);
  // pin shape
  bool addPinListToNet(vector<idb::IdbPin*>& idb_pin_list, DrcNet* drc_net);
  bool addPinToNet(idb::IdbPin* idb_pin, DrcNet* drc_net);
  bool addPortToNet(idb::IdbLayerShape* idb_shape, DrcNet* drc_net);
  // segment via
  bool addWireListToNet(vector<idb::IdbRegularWire*> idb_wire_list, DrcNet* drc_net);
  bool addWireToNet(idb::IdbRegularWire* idb_wire, DrcNet* drc_net);
  bool addIdbSegmentToNet(idb::IdbRegularWireSegment* idb_segment, DrcNet* drc_net);
  bool addViaToNet(idb::IdbRegularWireSegment* idb_segment, DrcNet* drc_net);
  bool addSegmentToNet(idb::IdbRegularWireSegment* idb_segment, DrcNet* drc_net);
  /// basic routing shape wrap function
  void addViaShapeToNet(idb::IdbLayerShape* layer_shape, DrcNet* drc_net, idb::IdbCoordinate<int32_t>* center_point, bool is_cut);
  void addSegmentShapeToNet(int layer_id, int layer_order, idb::IdbCoordinate<int32_t>* start, idb::IdbCoordinate<int32_t>* end,
                            DrcNet* drc_net);
  void addRectShapeToNet(idb::IdbRegularWireSegment* idb_segment, DrcNet* drc_net);
};

}  // namespace idrc
