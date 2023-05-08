#include "DrcIDBWrapper.h"

using namespace std;
namespace idrc {

void DrcIDBWrapper::input(idb::IdbBuilder* idb_builder)
{
  double start, end;
  start = DRCCOMUtil::microtime();

  initIDB(idb_builder);
  wrap();

  end = DRCCOMUtil::microtime();
  std::cout << "[DBWrapper Info] \033[1;32mTotal elapsed time:" << (end - start) << "s \033[0m\n";
}

void DrcIDBWrapper::initTech(idb::IdbBuilder* db_builder)
{
  double start, end;
  start = DRCCOMUtil::microtime();

  if (db_builder == nullptr) {
    initIDBLayout();
    wrapTech();
  } else {
    initIDB(db_builder);
    wrapTech();
  }

  end = DRCCOMUtil::microtime();
  std::cout << "[DrcIDBWrapper Info] \033[1;32mTotal elapsed time:" << (end - start) << "s \033[0m\n";
}

void DrcIDBWrapper::initIDB(idb::IdbBuilder* db_builder)
{
  if (db_builder == nullptr) {
    _db_builder = new idb::IdbBuilder();
    _db_builder->buildLef(get_config()->get_lef_paths());
    _db_builder->buildDef(get_config()->get_def_path());
  } else {
    _db_builder = db_builder;
  }

  if (get_idb_layout() == nullptr || get_idb_design() == nullptr) {
    std::cout << "[IDBWrapper Error] Database is empty!" << std::endl;
    exit(1);
  }
}

void DrcIDBWrapper::wrapDesign()
{
  wrapNetList();
  wrapBlockageList();
  wrapNetPolyList();
}

void DrcIDBWrapper::wrap()
{
  std::cout << "[IDBWrapper Info] build drc db ..." << std::endl;
  wrapRoutingLayerList();
  wrapCutLayerList();
  wrapViaLib();
  wrapNetList();
  wrapBlockageList();
  wrapNetPolyList();
  std::cout << "[IDBWrapper Info] build drc db success ...??" << std::endl;
}

void DrcIDBWrapper::wrapTech()
{
  wrapRoutingLayerList();
  wrapCutLayerList();
  wrapViaLib();
}

void DrcIDBWrapper::initIDBLayout()
{
  _db_builder = new idb::IdbBuilder();
  _db_builder->buildLef(get_config()->get_lef_paths());
  if (get_idb_layout() == nullptr) {
    std::cout << "[IDBWrapper Error] Database is empty!" << std::endl;
    exit(1);
  }
}

/**
 * @brief Create a poly structure for each net for end-of-line and minstep check
 *
 */
void DrcIDBWrapper::wrapNetPolyList()
{
  for (auto& net : get_design()->get_drc_net_list()) {
    initPolyPolygon(net);
    initPolyEdges(net);
    initPolyCorners(net);
    // initPolyMaxRects(net);
  }
}

void DrcIDBWrapper::initPolyEdges(DrcNet* net)
{
  int routing_layer_num = get_tech()->get_drc_routing_layer_list().size();
  // test_eol
  // int routing_layer_num = 1;
  // std::vector<std::set<std::pair<DrcCoordinate<int>, DrcCoordinate<int>>>> polygons_edges(routing_layer_num);
  // Assign the edges of the fused polygon to the edges in the poly
  for (int i = 0; i < routing_layer_num; i++) {
    for (auto& poly : net->get_route_polys(i)) {
      auto polygon = poly->getPolygon();
      initPolyOuterEdges(net, poly.get(), polygon, i);
      // pending
      for (auto holeIt = polygon->begin_holes(); holeIt != polygon->end_holes(); holeIt++) {
        auto& hole_poly = *holeIt;
        initPolyInnerEdges(net, poly.get(), hole_poly, i);
      }
    }
  }
}

void DrcIDBWrapper::initPolyOuterEdges(DrcNet* net, DrcPoly* poly, DrcPolygon* polygon, int layer_id)
{
  DrcCoordinate bp(-1, -1), ep(-1, -1), firstPt(-1, -1);
  BoostPoint bp1, ep1, firstPt1;
  std::vector<std::unique_ptr<DrcEdge>> tmpEdges;
  // skip the first pt
  auto outerIt = polygon->begin();
  bp.set((*outerIt).x(), (*outerIt).y());

  bp1 = *outerIt;
  firstPt.set((*outerIt).x(), (*outerIt).y());
  firstPt1 = *outerIt;
  outerIt++;
  // loop from second to last pt (n-1) edges
  for (; outerIt != polygon->end(); outerIt++) {
    ep.set((*outerIt).x(), (*outerIt).y());
    ep1 = *outerIt;
    // auto edge = make_unique<DrcEdge>();
    std::unique_ptr<DrcEdge> edge(new DrcEdge);
    edge->set_layer_id(layer_id);
    edge->addToPoly(poly);
    edge->addToNet(net);
    // edge->setPoints(bp, ep);
    edge->setSegment(bp1, ep1);
    edge->setDir();
    edge->set_is_fixed(false);
    _region_query->add_routing_edge_to_rtree(layer_id, edge.get());
    if (!tmpEdges.empty()) {
      edge->setPrevEdge(tmpEdges.back().get());
      tmpEdges.back()->setNextEdge(edge.get());
    }
    tmpEdges.push_back(std::move(edge));
    bp.set(ep);
    bp1 = ep1;
  }
  // last edge
  auto edge = std::make_unique<DrcEdge>();
  edge->set_layer_id(layer_id);
  edge->addToPoly(poly);
  edge->addToNet(net);
  // edge->setPoints(bp, firstPt);
  edge->setSegment(bp1, firstPt1);
  edge->setDir();
  edge->set_is_fixed(false);
  edge->setPrevEdge(tmpEdges.back().get());
  tmpEdges.back()->setNextEdge(edge.get());
  // set first edge
  tmpEdges.front()->setPrevEdge(edge.get());
  edge->setNextEdge(tmpEdges.front().get());
  _region_query->add_routing_edge_to_rtree(layer_id, edge.get());

  tmpEdges.push_back(std::move(edge));
  // add to polygon edges
  poly->addEdges(tmpEdges);
}

void DrcIDBWrapper::initPolyInnerEdges(DrcNet* net, DrcPoly* poly, const BoostPolygon& hole_poly, int layer_id)
{
  DrcCoordinate bp(-1, -1), ep(-1, -1), firstPt(-1, -1);
  BoostPoint bp1, ep1, firstPt1;
  vector<unique_ptr<DrcEdge>> tmpEdges;
  // skip the first pt
  auto innerIt = hole_poly.begin();
  bp.set((*innerIt).x(), (*innerIt).y());
  bp1 = *innerIt;
  firstPt.set((*innerIt).x(), (*innerIt).y());
  firstPt1 = *innerIt;
  innerIt++;
  // loop from second to last pt (n-1) edges
  for (; innerIt != hole_poly.end(); innerIt++) {
    ep.set((*innerIt).x(), (*innerIt).y());
    ep1 = *innerIt;
    auto edge = make_unique<DrcEdge>();
    edge->set_layer_id(layer_id);
    edge->addToPoly(poly);
    edge->addToNet(net);
    edge->setDir();
    edge->setSegment(bp1, ep1);
    _region_query->add_routing_edge_to_rtree(layer_id, edge.get());

    edge->set_is_fixed(false);
    if (!tmpEdges.empty()) {
      edge->setPrevEdge(tmpEdges.back().get());
      tmpEdges.back()->setNextEdge(edge.get());
    }
    tmpEdges.push_back(std::move(edge));
    bp.set(ep);
    bp1 = ep1;
  }
  auto edge = make_unique<DrcEdge>();
  edge->set_layer_id(layer_id);
  edge->addToPoly(poly);
  edge->addToNet(net);
  edge->setSegment(bp1, firstPt1);
  edge->setDir();
  edge->set_is_fixed(false);
  _region_query->add_routing_edge_to_rtree(layer_id, edge.get());
  edge->setPrevEdge(tmpEdges.back().get());
  tmpEdges.back()->setNextEdge(edge.get());
  // set first edge
  tmpEdges.front()->setPrevEdge(edge.get());
  edge->setNextEdge(tmpEdges.front().get());

  tmpEdges.push_back(std::move(edge));
  // add to polygon edges
  poly->addEdges(tmpEdges);
}

void DrcIDBWrapper::initPolyCorners(DrcNet* net)
{
  // test
  int numLayers = get_tech()->get_drc_routing_layer_list().size();
  // int numLayers = 1;
  for (int i = 0; i < numLayers; i++) {
    for (auto& poly : net->get_route_polys(i)) {
      initPolyCornerMain(net, poly.get());
    }
  }
}

void DrcIDBWrapper::initPolyCornerMain(DrcNet* net, DrcPoly* poly)
{
  for (auto& edges : poly->getEdges()) {
    std::vector<std::unique_ptr<DrcCorner>> tmpCorners;
    auto prevEdge = edges.back().get();
    // auto layerNum = prevEdge->get_layer_id();
    DrcCorner* pre_corner = nullptr;
    for (int i = 0; i < (int) edges.size(); i++) {
      auto nextEdge = edges[i].get();
      auto uCurrCorner = std::make_unique<DrcCorner>();
      auto currCorner = uCurrCorner.get();
      tmpCorners.push_back(std::move(uCurrCorner));
      // set edge attributes
      prevEdge->setHighCorner(currCorner);
      nextEdge->setLowCorner(currCorner);
      // set currCorner attributes
      currCorner->setPrevEdge(prevEdge);
      currCorner->setNextEdge(nextEdge);
      currCorner->x(prevEdge->high().x());
      currCorner->y(prevEdge->high().y());
      // int orient = bp::orientation(*prevEdge, *nextEdge);
      // if (orient == 1) {
      //   currCorner->setType(frCornerTypeEnum::CONVEX);
      // } else if (orient == -1) {
      //   currCorner->setType(frCornerTypeEnum::CONCAVE);
      // } else {
      //   currCorner->setType(frCornerTypeEnum::UNKNOWN);
      // }

      if ((prevEdge->get_edge_dir() == EdgeDirection::kNorth && nextEdge->get_edge_dir() == EdgeDirection::kWest)
          || (prevEdge->get_edge_dir() == EdgeDirection::kWest && nextEdge->get_edge_dir() == EdgeDirection::kNorth)) {
        currCorner->setDir(CornerDirEnum::kNE);
      } else if ((prevEdge->get_edge_dir() == EdgeDirection::kWest && nextEdge->get_edge_dir() == EdgeDirection::kSouth)
                 || (prevEdge->get_edge_dir() == EdgeDirection::kSouth && nextEdge->get_edge_dir() == EdgeDirection::kWest)) {
        currCorner->setDir(CornerDirEnum::kNW);
      } else if ((prevEdge->get_edge_dir() == EdgeDirection::kSouth && nextEdge->get_edge_dir() == EdgeDirection::kEast)
                 || (prevEdge->get_edge_dir() == EdgeDirection::kEast && nextEdge->get_edge_dir() == EdgeDirection::kSouth)) {
        currCorner->setDir(CornerDirEnum::kSW);
      } else if ((prevEdge->get_edge_dir() == EdgeDirection::kEast && nextEdge->get_edge_dir() == EdgeDirection::kNorth)
                 || (prevEdge->get_edge_dir() == EdgeDirection::kNorth && nextEdge->get_edge_dir() == EdgeDirection::kEast)) {
        currCorner->setDir(CornerDirEnum::kSE);
      }

      // set fixed / route status
      // if (currCorner->getType() == frCornerTypeEnum::CONVEX) {
      //   currCorner->setFixed(false);
      //   for (auto& rect : net->getRectangles(true)[layerNum]) {
      //     if (isCornerOverlap(currCorner, rect)) {
      //       currCorner->setFixed(true);
      //       break;
      //     }
      //   }
      // } else if (currCorner->getType() == frCornerTypeEnum::CONCAVE) {
      //   currCorner->setFixed(true);
      //   auto cornerPt = currCorner->getNextEdge()->low();
      //   for (auto& rect : net->getRectangles(false)[layerNum]) {
      //     if (gtl::contains(rect, cornerPt, true) && !gtl::contains(rect, cornerPt, false)) {
      //       currCorner->setFixed(false);
      //       break;
      //     }
      //   }
      // }
      // currCorner->setFixed(prevEdge->isFixed() && nextEdge->isFixed());

      if (pre_corner) {
        pre_corner->setNextCorner(currCorner);
        currCorner->setPrevCorner(pre_corner);
      }
      pre_corner = currCorner;
      prevEdge = nextEdge;
    }
    // update attributes between first and last corners
    auto currCorner = tmpCorners.front().get();
    pre_corner->setNextCorner(currCorner);
    currCorner->setPrevCorner(pre_corner);
    // add to polygon corners
    poly->addCorners(tmpCorners);
  }
}

// void DrcIDBWrapper::initPolyMaxRects(DrcNet* net)
// {
//   int numLayers = get_tech()->get_drc_routing_layer_list().size();
//   std::vector<set<pair<DrcCoordinate<int>, DrcCoordinate<int>>>> fixedMaxRectangles(numLayers);
//   // // get all fixed max rectangles
//   // initNet_pins_maxRectangles_getFixedMaxRectangles(net, fixedMaxRectangles);

//   // gen all max rectangles
//   std::vector<BoostRect> rects;
//   for (int i = 0; i < numLayers; i++) {
//     for (auto& poly : net->get_route_polys(i)) {
//       rects.clear();
//       bp::get_max_rectangles(rects,*(poly->getPolygon()));
//       for (auto& rect : rects) {
//         initNet_pins_maxRectangles_helper(net, pin.get(), rect, i, fixedMaxRectangles);
//       }
//     }
//   }
// }

/**
 * @brief Assign each layer of net polygon to each poly
 *
 * @param net
 */
void DrcIDBWrapper::initPolyPolygon(DrcNet* net)
{
  int routing_layer_num = get_tech()->get_drc_routing_layer_list().size();
  // test_eol
  // int routing_layer_num = 1;

  std::vector<PolygonSet> layer_routing_polys(routing_layer_num);
  std::vector<PolygonWithHoles> polygons;

  for (int routing_layer_id = 0; routing_layer_id < routing_layer_num; routing_layer_id++) {
    polygons.clear();
    layer_routing_polys[routing_layer_id] = net->get_routing_polygon_set_by_id(routing_layer_id);
    //输出到polys中；
    layer_routing_polys[routing_layer_id].get(polygons);
    for (auto& polygon : polygons) {
      net->addPoly(polygon, routing_layer_id);
    }
  }
}

/**
 * @brief init cut layer list from Idb
 *
 */
void DrcIDBWrapper::wrapCutLayerList()
{
  Tech* tech = get_tech();
  /// idb layers
  idb::IdbLayers* idb_layers = get_idb_layout()->get_layers();
  if (idb_layers == nullptr) {
    std::cout << "[IDBWrapper Error] Layers is empty!" << std::endl;
    exit(1);
  }
  /// tech layers
  std::vector<DrcCutLayer*>& drc_cut_layers = tech->get_drc_cut_layer_list();
  drc_cut_layers.clear();

  /// set value
  for (idb::IdbLayer* idb_layer : idb_layers->get_cut_layers()) {
    /// idb routing layer
    idb::IdbLayerCut* idb_cut_layer = dynamic_cast<idb::IdbLayerCut*>(idb_layer);

    /// tech routing layer
    DrcCutLayer* drc_cut_layer = new DrcCutLayer();
    wrapCutLayer(idb_cut_layer, drc_cut_layer);

    drc_cut_layers.emplace_back(drc_cut_layer);
  }
}

/**
 * @brief initialize the routing layer list
 *
 */
void DrcIDBWrapper::wrapRoutingLayerList()
{
  Tech* tech = get_tech();
  /// idb layers
  idb::IdbLayers* idb_layers = get_idb_layout()->get_layers();
  if (idb_layers == nullptr) {
    std::cout << "[IDBWrapper Error] Layers is empty!" << std::endl;
    exit(1);
  }
  /// tech layers
  std::vector<DrcRoutingLayer*>& drc_routing_layers = tech->get_drc_routing_layer_list();
  drc_routing_layers.clear();

  /// set value
  for (idb::IdbLayer* idb_layer : idb_layers->get_routing_layers()) {
    /// idb routing layer
    idb::IdbLayerRouting* idb_routing_layer = dynamic_cast<idb::IdbLayerRouting*>(idb_layer);

    /// tech routing layer
    DrcRoutingLayer* drc_routing_layer = new DrcRoutingLayer();
    wrapRoutingLayer(idb_routing_layer, drc_routing_layer);

    drc_routing_layers.emplace_back(drc_routing_layer);
  }
}

/**
 * @brief init a cut layer from iDB
 *
 * @param idb_cut_layer
 * @param drc_cut_layer
 */
void DrcIDBWrapper::wrapCutLayer(idb::IdbLayerCut* idb_cut_layer, DrcCutLayer* drc_cut_layer)
{
  // layer value
  drc_cut_layer->set_name(idb_cut_layer->get_name());
  drc_cut_layer->set_layer_id(idb_cut_layer->get_id());
  drc_cut_layer->set_layer_type(LayerType::kCut);
  drc_cut_layer->set_default_width(idb_cut_layer->get_width());
  // drc_cut_layer->set_cut_spacing(idb_cut_layer->get_spacing());

  // cut class
  drc_cut_layer->set_lef58_cut_class_list(idb_cut_layer->get_lef58_cutclass_list());
  // rule
  drc_cut_layer->set_lef58_spacing_table_list(idb_cut_layer->get_lef58_spacing_table());
  // cut eol spacing
  drc_cut_layer->set_lef58_cut_eol_spacing(idb_cut_layer->get_lef58_eol_spacing());
  // enclosure
  drc_cut_layer->set_lef58_enclosure_list(idb_cut_layer->get_lef58_enclosure_list());
  drc_cut_layer->set_lef58_enclosure_edge_list(idb_cut_layer->get_lef58_enclosure_edge_list());

  // rule value
  // std::vector<EnclosureRule*>& enclosure_rule_list = drc_cut_layer->getEnclosureRuleList();
  // std::vector<EnclosureRule*>& below_enclosure_rule_list = drc_cut_layer->getBelowEnclosureRuleList();
  // std::vector<EnclosureRule*>& above_enclosure_rule_list = drc_cut_layer->getAboveEnclosureRuleList();
  // if (idb_cut_layer->get_enclosure_above() != nullptr) {
  //   // EnclosureRule* enclosure_rule_above = new EnclosureRule();
  //   enclosure_rule_above->setRequiredDir(EnclosureDirEnum::Above);
  //   enclosure_rule_above->setOverhang1(idb_cut_layer->get_enclosure_above()->get_overhang_1());
  //   enclosure_rule_above->setOverhang2(idb_cut_layer->get_enclosure_below()->get_overhang_2());
  //   // enclosure_rule_list.push_back(enclosure_rule_above);
  //   // above_enclosure_rule_list.push_back(enclosure_rule_above);
  // }
  // if (idb_cut_layer->get_enclosure_below() != nullptr) {
  //   // EnclosureRule* enclosure_rule_below = new EnclosureRule();
  //   enclosure_rule_below->setRequiredDir(EnclosureDirEnum::Below);
  //   enclosure_rule_below->setOverhang1(idb_cut_layer->get_enclosure_above()->get_overhang_1());
  //   enclosure_rule_below->setOverhang2(idb_cut_layer->get_enclosure_below()->get_overhang_2());
  //   // enclosure_rule_list.push_back(enclosure_rule_below);
  //   // below_enclosure_rule_list.push_back(enclosure_rule_below);
  // }
}

/**
 * @brief transform iDB layer to iDRC layer
 *
 * @param idb_routing_layer
 * @param drc_routing_layer
 */
void DrcIDBWrapper::wrapRoutingLayer(idb::IdbLayerRouting* idb_routing_layer, DrcRoutingLayer* drc_routing_layer)
{
  // layer value
  drc_routing_layer->set_name(idb_routing_layer->get_name());
  drc_routing_layer->set_layer_id(idb_routing_layer->get_id());
  drc_routing_layer->set_layer_type(LayerType::kRouting);
  drc_routing_layer->set_default_width(idb_routing_layer->get_width());
  // direction
  LayerDirection direction = idb_routing_layer->is_horizontal() ? LayerDirection::kHorizontal : LayerDirection::kVertical;
  drc_routing_layer->set_direction(direction);
  // rule value
  drc_routing_layer->set_min_width(idb_routing_layer->get_min_width());
  drc_routing_layer->set_min_area(idb_routing_layer->get_area());
  if (idb_routing_layer->get_min_enclose_area_list()) {
    if (idb_routing_layer->get_min_enclose_area_list()->get_min_area_list().size()) {
      drc_routing_layer->set_min_enclosed_area(idb_routing_layer->get_min_enclose_area_list()->get_min_area_list()[0]._area);
    }
  }

  // drc_routing_layer->set_min_enclosed_area(idb_routing_layer->get_min_enclose_area_list());
  // min enclosed area
  // int minEnclosedArea = (idb_routing_layer->get_min_enclose_area_list()->get_min_area_list()).front()._area;
  // drc_routing_layer->set_min_enclosed_area(minEnclosedArea);
  /// spacing
  std::vector<SpacingRangeRule*>& spacing_range_rule_list = drc_routing_layer->get_spacing_range_rule_list();
  idb::IdbLayerSpacingList* idb_spacing_list = idb_routing_layer->get_spacing_list();
  if (idb_spacing_list != nullptr) {
    for (idb::IdbLayerSpacing* idb_spacing : idb_spacing_list->get_spacing_list()) {
      if (idb_spacing->isDefault()) {
        /// make sure the first spacing is default value
        drc_routing_layer->set_min_spacing(idb_spacing->get_min_spacing());
      } else {
        SpacingRangeRule* spacing_range_rule = new SpacingRangeRule();
        spacing_range_rule->set_spacing(idb_spacing->get_min_spacing());
        spacing_range_rule->set_max_width(idb_spacing->get_max_width());
        spacing_range_rule->set_min_width(idb_spacing->get_min_width());
        spacing_range_rule_list.push_back(spacing_range_rule);
      }
    }
  }
  // SpacingTable
  drc_routing_layer->set_spacing_table(idb_routing_layer->get_spacing_table());
  // notch
  drc_routing_layer->set_notch_spacing_rule(idb_routing_layer->get_spacing_notchlength());
  drc_routing_layer->set_lef58_notch_spacing_rule(idb_routing_layer->get_lef58_spacing_notchlength());
  // jog
  drc_routing_layer->set_lef58_jog_spacing_rule(idb_routing_layer->get_lef58_spacingtable_jogtojog());
  // corner_fill
  drc_routing_layer->set_lef58_corner_fill_spacing_rule(idb_routing_layer->get_lef58_corner_fill_spacing());
  // minstep
  drc_routing_layer->set_lef58_min_step_rule(idb_routing_layer->get_lef58_min_step());
  drc_routing_layer->set_min_step_rule(idb_routing_layer->get_min_step());
  // area
  drc_routing_layer->set_lef58_area_rule_list(idb_routing_layer->get_lef58_area());

  // minimumcut
  // std::vector<MinimumCutRule*>& minimum_cut_rule_list = drc_routing_layer->getMinimumCutRuleList();
  // MinimumCutRule* minimum_cut_rule = new MinimumCutRule();
  if (idb_routing_layer->get_min_cut_num() != 0) {
    // minimum_cut_rule->setNumCuts(idb_routing_layer->get_min_cut_num());
    // minimum_cut_rule->setWidth(idb_routing_layer->get_min_cut_width());
    // minimum_cut_rule_list.push_back(minimum_cut_rule);
  }
  // rule
  drc_routing_layer->set_lef58_eol_spacing_rule_list(idb_routing_layer->get_lef58_spacing_eol_list());
}

/**
 * @brief initialize all via list in tech lef
 *
 */
void DrcIDBWrapper::wrapViaLib()
{
  Tech* tech = get_tech();
  /// idb via list in tech lef
  idb::IdbVias* idb_Via_list = get_idb_layout()->get_via_list();
  if (idb_Via_list == nullptr) {
    std::cout << "[IDBWrapper Error] Via list in tech lef is empty!" << std::endl;
    exit(1);
  }
  /// design via list
  std::vector<DrcVia*>& via_list = tech->get_via_lib();
  via_list.clear();
  via_list.reserve(idb_Via_list->get_num_via());

  for (idb::IdbVia* idb_via : idb_Via_list->get_via_list()) {
    wrapVia(idb_via, via_list);
  }
}

/**
 * @brief wrap iDB via to iDRC via list
 *
 * @param idb_via
 * @param via_list
 */
void DrcIDBWrapper::wrapVia(idb::IdbVia* idb_via, std::vector<DrcVia*>& via_list)
{
  if (idb_via == nullptr) {
    std::cout << "[IDBWrapper Error] via is empty!" << std::endl;
    exit(1);
  }

  DrcVia* via = new DrcVia();

  via->set_via_name(idb_via->get_name());
  /// set index
  via->set_via_idx(via_list.size());

  // set layer
  idb::IdbViaMaster* via_master = idb_via->get_instance();

  /// top
  idb::IdbLayerShape* idb_shape_top = via_master->get_top_layer_shape();
  idb::IdbLayerRouting* idb_layer_top = dynamic_cast<idb::IdbLayerRouting*>(idb_shape_top->get_layer());
  idb::IdbRect idb_boudingbox_top = idb_shape_top->get_bounding_box();

  DrcEnclosure top_enclosure;
  // layer is different to idb
  top_enclosure.set_layer_idx(idb_layer_top->get_id());

  DrcRectangle<int> rect_top;
  wrapRect(rect_top, &idb_boudingbox_top);
  top_enclosure.set_shape(rect_top);
  via->set_top_enclosure(top_enclosure);

  /// bottom
  idb::IdbLayerShape* idb_shape_bottom = via_master->get_bottom_layer_shape();
  idb::IdbLayerRouting* idb_layer_bottom = dynamic_cast<idb::IdbLayerRouting*>(idb_shape_bottom->get_layer());
  idb::IdbRect idb_boudingbox_bottom = idb_shape_bottom->get_bounding_box();

  DrcEnclosure bottom_enclosure;
  // layer is different to idb
  bottom_enclosure.set_layer_idx(idb_layer_bottom->get_id());

  DrcRectangle<int> rect_bottom;
  wrapRect(rect_bottom, &idb_boudingbox_bottom);
  bottom_enclosure.set_shape(rect_bottom);
  via->set_bottom_enclosure(bottom_enclosure);

  /// cut
  //   idb::IdbLayerShape* shape_cut = via_master->get_cut_layer_shape();

  /// tbd
  /// void set_center_coord(const Coordinate<int>& center_coord);

  via_list.emplace_back(via);
}

// DrcNet* DrcIDBWrapper::get_drc_net(int netId)
// {
//   DrcNet* drc_net = nullptr;
//   auto it = _id_to_net.find(netId);
//   if (it == _id_to_net.end()) {
//     DrcDesign* drc_design = get_design();
//     drc_net = drc_design->add_drc_net();
//     drc_net->set_net_id(netId);
//     _id_to_net[netId] = drc_net;
//   } else {
//     drc_net = it->second;
//   }
//   return drc_net;
// }

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
/////////
// wrap rect

void DrcIDBWrapper::wrapRect(DrcRect* drc_rect, idb::IdbRect* idb_rect)
{
  if (idb_rect == nullptr) {
    std::cout << "[IDBWrapper error:idb_rect is null]" << std::endl;
    return;
  }

  drc_rect->set_lb(idb_rect->get_low_x(), idb_rect->get_low_y());
  drc_rect->set_rt(idb_rect->get_high_x(), idb_rect->get_high_y());
}
BoostRect DrcIDBWrapper::getBoostRectFromIdbRect(idb::IdbRect* idb_rect)
{
  if (idb_rect == nullptr) {
    std::cout << "[IDBWrapper error:idb_rect is null]" << std::endl;
    return BoostRect(0, 0, 0, 0);
  }
  return BoostRect(idb_rect->get_low_x(), idb_rect->get_low_y(), idb_rect->get_high_x(), idb_rect->get_high_y());
}
// basic blockage wrap function
void DrcIDBWrapper::wrapBlockageFromLayerShape(idb::IdbLayerShape* layer_shape, DrcDesign* design)
{
  int layer_id = layer_shape->get_layer()->get_id();
  for (idb::IdbRect* rect : layer_shape->get_rect_list()) {
    DrcRect* drc_rect = new DrcRect();
    drc_rect->set_owner_type(RectOwnerType::kBlockage);
    drc_rect->set_is_fixed(true);
    drc_rect->set_layer_id(layer_id);

    wrapRect(drc_rect, rect);
    BoostRect boost_rect = getBoostRectFromIdbRect(rect);

    design->add_blockage(layer_id, drc_rect);
    design->add_blockage(layer_id, boost_rect);
    _region_query->add_fixed_rect_to_rtree(layer_id, drc_rect);
  }
}
// blockage
void DrcIDBWrapper::wrapBlockageList()
{
  wrapBlockageListInDef();
  wrapInstanceListBlockage();
  wrapSpecialNetListBlockage();
}
////////////////////////// blockage list in def
/////////////////////////
void DrcIDBWrapper::wrapBlockageListInDef()
{
  DrcDesign* design = get_design();
  /// iDB blockage list
  idb::IdbBlockageList* idb_blockages = get_idb_design()->get_blockage_list();
  if (idb_blockages == nullptr) {
    std::cout << "[IDBWrapper Error] Blockage list is empty!" << std::endl;
    return;
  }

  /// set value
  for (idb::IdbBlockage* idb_blockage : idb_blockages->get_blockage_list()) {
    addIdbBlockToDrcDesign(idb_blockage, design);
  }
}

void DrcIDBWrapper::addIdbBlockToDrcDesign(idb::IdbBlockage* idb_blockage, DrcDesign* design)
{
  if (idb_blockage == nullptr) {
    std::cout << "[IDBWrapper Error] Blockage is empty!" << std::endl;
    assert(false);
    return;
  }

  if (idb_blockage && idb_blockage->is_palcement_blockage()) {
    idb::IdbPlacementBlockage* idb_placement_blockage = dynamic_cast<idb::IdbPlacementBlockage*>(idb_blockage);
    addPlacementBlockage(idb_placement_blockage, design);
  } else {
    idb::IdbRoutingBlockage* idb_routing_blockage = dynamic_cast<idb::IdbRoutingBlockage*>(idb_blockage);
    addRoutingBlockage(idb_routing_blockage, design);
  }
}

void DrcIDBWrapper::addPlacementBlockage(idb::IdbPlacementBlockage* idb_blockage, DrcDesign* design)
{
  // do nothing
}
void DrcIDBWrapper::addRoutingBlockage(idb::IdbRoutingBlockage* idb_blockage, DrcDesign* design)
{
  int layer_id = idb_blockage->get_layer()->get_id();
  for (idb::IdbRect* idb_rect : idb_blockage->get_rect_list()) {
    DrcRect* drc_rect = new DrcRect();
    drc_rect->set_owner_type(RectOwnerType::kBlockage);
    drc_rect->set_is_fixed(true);
    drc_rect->set_layer_id(layer_id);
    drc_rect->set_net_id(-1);

    wrapRect(drc_rect, idb_rect);
    BoostRect boost_rect = getBoostRectFromIdbRect(idb_rect);

    design->add_blockage(layer_id, drc_rect);
    design->add_blockage(layer_id, boost_rect);
    _region_query->add_fixed_rect_to_rtree(layer_id, drc_rect);
  }
}

////////////////////////// blockage as blockage
/////////////////////////
void DrcIDBWrapper::wrapInstanceListBlockage()
{
  DrcDesign* design = get_design();
  // idb instance list manager ptr
  idb::IdbInstanceList* idb_instance_list_ptr = get_idb_design()->get_instance_list();
  if (idb_instance_list_ptr == nullptr) {
    std::cout << "[IDBWrapper Error] Instances is empty!" << std::endl;
    assert(false);
    return;
  }

  // idb instance list
  if (idb_instance_list_ptr) {
    std::vector<idb::IdbInstance*>& idb_instance_list = idb_instance_list_ptr->get_instance_list();
    for (size_t i = 0; i < idb_instance_list.size(); i++) {
      idb::IdbInstance* idb_instance = idb_instance_list[i];

      // instance obs
      addInstanceBlockage(idb_instance, design);
    }
  }
}
void DrcIDBWrapper::addInstanceBlockage(idb::IdbInstance* idb_instance, DrcDesign* design)
{
  // instance obs
  addInstanceObsBlockage(idb_instance, design);
  // instance pin
  addInstancePinBlockage(idb_instance, design);
}
void DrcIDBWrapper::addInstanceObsBlockage(idb::IdbInstance* idb_instance, DrcDesign* design)
{
  for (idb::IdbLayerShape* layer_shape : idb_instance->get_obs_box_list()) {
    wrapBlockageFromLayerShape(layer_shape, design);
  }
}
void DrcIDBWrapper::addInstancePinBlockage(idb::IdbInstance* idb_instance, DrcDesign* design)
{
  // instance pin
  for (idb::IdbPin* idb_pin : idb_instance->get_pin_list()->get_pin_list()) {
    /// only save pins as blockages that DO NOT connect to any net
    if (idb_pin->get_net() != nullptr) {
      continue;
    }

    for (idb::IdbLayerShape* layer_shape : idb_pin->get_port_box_list()) {
      wrapBlockageFromLayerShape(layer_shape, design);
    }
  }
}
////////////////////////// specialnets as blockage
/////////////////////////
void DrcIDBWrapper::wrapSpecialNetListBlockage()
{
  DrcDesign* design = get_design();

  // special net
  idb::IdbSpecialNetList* idb_snet_list = get_idb_design()->get_special_net_list();
  for (idb::IdbSpecialNet* idb_net : idb_snet_list->get_net_list()) {
    addSpecialNetBlockage(idb_net, design);
  }
}

void DrcIDBWrapper::addSpecialNetBlockage(idb::IdbSpecialNet* idb_net, DrcDesign* design)
{
  for (idb::IdbSpecialWire* idb_wire : idb_net->get_wire_list()->get_wire_list()) {
    for (idb::IdbSpecialWireSegment* idb_segment : idb_wire->get_segment_list()) {
      if (idb_segment->is_via()) {
        /// via
        addSpecialNetViaBlockage(idb_segment, design);
      } else {
        // wire
        addSpecialNetWireBlockage(idb_segment, design);
      }
    }
  }
}
void DrcIDBWrapper::addSpecialNetViaBlockage(idb::IdbSpecialWireSegment* idb_segment, DrcDesign* design)
{
  // get via
  idb::IdbVia* idb_via = idb_segment->get_via();
  // get top layer

  idb::IdbLayerShape idb_via_shape_top = idb_via->get_top_layer_shape();
  wrapBlockageFromLayerShape(&idb_via_shape_top, design);

  // get bottom layer

  idb::IdbLayerShape idb_via_shape_bottom = idb_via->get_bottom_layer_shape();
  wrapBlockageFromLayerShape(&idb_via_shape_bottom, design);
}
void DrcIDBWrapper::addSpecialNetWireBlockage(idb::IdbSpecialWireSegment* idb_segment, DrcDesign* design)
{
  int layer_id = idb_segment->get_layer()->get_id();
  idb::IdbRect* idb_rect = idb_segment->get_bounding_box();
  DrcRect* drc_rect = new DrcRect();
  drc_rect->set_owner_type(RectOwnerType::kBlockage);
  drc_rect->set_is_fixed(true);
  drc_rect->set_layer_id(layer_id);

  wrapRect(drc_rect, idb_rect);
  BoostRect boost_rect = getBoostRectFromIdbRect(idb_rect);

  design->add_blockage(layer_id, drc_rect);
  design->add_blockage(layer_id, boost_rect);
  _region_query->add_fixed_rect_to_rtree(layer_id, drc_rect);
}

////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
////////net
void DrcIDBWrapper::wrapNetList()
{
  DrcDesign* design = get_design();
  // idb net list manager ptr
  idb::IdbNetList* idb_net_list_ptr = get_idb_design()->get_net_list();
  if (idb_net_list_ptr == nullptr) {
    std::cout << "[IDBWrapper Error] Net list is empty!" << std::endl;
    assert(false);
    return;
  }

  /// design net list
  design->clear_drc_net_list();

  // int lower_bound_value = 195000;
  // int upper_bound_value = 200000;

  /// warp net list
  // int number = 0;
  int index = 0;
  if (idb_net_list_ptr) {
    for (idb::IdbNet* idb_net : idb_net_list_ptr->get_net_list()) {
      if (index++ % 1000 == 0) {
        std::cout << "-" << std::flush;
      }
      if (index++ % 100000 == 0) {
        std::cout << std::endl;
      }
      // number++;
      // if (number < lower_bound_value) {
      //   continue;
      // }
      // if (number > upper_bound_value) {
      //   return;
      // }

      // if (idb_net->get_net_name() == "u0_soc_top/u0_ysyx_210232/core/Ex/alu/_0754_") {
      //   std::cout << idb_net->get_net_name() << std::endl;
      // } else {
      //   continue;
      // }
      // ///////////////////////////////
      // // Specialization processing //
      // if (idb_net->get_io_pin() != nullptr) {
      //   continue;
      // }
      // bool has_io_cell = false;
      // for (idb::IdbInstance* instance : idb_net->get_instance_list()->get_instance_list()) {
      //   if (instance->is_io_instance()) {
      //     has_io_cell = true;
      //     break;
      //   }
      // }
      // if (has_io_cell) {
      //   continue;
      // }
      // // End Specialization processing //
      // ///////////////////////////////

      addNetToDrcDesign(idb_net, design);
    }
  }
  std::cout << std::endl;
}

bool DrcIDBWrapper::addNetToDrcDesign(idb::IdbNet* idb_net, DrcDesign* design)
{
  if (idb_net == nullptr) {
    return false;
  }
  if (nullptr != idb_net->get_io_pin()) {
    return false;
  }
  if (idb_net->get_pin_number() < 2) {
    return false;
  }
  DrcNet* drc_net = design->add_drc_net();
  drc_net->set_net_id(design->get_net_num());

  // wrap pin list
  idb::IdbPins* idb_pin_list_ptr = idb_net->get_instance_pin_list();
  addPinListToNet(idb_pin_list_ptr->get_pin_list(), drc_net);
  // wrap segment and via
  idb::IdbRegularWireList* idb_wire_list_ptr = idb_net->get_wire_list();
  addWireListToNet(idb_wire_list_ptr->get_wire_list(), drc_net);
  return true;
}
/////////pin shape
bool DrcIDBWrapper::addPinListToNet(vector<idb::IdbPin*>& idb_pin_list, DrcNet* drc_net)
{
  for (idb::IdbPin* idb_pin : idb_pin_list) {
    addPinToNet(idb_pin, drc_net);
  }
  return true;
}

bool DrcIDBWrapper::addPinToNet(idb::IdbPin* idb_pin, DrcNet* drc_net)
{
  for (auto& idb_port : idb_pin->get_port_box_list()) {
    addPortToNet(idb_port, drc_net);
  }
  return true;
}

bool DrcIDBWrapper::addPortToNet(idb::IdbLayerShape* idb_shape, DrcNet* drc_net)
{
  if (idb_shape == nullptr) {
    return false;
  }
  /// set layer id
  int layer_id = idb_shape->get_layer()->get_id();
  /// wrap rect shape list
  for (auto& idb_rect : idb_shape->get_rect_list()) {
    DrcRect* drc_rect = new DrcRect();
    drc_rect->set_net_id(drc_net->get_net_id());
    drc_rect->set_owner_type(RectOwnerType::kPin);
    drc_rect->set_is_fixed(true);
    drc_rect->set_layer_id(layer_id);

    wrapRect(drc_rect, idb_rect);
    BoostRect boost_rect = getBoostRectFromIdbRect(idb_rect);

    drc_net->add_pin_rect(layer_id, drc_rect);
    drc_net->add_pin_rect(layer_id, boost_rect);
    auto boost_box = DRCUtil::getBoostRect(drc_rect);
    drc_net->add_routing_rect(layer_id, boost_box);
    _region_query->add_fixed_rect_to_rtree(layer_id, drc_rect);
  }
  return true;
}

// segment and via
bool DrcIDBWrapper::addWireListToNet(vector<idb::IdbRegularWire*> idb_wire_list, DrcNet* drc_net)
{
  for (auto& idb_wire : idb_wire_list) {
    addWireToNet(idb_wire, drc_net);
  }
  return true;
}
bool DrcIDBWrapper::addWireToNet(idb::IdbRegularWire* idb_wire, DrcNet* drc_net)
{
  if (idb_wire == nullptr) {
    return false;
  }
  for (auto& idb_segemnt : idb_wire->get_segment_list()) {
    addIdbSegmentToNet(idb_segemnt, drc_net);
  }
  return true;
}
bool DrcIDBWrapper::addIdbSegmentToNet(idb::IdbRegularWireSegment* idb_segment, DrcNet* drc_net)
{
  if (idb_segment == nullptr) {
    return false;
  }
  if (idb_segment->is_via()) {
    // via
    addViaToNet(idb_segment, drc_net);
  } else if (idb_segment->is_rect()) {
    addRectShapeToNet(idb_segment, drc_net);
  } else if (idb_segment->get_point_number() == 2) {
    // segment
    addSegmentToNet(idb_segment, drc_net);
  } else if (idb_segment->get_point_number() != 2) {
    std::cout << "idb segment point num :: " << idb_segment->get_point_number() << std::endl;
  }
  return true;
}

// rect
void DrcIDBWrapper::addRectShapeToNet(idb::IdbRegularWireSegment* idb_segment, DrcNet* drc_net)
{
  int layer_id = idb_segment->get_layer()->get_id();
  auto delta_rect = idb_segment->get_delta_rect();
  auto end_point = idb_segment->get_point_end();

  int lb_x = end_point->get_x() + delta_rect->get_low_x();
  int lb_y = end_point->get_y() + delta_rect->get_low_y();
  int rt_x = end_point->get_x() + delta_rect->get_high_x();
  int rt_y = end_point->get_y() + delta_rect->get_high_y();

  DrcRect* drc_rect = new DrcRect();

  drc_rect->set_net_id(drc_net->get_net_id());
  drc_rect->set_owner_type(RectOwnerType::kSegment);
  drc_rect->set_is_fixed(false);
  drc_rect->set_layer_id(layer_id);

  drc_rect->set_coordinate(lb_x, lb_y, rt_x, rt_y);
  BoostRect boost_rect(lb_x, lb_y, rt_x, rt_y);

  drc_net->add_routing_rect(layer_id, drc_rect);
  drc_net->add_routing_rect(layer_id, boost_rect);
  _region_query->add_routing_rect_to_rtree(layer_id, drc_rect);
}

// via
bool DrcIDBWrapper::addViaToNet(idb::IdbRegularWireSegment* idb_segment, DrcNet* drc_net)
{
  // int layer_id = idb_segment->get_layer()->get_id();
  vector<idb::IdbVia*> idb_via_list = idb_segment->get_via_list();
  if (idb_via_list.size() > 1) {
    std::cout << "idb via list size :: " << idb_via_list.size() << std::endl;
  }
  idb::IdbVia* idb_via = idb_via_list.back();
  idb::IdbLayerShape* bottom_shape = idb_via->get_instance()->get_bottom_layer_shape();
  idb::IdbLayerShape* top_shape = idb_via->get_instance()->get_top_layer_shape();
  idb::IdbLayerShape* cut_shape = idb_via->get_instance()->get_cut_layer_shape();
  idb::IdbCoordinate<int32_t>* center_point = idb_via->get_coordinate();
  addViaShapeToNet(bottom_shape, drc_net, center_point, false);
  addViaShapeToNet(top_shape, drc_net, center_point, false);
  addViaShapeToNet(cut_shape, drc_net, center_point, true);
  // if (bottom_shape->get_layer()->get_id() == 1) {
  //   addViaShapeToNet(bottom_shape, drc_net, center_point);
  //   addViaShapeToNet(top_shape, drc_net, center_point);
  // } else {
  //   if (layer_id == bottom_shape->get_layer()->get_id()) {
  //     addViaShapeToNet(bottom_shape, drc_net, center_point);
  //   } else if (layer_id == top_shape->get_layer()->get_id()) {
  //     addViaShapeToNet(top_shape, drc_net, center_point);
  //   }
  // }
  return true;
}
// segment
bool DrcIDBWrapper::addSegmentToNet(idb::IdbRegularWireSegment* idb_segment, DrcNet* drc_net)
{
  int layer_id = idb_segment->get_layer()->get_id();
  idb::IdbCoordinate<int32_t>* start = idb_segment->get_point_start();
  idb::IdbCoordinate<int32_t>* end = idb_segment->get_point_end();
  addSegmentShapeToNet(layer_id, start, end, drc_net);
  return true;
}

/// basic routing shape wrap function
void DrcIDBWrapper::addViaShapeToNet(idb::IdbLayerShape* layer_shape, DrcNet* drc_net, idb::IdbCoordinate<int32_t>* center_point,
                                     bool is_cut)
{
  int layer_id = layer_shape->get_layer()->get_id();
  // idb::IdbRect* idb_rect = (layer_shape->get_rect_list()).back();
  for (auto& idb_rect : layer_shape->get_rect_list()) {
    int lb_x = idb_rect->get_low_x() + center_point->get_x();
    int lb_y = idb_rect->get_low_y() + center_point->get_y();
    int rt_x = idb_rect->get_high_x() + center_point->get_x();
    int rt_y = idb_rect->get_high_y() + center_point->get_y();

    // if (layer_id == 2 && lb_x == 2277200 && lb_y == 2048260 && rt_x == 2277490 && rt_y == 2048460) {
    //   std::cout << "find via rect !!!!!!!!" << std::endl;
    // }
    // if (layer_id == 1 && lb_x == 2276835 && lb_y == 2048215 && rt_x == 2277035 && rt_y == 2048505) {
    //   std::cout << "find via rect !!!!!!!!" << std::endl;
    // }
    DrcRect* drc_rect = new DrcRect();
    drc_rect->set_net_id(drc_net->get_net_id());
    drc_rect->set_owner_type(RectOwnerType::kViaMetal);
    drc_rect->set_is_fixed(false);
    drc_rect->set_layer_id(layer_id);

    drc_rect->set_coordinate(lb_x, lb_y, rt_x, rt_y);
    BoostRect boost_rect(lb_x, lb_y, rt_x, rt_y);
    if (is_cut) {
      drc_rect->set_owner_type(RectOwnerType::kViaCut);
      // auto& cut_class_list = _tech->get_drc_cut_layer_list()[layer_id]->get_lef58_cut_class_list();
      // drc_rect->set_cut_class(cut_class_list);
      drc_net->add_cut_rect(layer_id, drc_rect);
      _region_query->add_cut_rect_to_rtree(layer_id, drc_rect);
    } else {
      drc_rect->set_owner_type(RectOwnerType::kViaMetal);
      drc_net->add_routing_rect(layer_id, drc_rect);
      drc_net->add_routing_rect(layer_id, boost_rect);
      _region_query->add_routing_rect_to_rtree(layer_id, drc_rect);
    }
  }
}

void DrcIDBWrapper::addSegmentShapeToNet(int layer_id, idb::IdbCoordinate<int32_t>* start, idb::IdbCoordinate<int32_t>* end,
                                         DrcNet* drc_net)
{
  int width = _tech->getRoutingWidth(layer_id);
  int lb_x = std::min(start->get_x(), end->get_x()) - width / 2;
  int lb_y = std::min(start->get_y(), end->get_y()) - width / 2;
  int rt_x = std::max(start->get_x(), end->get_x()) + width / 2;
  int rt_y = std::max(start->get_y(), end->get_y()) + width / 2;

  DrcRect* drc_rect = new DrcRect();

  drc_rect->set_net_id(drc_net->get_net_id());
  drc_rect->set_owner_type(RectOwnerType::kSegment);
  drc_rect->set_is_fixed(false);
  drc_rect->set_layer_id(layer_id);

  drc_rect->set_coordinate(lb_x, lb_y, rt_x, rt_y);
  BoostRect boost_rect(lb_x, lb_y, rt_x, rt_y);

  drc_net->add_routing_rect(layer_id, drc_rect);
  drc_net->add_routing_rect(layer_id, boost_rect);
  _region_query->add_routing_rect_to_rtree(layer_id, drc_rect);
}

}  // namespace idrc
