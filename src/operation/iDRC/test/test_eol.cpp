#include <string>

#include "DRC.h"
#include "DrcAPI.hpp"
#include "DrcIDBWrapper.h"
#include "DrcRect.h"
#include "EOLSpacingCheck.hpp"
#include "SpotParser.h"

using namespace idrc;

void runTestCase1()
{
  RegionQuery* rq = RegionQuery::getInstance();
  DrcIDBWrapper* idb_wrapper = new DrcIDBWrapper(rq);
  //需要线网，tech层数
  DrcNet* drc_net = new DrcNet();
  DrcRect* drc_rect1 = new DrcRect();
  drc_rect1->set_net_id(1);
  drc_rect1->set_owner_type(RectOwnerType::kSegment);
  drc_rect1->set_is_fixed(false);
  drc_rect1->set_layer_id(0);
  drc_rect1->set_coordinate(1, 1, 2, 5);
  drc_net->add_routing_rect(0, drc_rect1);
  BoostRect boost_rect(1, 1, 2, 5);
  drc_net->add_routing_rect(0, boost_rect);
  //   drc_net->add_routing_rect(0, boost_rect);
  rq->add_routing_rect_to_rtree(0, drc_rect1);

  // DrcRect* drc_rect2 = new DrcRect();
  // drc_rect2->set_net_id(1);
  // drc_rect2->set_owner_type(RectOwnerType::kSegment);
  // drc_rect2->set_is_fixed(false);
  // drc_rect2->set_layer_id(0);
  // drc_rect2->set_coordinate(3, 3, 6, 6);
  // drc_net->add_routing_rect(0, drc_rect2);
  // BoostRect boost_rect1(3, 3, 6, 6);
  // drc_net->add_routing_rect(0, boost_rect1);

  DrcRect* drc_rect3 = new DrcRect();
  drc_rect3->set_net_id(1);
  drc_rect3->set_owner_type(RectOwnerType::kSegment);
  drc_rect3->set_is_fixed(false);
  drc_rect3->set_layer_id(0);
  drc_rect3->set_coordinate(1, 6, 2, 7);
  drc_net->add_routing_rect(0, drc_rect3);
  BoostRect boost_rect2(1, 6, 2, 7);
  drc_net->add_routing_rect(0, boost_rect2);
  rq->add_routing_rect_to_rtree(0, drc_rect3);

  idb_wrapper->initPolyPolygon(drc_net);

  idb_wrapper->initPolyEdges(drc_net);

  std::vector<std::unique_ptr<DrcPoly>>& poly = drc_net->get_route_polys(0);
  std::cout << poly.size() << std::endl;

  std::vector<std::vector<std::unique_ptr<DrcEdge>>>& ss = poly[0]->getEdges();
  std::cout << ss.size() << std::endl;

  for (int j = 0; j < ss.size(); j++) {
    for (int i = 0; i < ss[j].size(); i++) {
      std::cout << ss[j][i]->get_begin_x() << "   " << ss[j][i]->get_begin_y() << std::endl;
      std::cout << ss[j][i]->get_end_x() << "   " << ss[j][i]->get_end_y() << std::endl;
    }
  }

  auto eol_spacing_check = EOLSpacingCheck::getInstance();
  auto& rules = eol_spacing_check->get_lef58_eol_spacing_rule_list();
  idb::routinglayer::Lef58SpacingEol rule;
  rule.set_eol_space(2);
  rule.set_eol_width(2);
  rule.set_eol_within(1);
  rules.push_back(make_shared<idb::routinglayer::Lef58SpacingEol>(rule));

  eol_spacing_check->checkEOLSpacing(drc_net);
}

void runTestCase2()
{
  RegionQuery* rq = RegionQuery::getInstance();
  DrcIDBWrapper* idb_wrapper = new DrcIDBWrapper(rq);
  //需要线网，tech层数
  DrcNet* drc_net = new DrcNet();
  DrcRect* drc_rect1 = new DrcRect();
  drc_rect1->set_net_id(1);
  drc_rect1->set_owner_type(RectOwnerType::kSegment);
  drc_rect1->set_is_fixed(false);
  drc_rect1->set_layer_id(0);
  drc_rect1->set_coordinate(1, 1, 3, 5);
  drc_net->add_routing_rect(0, drc_rect1);
  BoostRect boost_rect(1, 1, 3, 5);
  drc_net->add_routing_rect(0, boost_rect);
  //   drc_net->add_routing_rect(0, boost_rect);
  rq->add_routing_rect_to_rtree(0, drc_rect1);

  // DrcRect* drc_rect2 = new DrcRect();
  // drc_rect2->set_net_id(1);
  // drc_rect2->set_owner_type(RectOwnerType::kSegment);
  // drc_rect2->set_is_fixed(false);
  // drc_rect2->set_layer_id(0);
  // drc_rect2->set_coordinate(3, 3, 6, 6);
  // drc_net->add_routing_rect(0, drc_rect2);
  // BoostRect boost_rect1(3, 3, 6, 6);
  // drc_net->add_routing_rect(0, boost_rect1);

  DrcRect* drc_rect3 = new DrcRect();
  drc_rect3->set_net_id(1);
  drc_rect3->set_owner_type(RectOwnerType::kSegment);
  drc_rect3->set_is_fixed(false);
  drc_rect3->set_layer_id(0);
  drc_rect3->set_coordinate(2, 4, 6, 7);
  drc_net->add_routing_rect(0, drc_rect3);
  BoostRect boost_rect2(2, 4, 6, 7);
  drc_net->add_routing_rect(0, boost_rect2);
  rq->add_routing_rect_to_rtree(0, drc_rect3);

  idb_wrapper->initPolyPolygon(drc_net);

  idb_wrapper->initPolyEdges(drc_net);

  std::vector<std::unique_ptr<DrcPoly>>& poly = drc_net->get_route_polys(0);
  std::cout << poly.size() << std::endl;

  std::vector<std::vector<std::unique_ptr<DrcEdge>>>& ss = poly[0]->getEdges();
  std::cout << ss.size() << std::endl;

  for (int j = 0; j < ss.size(); j++) {
    for (int i = 0; i < ss[j].size(); i++) {
      std::cout << ss[j][i]->get_begin_x() << "   " << ss[j][i]->get_begin_y() << std::endl;
      std::cout << ss[j][i]->get_end_x() << "   " << ss[j][i]->get_end_y() << std::endl;
    }
  }

  auto eol_spacing_check = EOLSpacingCheck::getInstance();
  auto& rules = eol_spacing_check->get_lef58_eol_spacing_rule_list();
  idb::routinglayer::Lef58SpacingEol rule;
  rule.set_eol_space(3);
  rule.set_eol_width(3);
  rule.set_eol_within(1);
  rules.push_back(make_shared<idb::routinglayer::Lef58SpacingEol>(rule));

  eol_spacing_check->checkEOLSpacing(drc_net);
}

// adj
void runTestCase3()
{
  RegionQuery* rq = RegionQuery::getInstance();
  DrcIDBWrapper* idb_wrapper = new DrcIDBWrapper(rq);
  //需要线网，tech层数
  DrcNet* drc_net = new DrcNet();
  DrcRect* drc_rect1 = new DrcRect();
  drc_rect1->set_net_id(1);
  drc_rect1->set_owner_type(RectOwnerType::kSegment);
  drc_rect1->set_is_fixed(false);
  drc_rect1->set_layer_id(0);
  drc_rect1->set_coordinate(1, 1, 2, 5);
  drc_net->add_routing_rect(0, drc_rect1);
  BoostRect boost_rect(1, 1, 2, 5);
  drc_net->add_routing_rect(0, boost_rect);
  //   drc_net->add_routing_rect(0, boost_rect);
  rq->add_routing_rect_to_rtree(0, drc_rect1);

  DrcRect* drc_rect2 = new DrcRect();
  drc_rect2->set_net_id(1);
  drc_rect2->set_owner_type(RectOwnerType::kSegment);
  drc_rect2->set_is_fixed(false);
  drc_rect2->set_layer_id(0);
  drc_rect2->set_coordinate(3, 3, 9, 4);
  drc_net->add_routing_rect(0, drc_rect2);
  BoostRect boost_rect1(3, 3, 9, 4);
  drc_net->add_routing_rect(0, boost_rect1);
  rq->add_routing_rect_to_rtree(0, drc_rect2);

  DrcRect* drc_rect3 = new DrcRect();
  drc_rect3->set_net_id(1);
  drc_rect3->set_owner_type(RectOwnerType::kSegment);
  drc_rect3->set_is_fixed(false);
  drc_rect3->set_layer_id(0);
  drc_rect3->set_coordinate(1, 6, 2, 7);
  drc_net->add_routing_rect(0, drc_rect3);
  BoostRect boost_rect2(1, 6, 2, 7);
  drc_net->add_routing_rect(0, boost_rect2);
  rq->add_routing_rect_to_rtree(0, drc_rect3);

  idb_wrapper->initPolyPolygon(drc_net);

  idb_wrapper->initPolyEdges(drc_net);

  std::vector<std::unique_ptr<DrcPoly>>& poly = drc_net->get_route_polys(0);
  std::cout << poly.size() << std::endl;

  std::vector<std::vector<std::unique_ptr<DrcEdge>>>& ss = poly[0]->getEdges();
  std::cout << ss.size() << std::endl;

  for (int j = 0; j < ss.size(); j++) {
    for (int i = 0; i < ss[j].size(); i++) {
      std::cout << ss[j][i]->get_begin_x() << "   " << ss[j][i]->get_begin_y() << std::endl;
      std::cout << ss[j][i]->get_end_x() << "   " << ss[j][i]->get_end_y() << std::endl;
    }
  }

  auto eol_spacing_check = EOLSpacingCheck::getInstance();
  auto& rules = eol_spacing_check->get_lef58_eol_spacing_rule_list();
  idb::routinglayer::Lef58SpacingEol rule;
  rule.set_eol_space(2);
  rule.set_eol_width(2);
  rule.set_eol_within(1);
  idb::routinglayer::Lef58SpacingEol::AdjEdgeLength adj;
  adj.set_min_length(5);
  rule.set_adj_edge_length(adj);
  rules.push_back(make_shared<idb::routinglayer::Lef58SpacingEol>(rule));

  eol_spacing_check->checkEOLSpacing(drc_net);
}

// prl
void runTestCase4()
{
  RegionQuery* rq = RegionQuery::getInstance();
  DrcIDBWrapper* idb_wrapper = new DrcIDBWrapper(rq);
  //需要线网，tech层数
  DrcNet* drc_net = new DrcNet();
  DrcRect* drc_rect1 = new DrcRect();
  drc_rect1->set_net_id(1);
  drc_rect1->set_owner_type(RectOwnerType::kSegment);
  drc_rect1->set_is_fixed(false);
  drc_rect1->set_layer_id(0);
  drc_rect1->set_coordinate(1, 1, 2, 5);
  drc_net->add_routing_rect(0, drc_rect1);
  BoostRect boost_rect(1, 1, 2, 5);
  drc_net->add_routing_rect(0, boost_rect);
  //   drc_net->add_routing_rect(0, boost_rect);
  rq->add_routing_rect_to_rtree(0, drc_rect1);

  DrcRect* drc_rect2 = new DrcRect();
  drc_rect2->set_net_id(1);
  drc_rect2->set_owner_type(RectOwnerType::kSegment);
  drc_rect2->set_is_fixed(false);
  drc_rect2->set_layer_id(0);
  drc_rect2->set_coordinate(3, 3, 4, 5);
  drc_net->add_routing_rect(0, drc_rect2);
  BoostRect boost_rect1(3, 3, 4, 5);
  drc_net->add_routing_rect(0, boost_rect1);
  rq->add_routing_rect_to_rtree(0, drc_rect2);

  DrcRect* drc_rect3 = new DrcRect();
  drc_rect3->set_net_id(1);
  drc_rect3->set_owner_type(RectOwnerType::kSegment);
  drc_rect3->set_is_fixed(false);
  drc_rect3->set_layer_id(0);
  drc_rect3->set_coordinate(1, 6, 2, 7);
  drc_net->add_routing_rect(0, drc_rect3);
  BoostRect boost_rect2(1, 6, 2, 7);
  drc_net->add_routing_rect(0, boost_rect2);
  rq->add_routing_rect_to_rtree(0, drc_rect3);

  DrcRect* drc_rect4 = new DrcRect();
  drc_rect4->set_net_id(1);
  drc_rect4->set_owner_type(RectOwnerType::kSegment);
  drc_rect4->set_is_fixed(false);
  drc_rect4->set_layer_id(0);
  drc_rect4->set_coordinate(1, 8, 2, 10);
  drc_net->add_routing_rect(0, drc_rect4);
  BoostRect boost_rect4(1, 8, 2, 10);
  drc_net->add_routing_rect(0, boost_rect4);
  rq->add_routing_rect_to_rtree(0, drc_rect4);

  idb_wrapper->initPolyPolygon(drc_net);

  idb_wrapper->initPolyEdges(drc_net);

  std::vector<std::unique_ptr<DrcPoly>>& poly = drc_net->get_route_polys(0);
  std::cout << poly.size() << std::endl;

  std::vector<std::vector<std::unique_ptr<DrcEdge>>>& ss = poly[0]->getEdges();
  std::cout << ss.size() << std::endl;

  for (int j = 0; j < ss.size(); j++) {
    for (int i = 0; i < ss[j].size(); i++) {
      std::cout << ss[j][i]->get_begin_x() << "   " << ss[j][i]->get_begin_y() << std::endl;
      std::cout << ss[j][i]->get_end_x() << "   " << ss[j][i]->get_end_y() << std::endl;
    }
  }

  auto eol_spacing_check = EOLSpacingCheck::getInstance();
  auto& rules = eol_spacing_check->get_lef58_eol_spacing_rule_list();
  idb::routinglayer::Lef58SpacingEol rule;
  rule.set_eol_space(2);
  rule.set_eol_width(2);
  rule.set_eol_within(1);
  idb::routinglayer::Lef58SpacingEol::ParallelEdge prl;
  prl.set_par_space(2);
  prl.set_par_within(1);

  rule.set_parallel_edge(prl);
  rules.push_back(make_shared<idb::routinglayer::Lef58SpacingEol>(rule));

  eol_spacing_check->checkEOLSpacing(drc_net);
}

// cut
void runTestCase5()
{
  RegionQuery* rq = RegionQuery::getInstance();
  DrcIDBWrapper* idb_wrapper = new DrcIDBWrapper(rq);
  //需要线网，tech层数
  DrcNet* drc_net = new DrcNet();
  DrcRect* drc_rect1 = new DrcRect();
  drc_rect1->set_net_id(1);
  drc_rect1->set_owner_type(RectOwnerType::kSegment);
  drc_rect1->set_is_fixed(false);
  drc_rect1->set_layer_id(0);
  drc_rect1->set_coordinate(1, 1, 2, 5);
  drc_net->add_routing_rect(0, drc_rect1);
  BoostRect boost_rect(1, 1, 2, 5);
  drc_net->add_routing_rect(0, boost_rect);
  rq->add_routing_rect_to_rtree(0, drc_rect1);

  DrcRect* drc_rect2 = new DrcRect();
  drc_rect2->set_net_id(1);
  drc_rect2->set_owner_type(RectOwnerType::kSegment);
  drc_rect2->set_is_fixed(false);
  drc_rect2->set_layer_id(0);
  drc_rect2->set_coordinate(3, 6, 4, 7);
  drc_net->add_routing_rect(0, drc_rect2);
  BoostRect boost_rect1(3, 6, 4, 7);
  drc_net->add_routing_rect(0, boost_rect1);
  rq->add_routing_rect_to_rtree(0, drc_rect2);

  // DrcRect* drc_rect3 = new DrcRect();
  // drc_rect3->set_net_id(1);
  // drc_rect3->set_owner_type(RectOwnerType::kSegment);
  // drc_rect3->set_is_fixed(false);
  // drc_rect3->set_layer_id(0);
  // drc_rect3->set_coordinate(1, 12, 2, 14);
  // drc_net->add_routing_rect(0, drc_rect3);
  // BoostRect boost_rect2(1, 12, 2, 14);
  // drc_net->add_routing_rect(0, boost_rect2);
  // rq->add_routing_rect_to_rtree(0, drc_rect3);

  DrcRect* drc_rect4 = new DrcRect();
  drc_rect4->set_net_id(1);
  drc_rect4->set_owner_type(RectOwnerType::kSegment);
  drc_rect4->set_is_fixed(false);
  drc_rect4->set_layer_id(0);
  drc_rect4->set_coordinate(1, 8, 2, 10);
  drc_net->add_routing_rect(0, drc_rect4);
  BoostRect boost_rect4(1, 8, 2, 10);
  drc_net->add_routing_rect(0, boost_rect4);
  rq->add_routing_rect_to_rtree(0, drc_rect4);

  DrcRect* drc_cut_rect = new DrcRect();
  drc_cut_rect->set_coordinate(1, 3, 2, 4);
  // BoostRect boost_cut_rect(1, 3, 2, 4);
  rq->add_cut_rect_to_rtree(1, drc_cut_rect);

  idb_wrapper->initPolyPolygon(drc_net);

  idb_wrapper->initPolyEdges(drc_net);

  std::vector<std::unique_ptr<DrcPoly>>& poly = drc_net->get_route_polys(0);
  std::cout << poly.size() << std::endl;

  std::vector<std::vector<std::unique_ptr<DrcEdge>>>& ss = poly[0]->getEdges();
  std::cout << ss.size() << std::endl;

  for (int j = 0; j < ss.size(); j++) {
    for (int i = 0; i < ss[j].size(); i++) {
      std::cout << ss[j][i]->get_begin_x() << "   " << ss[j][i]->get_begin_y() << std::endl;
      std::cout << ss[j][i]->get_end_x() << "   " << ss[j][i]->get_end_y() << std::endl;
    }
  }

  auto eol_spacing_check = EOLSpacingCheck::getInstance();
  auto& rules = eol_spacing_check->get_lef58_eol_spacing_rule_list();
  idb::routinglayer::Lef58SpacingEol rule;
  rule.set_eol_space(2);
  rule.set_eol_width(2);
  rule.set_eol_within(2);
  idb::routinglayer::Lef58SpacingEol::EncloseCut cut;
  cut.set_direction("ABOVE");
  cut.set_enclose_dist(2);
  cut.set_cut_to_metal_space(7);
  cut.set_all_cuts(true);

  rule.set_enclose_cut(cut);
  rules.push_back(make_shared<idb::routinglayer::Lef58SpacingEol>(rule));

  eol_spacing_check->checkEOLSpacing(drc_net);
}

// End2End
void runTestCase6()
{
  RegionQuery* rq = RegionQuery::getInstance();
  DrcIDBWrapper* idb_wrapper = new DrcIDBWrapper(rq);
  //需要线网，tech层数
  DrcNet* drc_net = new DrcNet();
  DrcRect* drc_rect1 = new DrcRect();
  drc_rect1->set_net_id(1);
  drc_rect1->set_owner_type(RectOwnerType::kSegment);
  drc_rect1->set_is_fixed(false);
  drc_rect1->set_layer_id(0);
  drc_rect1->set_coordinate(1, 1, 2, 5);
  drc_net->add_routing_rect(0, drc_rect1);
  BoostRect boost_rect(1, 1, 2, 5);
  drc_net->add_routing_rect(0, boost_rect);
  rq->add_routing_rect_to_rtree(0, drc_rect1);

  DrcRect* drc_rect2 = new DrcRect();
  drc_rect2->set_net_id(1);
  drc_rect2->set_owner_type(RectOwnerType::kSegment);
  drc_rect2->set_is_fixed(false);
  drc_rect2->set_layer_id(0);
  drc_rect2->set_coordinate(6, 1, 7, 5);
  drc_net->add_routing_rect(0, drc_rect2);
  BoostRect boost_rect1(6, 1, 7, 5);
  drc_net->add_routing_rect(0, boost_rect1);
  rq->add_routing_rect_to_rtree(0, drc_rect2);

  DrcRect* drc_rect3 = new DrcRect();
  drc_rect3->set_net_id(1);
  drc_rect3->set_owner_type(RectOwnerType::kSegment);
  drc_rect3->set_is_fixed(false);
  drc_rect3->set_layer_id(0);
  drc_rect3->set_coordinate(1, 4, 7, 5);
  drc_net->add_routing_rect(0, drc_rect3);
  BoostRect boost_rect2(1, 4, 7, 5);
  drc_net->add_routing_rect(0, boost_rect2);
  rq->add_routing_rect_to_rtree(0, drc_rect3);

  DrcRect* drc_rect4 = new DrcRect();
  drc_rect4->set_net_id(1);
  drc_rect4->set_owner_type(RectOwnerType::kSegment);
  drc_rect4->set_is_fixed(false);
  drc_rect4->set_layer_id(0);
  drc_rect4->set_coordinate(1, 1, 7, 2);
  drc_net->add_routing_rect(0, drc_rect4);
  BoostRect boost_rect4(1, 1, 7, 2);
  drc_net->add_routing_rect(0, boost_rect4);
  rq->add_routing_rect_to_rtree(0, drc_rect4);

  // DrcRect* drc_cut_rect = new DrcRect();
  // drc_cut_rect->set_coordinate(1, 3, 2, 4);
  // // BoostRect boost_cut_rect(1, 3, 2, 4);
  // rq->add_cut_rect_to_rtree(1, drc_cut_rect);

  idb_wrapper->initPolyPolygon(drc_net);

  idb_wrapper->initPolyEdges(drc_net);

  std::vector<std::unique_ptr<DrcPoly>>& poly = drc_net->get_route_polys(0);
  std::cout << poly.size() << std::endl;

  std::vector<std::vector<std::unique_ptr<DrcEdge>>>& ss = poly[0]->getEdges();
  std::cout << ss.size() << std::endl;

  for (int j = 0; j < ss.size(); j++) {
    for (int i = 0; i < ss[j].size(); i++) {
      std::cout << ss[j][i]->get_begin_x() << "   " << ss[j][i]->get_begin_y() << std::endl;
      std::cout << ss[j][i]->get_end_x() << "   " << ss[j][i]->get_end_y() << std::endl;
    }
  }

  auto eol_spacing_check = EOLSpacingCheck::getInstance();
  auto& rules = eol_spacing_check->get_lef58_eol_spacing_rule_list();
  idb::routinglayer::Lef58SpacingEol rule;
  rule.set_eol_space(2);
  rule.set_eol_width(2);
  rule.set_eol_within(1);
  idb::routinglayer::Lef58SpacingEol::EndToEnd en2;
  en2.set_end_to_end_space(2);
  rule.set_end_to_end(en2);

  rules.push_back(make_shared<idb::routinglayer::Lef58SpacingEol>(rule));

  eol_spacing_check->checkEOLSpacing(drc_net);
}

// test inner edge
void runTestCase7()
{
  RegionQuery* rq = RegionQuery::getInstance();
  DrcIDBWrapper* idb_wrapper = new DrcIDBWrapper(rq);
  //需要线网，tech层数
  DrcNet* drc_net = new DrcNet();
  // DrcRect* drc_rect1 = new DrcRect();
  // drc_rect1->set_net_id(1);
  // drc_rect1->set_owner_type(RectOwnerType::kSegment);
  // drc_rect1->set_is_fixed(false);z
  // drc_rect1->set_layer_id(0);
  // drc_rect1->set_coordinate(1, 1, 2, 100);
  // drc_net->add_routing_rect(0, drc_rect1);
  // BoostRect boost_rect(1, 1, 2, 100);
  // drc_net->add_routing_rect(0, boost_rect);
  // rq->add_routing_rect_to_rtree(0, drc_rect1);

  // DrcRect* drc_rect2 = new DrcRect();
  // drc_rect2->set_net_id(1);
  // drc_rect2->set_owner_type(RectOwnerType::kSegment);
  // drc_rect2->set_is_fixed(false);
  // drc_rect2->set_layer_id(0);
  // drc_rect2->set_coordinate(1, 99, 100, 100);
  // drc_net->add_routing_rect(0, drc_rect2);
  // BoostRect boost_rect1(1, 99, 100, 100);
  // drc_net->add_routing_rect(0, boost_rect1);
  // rq->add_routing_rect_to_rtree(0, drc_rect2);

  // DrcRect* drc_rect3 = new DrcRect();
  // drc_rect3->set_net_id(1);
  // drc_rect3->set_owner_type(RectOwnerType::kSegment);
  // drc_rect3->set_is_fixed(false);
  // drc_rect3->set_layer_id(0);
  // drc_rect3->set_coordinate(99, 1, 100, 100);
  // drc_net->add_routing_rect(0, drc_rect3);
  // BoostRect boost_rect2(99, 1, 100, 100);
  // drc_net->add_routing_rect(0, boost_rect2);
  // rq->add_routing_rect_to_rtree(0, drc_rect3);

  DrcRect* drc_rect4 = new DrcRect();
  drc_rect4->set_net_id(1);
  drc_rect4->set_owner_type(RectOwnerType::kSegment);
  drc_rect4->set_is_fixed(false);
  drc_rect4->set_layer_id(0);
  drc_rect4->set_coordinate(50, 50, 150, 51);
  drc_net->add_routing_rect(0, drc_rect4);
  BoostRect boost_rect4(50, 50, 150, 51);
  drc_net->add_routing_rect(0, boost_rect4);
  rq->add_routing_rect_to_rtree(0, drc_rect4);

  DrcRect* drc_rect5 = new DrcRect();
  drc_rect5->set_net_id(1);
  drc_rect5->set_owner_type(RectOwnerType::kSegment);
  drc_rect5->set_is_fixed(false);
  drc_rect5->set_layer_id(0);
  drc_rect5->set_coordinate(48, 50, 49, 51);
  drc_net->add_routing_rect(0, drc_rect5);
  BoostRect boost_rect5(48, 50, 49, 51);
  drc_net->add_routing_rect(0, boost_rect5);
  rq->add_routing_rect_to_rtree(0, drc_rect5);

  // DrcRect* drc_cut_rect = new DrcRect();
  // drc_cut_rect->set_coordinate(1, 3, 2, 4);
  // // BoostRect boost_cut_rect(1, 3, 2, 4);
  // rq->add_cut_rect_to_rtree(1, drc_cut_rect);

  idb_wrapper->initPolyPolygon(drc_net);

  idb_wrapper->initPolyEdges(drc_net);

  std::vector<std::unique_ptr<DrcPoly>>& poly = drc_net->get_route_polys(0);
  std::cout << poly.size() << std::endl;

  std::vector<std::vector<std::unique_ptr<DrcEdge>>>& ss = poly[0]->getEdges();
  std::cout << ss.size() << std::endl;

  for (int j = 0; j < ss.size(); j++) {
    for (int i = 0; i < ss[j].size(); i++) {
      std::cout << ss[j][i]->get_begin_x() << "   " << ss[j][i]->get_begin_y() << std::endl;
      std::cout << ss[j][i]->get_end_x() << "   " << ss[j][i]->get_end_y() << std::endl;
    }
  }

  auto eol_spacing_check = EOLSpacingCheck::getInstance();
  auto& rules = eol_spacing_check->get_lef58_eol_spacing_rule_list();
  idb::routinglayer::Lef58SpacingEol rule;
  rule.set_eol_space(2);
  rule.set_eol_width(2);
  rule.set_eol_within(1);
  idb::routinglayer::Lef58SpacingEol::EndToEnd en2;
  en2.set_end_to_end_space(2);
  rule.set_end_to_end(en2);

  rules.push_back(make_shared<idb::routinglayer::Lef58SpacingEol>(rule));

  eol_spacing_check->checkEOLSpacing(drc_net);
  std::cout<<"endcheck"<<std::endl;
  SpotParser* _spot_parser = SpotParser::getInstance();

  _spot_parser->reportEOLSpacingViolation(eol_spacing_check);
}

int main(int argc, char* argv[])
{
  // runTestCase1();
  // runTestCase2();
  // runTestCase3();
  // runTestCase4();
  // runTestCase5();
  // runTestCase6();
  runTestCase7();
  return 0;
}
