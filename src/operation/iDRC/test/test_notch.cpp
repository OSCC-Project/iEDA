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
#include <string>

#include "DRC.h"
#include "DrcAPI.hpp"
#include "DrcIDBWrapper.h"
#include "DrcRect.h"
#include "NotchSpacingCheck.hpp"
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
  drc_rect1->set_coordinate(10, 10, 12, 40);
  drc_net->add_routing_rect(0, drc_rect1);
  BoostRect boost_rect(10, 10, 12, 40);
  drc_net->add_routing_rect(0, boost_rect);
  //   drc_net->add_routing_rect(0, boost_rect);
  rq->add_routing_rect_to_rtree(0, drc_rect1);

  DrcRect* drc_rect2 = new DrcRect();
  drc_rect2->set_net_id(1);
  drc_rect2->set_owner_type(RectOwnerType::kSegment);
  drc_rect2->set_is_fixed(false);
  drc_rect2->set_layer_id(0);
  drc_rect2->set_coordinate(21, 10, 23, 40);
  drc_net->add_routing_rect(0, drc_rect2);
  BoostRect boost_rect1(21, 10, 23, 40);
  drc_net->add_routing_rect(0, boost_rect1);

  DrcRect* drc_rect3 = new DrcRect();
  drc_rect3->set_net_id(1);
  drc_rect3->set_owner_type(RectOwnerType::kSegment);
  drc_rect3->set_is_fixed(false);
  drc_rect3->set_layer_id(0);
  drc_rect3->set_coordinate(8, 30, 24, 32);
  drc_net->add_routing_rect(0, drc_rect3);
  BoostRect boost_rect2(8, 30, 24, 32);
  drc_net->add_routing_rect(0, boost_rect2);
  rq->add_routing_rect_to_rtree(0, drc_rect3);

  DrcRect* drc_rect4 = new DrcRect();
  drc_rect4->set_net_id(1);
  drc_rect4->set_owner_type(RectOwnerType::kSegment);
  drc_rect4->set_is_fixed(false);
  drc_rect4->set_layer_id(0);
  drc_rect4->set_coordinate(20, 40, 25, 46);
  drc_net->add_routing_rect(0, drc_rect4);
  BoostRect boost_rect3(20, 40, 25, 46);
  drc_net->add_routing_rect(0, boost_rect3);
  rq->add_routing_rect_to_rtree(0, drc_rect4);

  DrcRect* drc_rect5 = new DrcRect();
  drc_rect5->set_net_id(1);
  drc_rect5->set_owner_type(RectOwnerType::kSegment);
  drc_rect5->set_is_fixed(false);
  drc_rect5->set_layer_id(0);
  drc_rect5->set_coordinate(10, 52, 13, 54);
  drc_net->add_routing_rect(0, drc_rect1);
  BoostRect boost_rect4(10, 52, 13, 54);
  drc_net->add_routing_rect(0, boost_rect4);
  //   drc_net->add_routing_rect(0, boost_rect);
  rq->add_routing_rect_to_rtree(0, drc_rect5);

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

  auto notch_spacing_check = NotchSpacingCheck::getInstance();

  // idb::IdbLayerSpacingNotchLength rule;
  // rule.set_min_spacing(10);
  // rule.set_notch_length(10);
  // notch_spacing_check->set_notch_spacing_rule(rule);

  idb::routinglayer::Lef58SpacingNotchlength rule1;
  rule1.set_min_notch_length(10);
  rule1.set_min_spacing(10);
  rule1.set_concave_ends_side_of_notch_width(3);
  notch_spacing_check->set_lef58_notch_spacing_rule(make_shared<idb::routinglayer::Lef58SpacingNotchlength>(rule1));

  notch_spacing_check->checkNotchSpacing(drc_net);
}

int main(int argc, char* argv[])
{
  // runTestCase1();
  // runTestCase2();
  // runTestCase3();
  // runTestCase4();
  // runTestCase5();
  // runTestCase6();
  runTestCase1();
  return 0;
}