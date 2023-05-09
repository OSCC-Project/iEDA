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
#include <gtest/gtest.h>

#include "../routinglayer_property_parser.h"

using namespace idb::routinglayer_property;

TEST(RoutingLayerTest, LEF58AREAcase1) {
  std::vector<lef58_area> areas;
  const std::string str = "AREA 0.014000 EXCEPTEDGELENGTH 0.130000 0.200000 EXCEPTMINSIZE 0.050000 0.200000 ;";

  bool ok = parse_lef58_area(str.begin(), str.end(), areas);

  EXPECT_TRUE(ok);
  EXPECT_EQ(areas.size(), 1);
  auto& area = areas[0];
  EXPECT_EQ(area._min_area, 0.014);
  EXPECT_FALSE(area._mask_num);
  EXPECT_FALSE(area._except_min_width);

  EXPECT_TRUE(area._exceptedgelength);
  auto& edgelength = area._exceptedgelength.value();
  EXPECT_EQ(edgelength._min_edge_length, 0.13);
  EXPECT_EQ(edgelength._max_edge_length.value(), 0.2);
  EXPECT_EQ(area._except_min_size.size(), 1);
  auto& except_min_size = area._except_min_size[0];
  EXPECT_EQ(except_min_size.first, 0.05);
  EXPECT_EQ(except_min_size.second, 0.2);

  EXPECT_FALSE(area._except_step);
  EXPECT_FALSE(area._rect_width);
  EXPECT_TRUE(area._exceptrectangle.empty());
  EXPECT_TRUE(area._trim_layer.empty());
  EXPECT_FALSE(area._overlap);
}

TEST(RoutingLayerTest, LEF58AREAcase2) {
  std::vector<lef58_area> areas;
  const std::string str = "AREA 0.038000 EXCEPTEDGELENGTH 0.130000 EXCEPTMINSIZE 0.050000 0.130000 ;";

  bool ok = parse_lef58_area(str.begin(), str.end(), areas);

  EXPECT_TRUE(ok);
  EXPECT_EQ(areas.size(), 1);
  auto& area = areas[0];
  EXPECT_EQ(area._min_area, 0.038);
  EXPECT_FALSE(area._mask_num);
  EXPECT_FALSE(area._except_min_width);

  EXPECT_TRUE(area._exceptedgelength);
  auto& edgelength = area._exceptedgelength.value();
  EXPECT_EQ(edgelength._min_edge_length, 0.13);
  EXPECT_FALSE(edgelength._max_edge_length);
  EXPECT_EQ(area._except_min_size.size(), 1);
  auto& except_min_size = area._except_min_size[0];
  EXPECT_EQ(except_min_size.first, 0.05);
  EXPECT_EQ(except_min_size.second, 0.13);

  EXPECT_FALSE(area._except_step);
  EXPECT_FALSE(area._rect_width);
  EXPECT_TRUE(area._exceptrectangle.empty());
  EXPECT_TRUE(area._trim_layer.empty());
  EXPECT_FALSE(area._overlap);
}

TEST(RoutingLayerTest, LEF58CORNERFILLSPACING) {
  lef58_cornerfillspacing spacing;
  const std::string str = "CORNERFILLSPACING 0.05 EDGELENGTH 0.05 0.12  ADJACENTEOL 0.06 ;";

  bool ok = parse_lef58_conerfillspacing(str.begin(), str.end(), spacing);
  EXPECT_TRUE(ok);
  EXPECT_EQ(spacing._spacing, 0.05);
  EXPECT_EQ(spacing._length1, 0.05);
  EXPECT_EQ(spacing._length2, 0.12);
  EXPECT_EQ(spacing._eol_width, 0.06);
}

TEST(RoutingLayerTest, LEF58MINIMUNCUT) {
  std::vector<lef58_minimumcut> cuts;
  const std::string str = R"(
    MINIMUMCUT CUTCLASS VSINGLECUT 2 CUTCLASS VDOUBLECUT 1 WIDTH 0.180000 WITHIN 0.100000 FROMBELOW ;
    MINIMUMCUT CUTCLASS VSINGLECUT 4 CUTCLASS VDOUBLECUT 2 WIDTH 0.440000 FROMBELOW ;
    MINIMUMCUT CUTCLASS VSINGLECUT 2 CUTCLASS VDOUBLECUT 1 WIDTH 0.180000 FROMBELOW LENGTH 0.180000 WITHIN 1.651000 ;
    MINIMUMCUT CUTCLASS VSINGLECUT 2 CUTCLASS VDOUBLECUT 1 WIDTH 1.000000 FROMBELOW LENGTH 1.000000 WITHIN 4.001000 ;
    MINIMUMCUT CUTCLASS VSINGLECUT 2 CUTCLASS VDOUBLECUT 1 WIDTH 1.500000 FROMBELOW LENGTH 5.000000 WITHIN 10.001000 ;
    MINIMUMCUT 2 WIDTH 1.800000 FROMABOVE ;
    MINIMUMCUT 2 WIDTH 3.000000 FROMABOVE LENGTH 10.000000 WITHIN 5.001000 ;)";

  bool ok = parse_lef58_minimumcut(str.begin(), str.end(), cuts);
  EXPECT_TRUE(ok);
  EXPECT_EQ(cuts.size(),7);
  auto& cut0 = cuts[0];
  EXPECT_EQ(cut0._cuts.size(), 2);
  EXPECT_EQ(cut0._cuts[0]._class_name, "VSINGLECUT");
  EXPECT_EQ(cut0._cuts[0]._num_cuts, 2);
  EXPECT_EQ(cut0._cuts[1]._class_name, "VDOUBLECUT");
  EXPECT_EQ(cut0._cuts[1]._num_cuts, 1);
  EXPECT_EQ(cut0._width, 0.18);
  EXPECT_EQ(cut0._cut_distance.value(), 0.1);
  EXPECT_EQ(cut0._direction, "FROMBELOW");

  auto& cut2 = cuts[2];
  EXPECT_EQ(cut2._cuts.size(), 2);
  EXPECT_EQ(cut2._cuts[0]._class_name, "VSINGLECUT");
  EXPECT_EQ(cut2._cuts[0]._num_cuts, 2);
  EXPECT_EQ(cut2._cuts[1]._class_name, "VDOUBLECUT");
  EXPECT_EQ(cut2._cuts[1]._num_cuts, 1);
  EXPECT_EQ(cut2._width, 0.18);
  EXPECT_FALSE(cut2._cut_distance);
  EXPECT_EQ(cut2._direction, "FROMBELOW");
  EXPECT_EQ(cut2._length, 0.18);
  EXPECT_EQ(cut2._length_within,1.651);

  auto& cut6=cuts[6];
  EXPECT_EQ(cut6._num_cuts, 2);
  EXPECT_EQ(cut6._width,3);
  EXPECT_EQ(cut6._direction,"FROMABOVE");
  EXPECT_EQ(cut6._length,10);
  EXPECT_EQ(cut6._length_within,5.001);
}

TEST(RoutingLayerTest, LEF58MINSTEP) {
  std::vector<lef58_minstep> minsteps;
  std::string str = "MINSTEP 0.05 MAXEDGES 1 MINADJACENTLENGTH 0.065 CONVEXCORNER ;";

  bool ok = parse_lef58_minstep(str.begin(), str.end(), minsteps);
  EXPECT_TRUE(ok);
  EXPECT_EQ(minsteps.size(),1);
  auto& minstep = minsteps[0];
  EXPECT_EQ(minstep._min_step_length, 0.05);
  EXPECT_EQ(minstep._max_edges, 1);
  EXPECT_EQ(minstep._min_adj_length,0.065);
  EXPECT_EQ(minstep._convex_corner,"CONVEXCORNER");

  EXPECT_TRUE(minstep._type.empty());
  EXPECT_FALSE(minstep._max_length);
  EXPECT_FALSE(minstep._min_adj_length2);
  EXPECT_TRUE(minstep._concave_corner.empty());
  EXPECT_TRUE(minstep._three_concave_corners.empty());
  EXPECT_FALSE(minstep._center_width);
  EXPECT_FALSE(minstep._min_between_length);
  EXPECT_TRUE(minstep._except_same_corners.empty());
  EXPECT_FALSE(minstep._no_adjacent_eol);
  EXPECT_FALSE(minstep._except_adjacent_length);
  EXPECT_TRUE(minstep._concavecorners.empty());
  EXPECT_FALSE(minstep._no_between_eol);

}

TEST(RoutingLayerTest, LEF58SPACINGNOTCHLENGTH){
  lef58_spacing_notchlength spacing;
  std::string str = R"(
    SPACING 0.07 NOTCHLENGTH 0.155 CONCAVEENDS 0.055 ;)";
  
  bool ok = parse_lef58_spacing_notchlength(str.begin(),str.end(), spacing);
  EXPECT_TRUE(ok);
  EXPECT_EQ(spacing._min_spacing, 0.07);
  EXPECT_EQ(spacing._min_notch_length,0.155);
  EXPECT_EQ(spacing._side_type, "CONCAVEENDS");
  EXPECT_EQ(spacing._side_of_notch_width, 0.055);

  EXPECT_FALSE(spacing._low_exclude_spacing);
  EXPECT_FALSE(spacing._high_exclude_spacing);
  EXPECT_FALSE(spacing._within);
  EXPECT_FALSE(spacing._side_of_notch_span_length);
  EXPECT_FALSE(spacing._notch_width);
}

TEST(RoutingLayerTest, LEF58SPACINGENDOFLINEcase1){
  std::vector<lef58_spacing_eol> spacings;
  std::string str = R"(
        SPACING 0.060 ENDOFLINE 0.070 WITHIN 0.025 ;
        SPACING 0.070 ENDOFLINE 0.070 WITHIN 0.025 PARALLELEDGE SUBTRACTEOLWIDTH 0.120 WITHIN 0.070 MINLENGTH 0.050 ;)";
  bool ok = parse_lef58_spacing_eol(str.begin(), str.end(), spacings);
  EXPECT_TRUE(ok);
  EXPECT_EQ(spacings.size(), 2);
  auto& spacing0 = spacings[0];
  EXPECT_EQ(spacing0._eol_space, 0.06);
  EXPECT_EQ(spacing0._eol_width, 0.07);
  EXPECT_EQ(spacing0._eol_within, 0.025);
  auto& spacing1 = spacings[1];
  EXPECT_EQ(spacing1._eol_space, 0.07);
  EXPECT_EQ(spacing1._eol_width, 0.07);
  EXPECT_EQ(spacing1._eol_within, 0.025);
  EXPECT_TRUE(spacing1._parallel_edge);
  auto& parallel_edge1 = spacing1._parallel_edge.value();
  EXPECT_EQ(parallel_edge1._subtract_eol_width, "SUBTRACTEOLWIDTH");
  EXPECT_EQ(parallel_edge1._par_space, 0.12);
  EXPECT_EQ(parallel_edge1._par_within, 0.07);
  EXPECT_EQ(parallel_edge1._min_length, 0.05);
}

TEST(RoutingLayerTest, LEF58SPACINGENDOFLINEcase2){
  std::vector<lef58_spacing_eol> spacings;
  std::string str = R"(
        SPACING 0.115 ENDOFLINE 0.055 WITHIN 0.000 PARALLELEDGE 0.060 WITHIN 0.120 MINLENGTH 0.150 TWOEDGES SAMEMETAL ;)";
  bool ok = parse_lef58_spacing_eol(str.begin(), str.end(), spacings);
  EXPECT_TRUE(ok);
  EXPECT_EQ(spacings.size(), 1);
  auto& spacing = spacings[0];
  EXPECT_EQ(spacing._eol_space, 0.115);
  EXPECT_EQ(spacing._eol_width, 0.055);
  EXPECT_EQ(spacing._eol_within, 0);
  EXPECT_TRUE(spacing._parallel_edge);
  auto& parallel_edge = spacing._parallel_edge.value();
  EXPECT_EQ(parallel_edge._par_space, 0.06);
  EXPECT_EQ(parallel_edge._par_within, 0.12);
  EXPECT_EQ(parallel_edge._min_length, 0.15);
  EXPECT_EQ(parallel_edge._two_edgs, "TWOEDGES");
  EXPECT_EQ(parallel_edge._same_metal, "SAMEMETAL");
}

TEST(RoutingLayerTest, LEF58SPACINGENDOFLINEcase3){
  std::vector<lef58_spacing_eol> spacings;
  std::string str = R"(
        SPACING 0.070 ENDOFLINE 0.070 WITHIN 0.025 ENDTOEND 0.080 ;
        SPACING 0.080 ENDOFLINE 0.070 WITHIN 0.025 ENDTOEND 0.080 PARALLELEDGE SUBTRACTEOLWIDTH 0.115 WITHIN 0.070 MINLENGTH 0.050 ;
        SPACING 0.100 ENDOFLINE 0.070 WITHIN 0.025 ENDTOEND 0.080 PARALLELEDGE SUBTRACTEOLWIDTH 0.115 WITHIN 0.070 MINLENGTH 0.050 ENCLOSECUT BELOW 0.050 CUTSPACING 0.145 ALLCUTS ;)";
  bool ok = parse_lef58_spacing_eol(str.begin(), str.end(), spacings);
  EXPECT_TRUE(ok);
  EXPECT_EQ(spacings.size(), 3);
  auto& spacing2 = spacings[2];
  EXPECT_EQ(spacing2._eol_space, 0.100);
  EXPECT_EQ(spacing2._eol_width, 0.070);
  EXPECT_EQ(spacing2._eol_within, 0.025);

  EXPECT_TRUE(spacing2._parallel_edge);
  auto& parallel_edge2 = spacing2._parallel_edge.value();
  EXPECT_EQ(parallel_edge2._subtract_eol_width, "SUBTRACTEOLWIDTH");
  EXPECT_EQ(parallel_edge2._par_space, 0.115);
  EXPECT_EQ(parallel_edge2._par_within, 0.07);
  EXPECT_EQ(parallel_edge2._min_length, 0.05);

  EXPECT_TRUE(spacing2._enclose_cut);
  auto enclose_cut2 = spacing2._enclose_cut.value();
  EXPECT_EQ(enclose_cut2._direction, "BELOW");
  EXPECT_EQ(enclose_cut2._enclose_dist, 0.05);
  EXPECT_EQ(enclose_cut2._cut_to_metal_space, 0.145);
  EXPECT_EQ(enclose_cut2._all_cuts, "ALLCUTS");
}

TEST(RoutingLayerTest, LEF58SPACINGTABLEJOGTOJOG){
  lef58_spacingtable_jogtojog spacingtable;
  std::string str = R"(
       SPACINGTABLE JOGTOJOGSPACING 0.3 JOGWIDTH 0.22
       SHORTJOGSPACING 0.06
       WIDTH 0.25 PARALLEL 0.3 WITHIN 0.29 LONGJOGSPACING 0.08
       WIDTH 0.25 PARALLEL 0.3 WITHIN 0.19 LONGJOGSPACING 0.10
       WIDTH 0.47 PARALLEL 0.5 WITHIN 0.32 LONGJOGSPACING 0.13
       WIDTH 0.63 PARALLEL 0.7 WITHIN 0.34 LONGJOGSPACING 0.15
       WIDTH 1.50 PARALLEL 1.5 WITHIN 0.50 LONGJOGSPACING 0.30 ;)";
  bool ok = parse_lef58_spacingtable_jogtojog(str.begin(), str.end(), spacingtable);
  EXPECT_TRUE(ok);
  EXPECT_EQ(spacingtable._jog2jog_spacing, 0.3);
  EXPECT_EQ(spacingtable._jog_width, 0.22);
  EXPECT_EQ(spacingtable._short_jog_spacing, 0.06);
  EXPECT_EQ(spacingtable._width.size(), 5);
  EXPECT_EQ(spacingtable._width[0]._width, 0.25);
  EXPECT_EQ(spacingtable._width[0]._par_length,0.3);
  EXPECT_EQ(spacingtable._width[0]._par_within, 0.29);
  EXPECT_EQ(spacingtable._width[0]._long_jog_spacing, 0.08);
  EXPECT_EQ(spacingtable._width[4]._width, 1.5);
  EXPECT_EQ(spacingtable._width[4]._par_length, 1.5);
  EXPECT_EQ(spacingtable._width[4]._par_within, 0.5);
  EXPECT_EQ(spacingtable._width[4]._long_jog_spacing, 0.3);

}