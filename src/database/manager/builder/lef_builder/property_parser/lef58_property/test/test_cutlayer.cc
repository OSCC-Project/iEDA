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

#include "../cutlayer_property_parser.h"

using namespace idb::cutlayer_property;

TEST(CutLayerTest, LEF58CUTCLASS) {
  std::string test_str = R"(
        CUTCLASS VSINGLECUT WIDTH 0.05 LENGTH 0.05 CUTS 1 ;
        CUTCLASS VDOUBLECUT WIDTH 0.05 LENGTH 0.13 CUTS 2 ;)";
  std::vector<lef58_cutclass> vec;
  bool ok = parse_lef58_cutclass(test_str.begin(), test_str.end(), vec);
  EXPECT_TRUE(ok);
  EXPECT_EQ(vec.size(), 2);
  auto& cut1 = vec[0];
  EXPECT_EQ(cut1._classname, "VSINGLECUT");
  EXPECT_EQ(cut1._via_width, 0.05);
  EXPECT_EQ(cut1._via_length, 0.05);
  EXPECT_EQ(cut1._num_cut, 1);
  EXPECT_TRUE(cut1._orient.empty());

  auto& cut2 = vec[1];
  EXPECT_EQ(cut2._classname, "VDOUBLECUT");
  EXPECT_EQ(cut2._via_width, 0.05);
  EXPECT_EQ(cut2._via_length, 0.13);
  EXPECT_EQ(cut2._num_cut, 2);
  EXPECT_TRUE(cut2._orient.empty());
}

TEST(CutLayerTest, LEF58ENCLOSURE) {
  std::string test_str = R"(
        ENCLOSURE CUTCLASS VSINGLECUT 0 0.03 ;
        ENCLOSURE CUTCLASS VSINGLECUT 0.02 0.02 ;
        ENCLOSURE CUTCLASS VSINGLECUT 0.01 0.025 ;
        ENCLOSURE CUTCLASS VDOUBLECUT END 0.02 SIDE 0.02 ;
        ENCLOSURE CUTCLASS VDOUBLECUT END 0.03 SIDE 0.01 ;
        ENCLOSURE CUTCLASS VDOUBLECUT END 0.01 SIDE 0.03 ;
        ENCLOSURE CUTCLASS VDOUBLECUT END 0.04 SIDE 0 ;
)";
  std::vector<lef58_enclosure> vec;
  bool ok = parse_lef58_enclosure(test_str.begin(), test_str.end(), vec);
  EXPECT_TRUE(ok);
  EXPECT_EQ(vec.size(), 7);
  EXPECT_EQ(vec[0]._classname, "VSINGLECUT");
  EXPECT_EQ(vec[0]._overhang1.value(), 0);
  EXPECT_EQ(vec[0]._overhang2.value(), 0.03);
  EXPECT_EQ(vec[3]._classname, "VDOUBLECUT");
  EXPECT_EQ(vec[3]._end_overhang1.value(), 0.02);
  EXPECT_EQ(vec[3]._side_overhang2.value(), 0.02);
}

TEST(CutLayerTest, LEF58ENCLOSUREEDGE) {
  std::string test_str = R"(
        ENCLOSUREEDGE CUTCLASS VSINGLECUT 0.015000 WIDTH 0.160500 PARALLEL 0.100000 WITHIN 0.130000 ;
        ENCLOSUREEDGE CUTCLASS VSINGLECUT  0.010000 WIDTH 0.070500 PARALLEL 0.100000 WITHIN 0.100000 ;
        ENCLOSUREEDGE CUTCLASS VSINGLECUT  0.005000 WIDTH 0.055500 PARALLEL 0.100000 WITHIN 0.065000 ;
        ENCLOSUREEDGE CUTCLASS VSINGLECUT  0.005000 WIDTH 0.050500 PARALLEL 0.100000 WITHIN 0.060000 EXCEPTTWOEDGES ;
        ENCLOSUREEDGE CUTCLASS VDOUBLECUT 0.015000 WIDTH 0.160500 PARALLEL 0.100000 WITHIN 0.130000 ;
        ENCLOSUREEDGE CUTCLASS VDOUBLECUT  0.010000 WIDTH 0.070500 PARALLEL 0.100000 WITHIN 0.100000 ;
        ENCLOSUREEDGE CUTCLASS VDOUBLECUT  0.005000 WIDTH 0.055500 PARALLEL 0.100000 WITHIN 0.065000 ;
        ENCLOSUREEDGE CUTCLASS VDOUBLECUT  0.005000 WIDTH 0.050500 PARALLEL 0.100000 WITHIN 0.060000 EXCEPTTWOEDGES ;
        ENCLOSUREEDGE CUTCLASS VSINGLECUT ABOVE 0.01 CONVEXCORNERS 0.120 0.060 PARALLEL 0.051 LENGTH 0.1  ;
        ENCLOSUREEDGE CUTCLASS VDOUBLECUT ABOVE 0.01 CONVEXCORNERS 0.120 0.060 PARALLEL 0.051 LENGTH 0.1 ;)";
  std::vector<lef58_enclosureedge> vec;
  bool ok = parse_lef58_enclosureedge(test_str.begin(), test_str.end(), vec);
  EXPECT_TRUE(ok);
  EXPECT_EQ(vec.size(), 10);
  EXPECT_EQ(vec[0]._classname, "VSINGLECUT");
  EXPECT_TRUE(vec[0]._direction.empty());
  EXPECT_EQ(vec[0]._overhang, 0.015);
  EXPECT_EQ(vec[0]._width_convex.which(), 0);
  auto& width0 = boost::get<lef58_enclosureedge_width>(vec[0]._width_convex);
  EXPECT_EQ(width0._min_width, 0.1605);
  EXPECT_EQ(width0._par_length, 0.1);
  EXPECT_EQ(width0._par_within, 0.13);

  EXPECT_EQ(vec[3]._width_convex.which(), 0);
  auto& width3 = boost::get<lef58_enclosureedge_width>(vec[3]._width_convex);
  EXPECT_EQ(width3._except_two_edges, "EXCEPTTWOEDGES");
  EXPECT_TRUE(width3._except_extracut.empty());

  EXPECT_EQ(vec[4]._classname, "VDOUBLECUT");

  EXPECT_EQ(vec[8]._classname, "VSINGLECUT");
  EXPECT_EQ(vec[8]._direction, "ABOVE");
  EXPECT_EQ(vec[8]._overhang, 0.01);
  EXPECT_EQ(vec[8]._width_convex.which(), 1);
  auto& convex8 = boost::get<lef58_enclosureedge_convexcorners>(vec[8]._width_convex);
  EXPECT_EQ(convex8._convex_length, 0.12);
  EXPECT_EQ(convex8._adjacent_length, 0.06);
  EXPECT_EQ(convex8._par_within, 0.051);
  EXPECT_EQ(convex8._length, 0.1);
}

TEST(CutLayerTest, LEF58EOLENCLOSURE) {
  std::string test_str = R"(
        EOLENCLOSURE 0.070 CUTCLASS VSINGLECUT ABOVE 0.030 PARALLELEDGE 0.115 EXTENSION 0.070 0.025 MINLENGTH 0.050 ;)";
  lef58_eolenclosure eol_enclosure;
  bool ok = parse_lef58_eolenclosure(test_str.begin(), test_str.end(), eol_enclosure);
  ASSERT_TRUE(ok);
  ASSERT_EQ(eol_enclosure._eol_width, 0.07);
  ASSERT_EQ(eol_enclosure._classname, "VSINGLECUT");
  ASSERT_EQ(eol_enclosure._direction, "ABOVE");
  ASSERT_EQ(eol_enclosure._overhang.which(), 1);
  auto& overhang = boost::get<lef58_eolenclosure_overhang>(eol_enclosure._overhang);
  ASSERT_EQ(overhang._overhang, 0.03);
  ASSERT_EQ(overhang._par_space.value(), 0.115);
  ASSERT_EQ(overhang._backward_ext.value(), 0.07);
  ASSERT_EQ(overhang._forward_ext.value(), 0.025);
  ASSERT_EQ(overhang._min_length, 0.05);
}

TEST(CutLayerTest, LEF58EOLSPACING) {
  std::string test_str = R"(
        EOLSPACING 0.08 0.09 CUTCLASS VSINGLECUT TO VDOUBLECUT 0.085 0.09 ENDWIDTH 0.07 PRL -0.04
        ENCLOSURE 0.04 0.00 EXTENSION 0.065 0.12 SPANLENGTH 0.055 ;)";
  lef58_eolspacing eolspacing;
  bool ok = parse_lef58_eolspacing(test_str.begin(), test_str.end(), eolspacing);
  EXPECT_TRUE(ok);
  EXPECT_EQ(eolspacing._cut_spacing1, 0.08);
  EXPECT_EQ(eolspacing._cut_spacing2, 0.09);
  EXPECT_EQ(eolspacing._classname1, "VSINGLECUT");
  EXPECT_EQ(eolspacing._to_classes.size(), 1);
  lef58_eolspacing_toclass& toclass = eolspacing._to_classes[0];
  EXPECT_EQ(toclass._classname, "VDOUBLECUT");
  EXPECT_EQ(toclass._cut_spacing1, 0.085);
  EXPECT_EQ(toclass._cut_spacing2, 0.09);

  EXPECT_EQ(eolspacing._eol_width, 0.07);
  EXPECT_EQ(eolspacing._prl, -0.04);
  EXPECT_EQ(eolspacing._smaller_overhang, 0.04);
  EXPECT_EQ(eolspacing._equal_overhang, 0);

  EXPECT_EQ(eolspacing._side_ext, 0.065);
  EXPECT_EQ(eolspacing._backward_ext, 0.12);
  EXPECT_EQ(eolspacing._span_length, 0.055);
}

TEST(CutLayerTest, LEF58SPACINGTABLEcase1) {
  std::string test_str = R"(
      SPACINGTABLE LAYER VIA2 PRL 0.02
      CUTCLASS    VSINGLECUT  VDOUBLECUT
      VSINGLECUT  0.000 0.060 0.000 0.060
      VDOUBLECUT  0.000 0.060 0.000 0.060 ;)";
  lef58_spacingtable spacingtable;
  bool ok = parse_lef58_spacingtable(test_str.begin(), test_str.end(), spacingtable);
  EXPECT_TRUE(ok);

  EXPECT_TRUE(spacingtable._layer);
  auto& layer = spacingtable._layer.value();
  EXPECT_EQ(layer._second_layername, "VIA2");

  EXPECT_TRUE(spacingtable._prl);
  auto& prl = spacingtable._prl.value();
  EXPECT_EQ(prl._prl, 0.02);

  auto& cutclass = spacingtable._cutclass;
  EXPECT_EQ(cutclass._classname1.size(), 2);
  EXPECT_EQ(cutclass._classname1[0]._classname, "VSINGLECUT");
  EXPECT_EQ(cutclass._classname1[1]._classname, "VDOUBLECUT");

  auto& cuts = spacingtable._cutclass._cuts;
  EXPECT_EQ(cuts.size(), 2);
  EXPECT_EQ(cuts[0]._classname2._classname, "VSINGLECUT");
  EXPECT_EQ(cuts[0]._cutspacings[0]._cut1.value(), 0);
  EXPECT_EQ(cuts[0]._cutspacings[0]._cut2.value(), 0.06);
  EXPECT_EQ(cuts[0]._cutspacings[1]._cut1.value(), 0);
  EXPECT_EQ(cuts[0]._cutspacings[1]._cut2.value(), 0.06);

  EXPECT_EQ(cuts[1]._classname2._classname, "VDOUBLECUT");
  EXPECT_EQ(cuts[1]._cutspacings[0]._cut1.value(), 0);
  EXPECT_EQ(cuts[1]._cutspacings[0]._cut2.value(), 0.06);
  EXPECT_EQ(cuts[1]._cutspacings[1]._cut1.value(), 0);
  EXPECT_EQ(cuts[1]._cutspacings[1]._cut2.value(), 0.06);
}

TEST(CutLayerTest, LEF58SPACINGTABLEcase2) {
  std::string test_str = R"(
        SPACINGTABLE PRL -0.04 MAXXY
        CUTCLASS        VSINGLECUT      VDOUBLECUT
        VSINGLECUT      0.070 0.080     0.075 0.080
        VDOUBLECUT      0.075 0.080     0.080 0.080 ;)";
  lef58_spacingtable spacingtable;
  bool ok = parse_lef58_spacingtable(test_str.begin(), test_str.end(), spacingtable);
  EXPECT_TRUE(ok);

  EXPECT_FALSE(spacingtable._layer);

  EXPECT_TRUE(spacingtable._prl);
  auto& prl = spacingtable._prl.value();
  EXPECT_EQ(prl._prl, -0.04);
  EXPECT_EQ(prl._maxxy, "MAXXY");

  auto& cuts = spacingtable._cutclass._cuts;
  EXPECT_EQ(cuts.size(), 2);
  EXPECT_EQ(cuts[0]._classname2._classname, "VSINGLECUT");
  EXPECT_EQ(cuts[0]._cutspacings[0]._cut1.value(), 0.07);
  EXPECT_EQ(cuts[0]._cutspacings[0]._cut2.value(), 0.08);
  EXPECT_EQ(cuts[0]._cutspacings[1]._cut1.value(), 0.075);
  EXPECT_EQ(cuts[0]._cutspacings[1]._cut2.value(), 0.08);

  EXPECT_EQ(cuts[1]._classname2._classname, "VDOUBLECUT");
  EXPECT_EQ(cuts[1]._cutspacings[0]._cut1.value(), 0.075);
  EXPECT_EQ(cuts[1]._cutspacings[0]._cut2.value(), 0.08);
  EXPECT_EQ(cuts[1]._cutspacings[1]._cut1.value(), 0.08);
  EXPECT_EQ(cuts[1]._cutspacings[1]._cut2.value(), 0.08);
}