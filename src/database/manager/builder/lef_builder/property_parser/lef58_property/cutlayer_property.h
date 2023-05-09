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
/**
 * @file cutlayer_property.h
 * @author pengming
 * @brief lef58 properties of cut layer
 * @version 0.1
 * @date 2022-10-14
 */
#pragma once
#include <boost/spirit/include/qi.hpp>

// #include "../property_parser.h"

namespace idb::cutlayer_property {
namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
using ascii::space_type, ascii::space;
using qi::lit, qi::double_, qi::int_, qi::char_, qi::lexeme;

struct lef58_cutclass
{
  std::string _classname;
  double _via_width;
  std::optional<double> _via_length;
  std::optional<int> _num_cut;
  std::string _orient;
};

struct lef58_enclosure
{
  std::string _classname;
  std::string _direction;
  std::optional<double> _overhang1;
  std::optional<double> _overhang2;
  std::optional<double> _end_overhang1;
  std::optional<double> _side_overhang2;
  std::optional<double> _min_width;
  std::optional<double> _cut_winthin;
  std::string _except_extracut_type;
  std::optional<double> _min_length;
  std::optional<double> _cut_within;
};

struct lef58_enclosureedge_width
{
  double _min_width;
  double _par_length;
  double _par_within;
  std::string _except_extracut;
  std::optional<double> _cut_within;
  std::string _except_two_edges;
  std::optional<double> _except_within;
};
struct lef58_enclosureedge_convexcorners
{
  double _convex_length;
  double _adjacent_length;
  double _par_within;
  double _length;
};
struct lef58_enclosureedge
{
  std::string _classname;
  std::string _direction;  // ABOVE|BELOW
  double _overhang;
  boost::variant<lef58_enclosureedge_width, lef58_enclosureedge_convexcorners> _width_convex;
};

struct lef58_eolenclosure_edgeoverhang
{
  std::string _applied_to;  // LONGEDGEONLY | SHORTEDGEONLY
  double _overhang;
};

struct lef58_eolenclosure_overhang
{
  double _overhang;
  std::optional<double> _extract_overhang;
  std::optional<double> _par_space;
  std::optional<double> _backward_ext;
  std::optional<double> _forward_ext;
  std::optional<double> _min_length;
  std::string _allsides;
};

struct lef58_eolenclosure
{
  double _eol_width;
  std::optional<double> _min_eol_width;
  std::string _edge_direction;  // HORIZONTAL | VERTICAL
  std::string _equalrectwidth;
  std::string _classname;
  std::string _direction;  // ABOVE | BELOW, specifies the rule only applies to above/below routing layer
  boost::variant<lef58_eolenclosure_edgeoverhang, lef58_eolenclosure_overhang> _overhang;
};
struct lef58_eolspacing_toclass
{
  std::string _classname;
  double _cut_spacing1;
  double _cut_spacing2;
};
struct lef58_eolspacing
{
  double _cut_spacing1;
  double _cut_spacing2;
  std::string _classname1;
  std::vector<lef58_eolspacing_toclass> _to_classes;
  double _eol_width;
  double _prl;
  double _smaller_overhang;
  double _equal_overhang;
  double _side_ext;
  double _backward_ext;
  double _span_length;
};
struct lef58_spacingtable_layer
{
  std::string _second_layername;
  // ...
};
struct lef58_spacingtable_prl
{
  double _prl;
  std::string _direction;  // HORIZONTAL | VERTICAL
  std::string _maxxy;
  // ...
};
struct lef58_spacingtable_classname
{
  std::string _classname;
  std::string _edge;  // SIDE|END
};
struct lef58_spacingtable_cutspacing
{
  std::optional<double> _cut1;
  std::optional<double> _cut2;
};
struct lef58_spacingtable_cutspacings
{
  lef58_spacingtable_classname _classname2;
  std::vector<lef58_spacingtable_cutspacing> _cutspacings;
};
struct lef58_spacingtable_cutclass
{
  std::vector<lef58_spacingtable_classname> _classname1;
  std::vector<lef58_spacingtable_cutspacings> _cuts;
};
struct lef58_spacingtable
{
  std::optional<lef58_spacingtable_layer> _layer;
  std::optional<lef58_spacingtable_prl> _prl;
  lef58_spacingtable_cutclass _cutclass;
};
}  // namespace idb::cutlayer_property

// BOOST FUSION ADAPT STRUCT
// Make data struct accessible for boost::spirit

BOOST_FUSION_ADAPT_STRUCT(idb::cutlayer_property::lef58_cutclass,
                          (std::string, _classname)(double, _via_width)(std::optional<double>, _via_length)(std::optional<int>,
                                                                                                            _num_cut)(std::string, _orient))

BOOST_FUSION_ADAPT_STRUCT(idb::cutlayer_property::lef58_enclosure,
                          (std::string, _classname)(std::string, _direction)(std::optional<double>,
                                                                             _overhang1)(std::optional<double>,
                                                                                         _overhang2)(std::optional<double>, _end_overhang1)(
                              std::optional<double>, _side_overhang2)(std::optional<double>, _min_width)(std::optional<double>,
                                                                                                         _cut_winthin)(
                              std::string, _except_extracut_type)(std::optional<double>, _min_length)(std::optional<double>, _cut_within))

BOOST_FUSION_ADAPT_STRUCT(idb::cutlayer_property::lef58_enclosureedge_width,
                          (double, _min_width)(double, _par_length)(double, _par_within)(std::string, _except_extracut)(
                              std::optional<double>, _cut_within)(std::string, _except_two_edges)(std::optional<double>, _except_within)

)
BOOST_FUSION_ADAPT_STRUCT(idb::cutlayer_property::lef58_enclosureedge_convexcorners,
                          (double, _convex_length)(double, _adjacent_length)(double, _par_within)(double, _length))
using width_convex_vairant
    = boost::variant<idb::cutlayer_property::lef58_enclosureedge_width, idb::cutlayer_property::lef58_enclosureedge_convexcorners>;
BOOST_FUSION_ADAPT_STRUCT(idb::cutlayer_property::lef58_enclosureedge,
                          (std::string, _classname)(std::string, _direction)(double, _overhang)(width_convex_vairant, _width_convex))

BOOST_FUSION_ADAPT_STRUCT(idb::cutlayer_property::lef58_eolenclosure_edgeoverhang, (std::string, _applied_to)(double, _overhang))

BOOST_FUSION_ADAPT_STRUCT(idb::cutlayer_property::lef58_eolenclosure_overhang,
                          (double, _overhang)(std::optional<double>, _extract_overhang)(std::optional<double>,
                                                                                        _par_space)(std::optional<double>, _backward_ext)(
                              std::optional<double>, _forward_ext)(std::optional<double>, _min_length)(std::string, _allsides))
using enclosure_variant
    = boost::variant<idb::cutlayer_property::lef58_eolenclosure_edgeoverhang, idb::cutlayer_property::lef58_eolenclosure_overhang>;

BOOST_FUSION_ADAPT_STRUCT(idb::cutlayer_property::lef58_eolenclosure,
                          (double, _eol_width)(std::optional<double>, _min_eol_width)(std::string, _edge_direction)(
                              std::string, _equalrectwidth)(std::string, _classname)(std::string, _direction)(enclosure_variant, _overhang))

BOOST_FUSION_ADAPT_STRUCT(idb::cutlayer_property::lef58_eolspacing_toclass,
                          (std::string, _classname)(double, _cut_spacing1)(double, _cut_spacing2))
using to_class_vector = std::vector<idb::cutlayer_property::lef58_eolspacing_toclass>;
BOOST_FUSION_ADAPT_STRUCT(
    idb::cutlayer_property::lef58_eolspacing,
    (double, _cut_spacing1)(double, _cut_spacing2)(std::string, _classname1)(to_class_vector, _to_classes)(double, _eol_width)(
        double, _prl)(double, _smaller_overhang)(double, _equal_overhang)(double, _side_ext)(double, _backward_ext)(double, _span_length))

BOOST_FUSION_ADAPT_STRUCT(idb::cutlayer_property::lef58_spacingtable_layer, (std::string, _second_layername))

BOOST_FUSION_ADAPT_STRUCT(idb::cutlayer_property::lef58_spacingtable_prl, (double, _prl)(std::string, _direction)(std::string, _maxxy))

BOOST_FUSION_ADAPT_STRUCT(idb::cutlayer_property::lef58_spacingtable_classname, (std::string, _classname)(std::string, _edge))

BOOST_FUSION_ADAPT_STRUCT(idb::cutlayer_property::lef58_spacingtable_cutspacing,
                          (std::optional<double>, _cut1)(std::optional<double>, _cut2))

BOOST_FUSION_ADAPT_STRUCT(idb::cutlayer_property::lef58_spacingtable_cutspacings,
                          (idb::cutlayer_property::lef58_spacingtable_classname,
                           _classname2)(std::vector<idb::cutlayer_property::lef58_spacingtable_cutspacing>, _cutspacings))

BOOST_FUSION_ADAPT_STRUCT(idb::cutlayer_property::lef58_spacingtable_cutclass,
                          (std::vector<idb::cutlayer_property::lef58_spacingtable_classname>,
                           _classname1)(std::vector<idb::cutlayer_property::lef58_spacingtable_cutspacings>, _cuts))

BOOST_FUSION_ADAPT_STRUCT(idb::cutlayer_property::lef58_spacingtable,
                          (std::optional<idb::cutlayer_property::lef58_spacingtable_layer>,
                           _layer)(std::optional<idb::cutlayer_property::lef58_spacingtable_prl>,
                                   _prl)(idb::cutlayer_property::lef58_spacingtable_cutclass, _cutclass))
