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
 * @file routinglayer_property.h
 * @author pengming
 * @brief
 * @version 0.1
 * @date 2022-10-17
 */

#pragma once
#include <boost/spirit/include/qi.hpp>
#include <boost/variant/variant.hpp>
#include <optional>

// #include "../property_parser.h"

namespace idb::routinglayer_property {
namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
using ascii::space_type, ascii::space;
using qi::lit, qi::double_, qi::int_, qi::char_, qi::lexeme;

using double_pair = std::pair<double, double>;

struct lef58_area_exceptedgelength
{
  double _min_edge_length;
  std::optional<double> _max_edge_length;
};
struct lef58_area
{
  double _min_area;
  std::optional<int> _mask_num;
  std::optional<double> _except_min_width;
  std::optional<lef58_area_exceptedgelength> _exceptedgelength;
  std::vector<double_pair> _except_min_size;
  std::optional<double_pair> _except_step;

  std::optional<double> _rect_width;
  std::string _exceptrectangle;
  std::string _trim_layer;
  std::optional<int> _overlap;
};

struct lef58_cornerfillspacing
{
  double _spacing;
  double _length1;
  double _length2;
  double _eol_width;
};

struct name_cuts
{
  std::string _class_name;
  int _num_cuts;
};
struct lef58_minimumcut
{
  std::optional<int> _num_cuts;
  std::vector<name_cuts> _cuts;
  double _width;
  std::optional<double> _cut_distance;
  std::string _direction;
  std::optional<double> _length;
  std::optional<double> _length_within;
  std::optional<double> _area;
  std::optional<double> _area_within;
  std::string _samemetal_overlap;
  std::string _fully_enclosed;
};

struct lef58_minstep
{
  double _min_step_length;
  // [INSIDECORNER|OUTSIDECORNER|STEP]
  std::string _type;
  std::optional<double> _max_length;
  std::optional<int> _max_edges;

  std::optional<double> _min_adj_length;
  std::optional<double> _min_adj_length2;
  std::string _convex_corner;
  // [EXCEPTWITHIN exceptWithin]
  std::optional<double> _except_within;
  std::string _concave_corner;
  std::string _three_concave_corners;
  std::optional<double> _center_width;

  std::optional<double> _min_between_length;
  std::string _except_same_corners;

  std::optional<double> _no_adjacent_eol;
  std::optional<double> _except_adjacent_length;
  std::optional<double> _min_adjacent_length;
  std::string _concavecorners;

  std::optional<double> _no_between_eol;
};

struct lef58_spacing_notchlength
{
  double _min_spacing;
  double _min_notch_length;
  std::optional<double> _low_exclude_spacing;
  std::optional<double> _high_exclude_spacing;
  std::optional<double> _within;
  std::optional<double> _side_of_notch_span_length;
  // {WIDTH | CONCAVEENDS}
  std::string _side_type;
  std::optional<double> _side_of_notch_width;
  std::optional<double> _notch_width;
};

//////////////////////////////////
// LEF58_SPACING with ENDOFLINE //
//////////////////////////////////
struct lef58_spacing_eol_withcut
{
  std::string _cutclass;
  std::string _above;
  double _with_cut_space;
  std::optional<double> _enclosure_end_width;
  std::optional<double> _enclosure_end_within;
};

struct lef58_spacing_eol_endprlspacing
{
  double _end_prl_space;
  double _end_prl;
};
struct lef58_spacing_eol_endtoend
{
  double _end_to_end_space;
  std::optional<double> _one_cut_space;
  std::optional<double> _two_cut_space;
  std::optional<double> _extension;
  std::optional<double> _wrong_dir_extension;
  std::optional<double> _other_end_width;
};

struct lef58_spacing_eol_paralleledge
{
  std::string _subtract_eol_width;
  double _par_space;
  double _par_within;
  std::optional<double> _prl;
  std::optional<double> _min_length;
  std::string _two_edgs;
  std::string _same_metal;
  std::string _non_eol_corner_only;
  std::string _parallel_same_mask;
};

struct lef58_spacing_eol_enclosecut
{
  // [BELOW | ABOVE]
  std::string _direction;
  double _enclose_dist;
  double _cut_to_metal_space;
  std::string _all_cuts;
};

struct lef58_spacing_eol_toconcavecorner
{
  std::optional<double> _min_length;
  std::optional<double> _min_adj_length1;
  std::optional<double> _min_adj_length2;
};

struct lef58_spacing_eol
{
  double _eol_space;
  double _eol_width;
  std::string _exact_width;
  std::optional<double> _wrong_dir_space;
  std::optional<double> _opposite_width;
  std::optional<double> _eol_within;
  std::optional<double> _wrong_dir_within;
  std::string _same_mask;
  std::optional<double_pair> _except_exact_width;
  std::optional<double> _fill_triangle;
  std::optional<lef58_spacing_eol_withcut> _withcut;
  std::optional<lef58_spacing_eol_endprlspacing> _end_prl_spacing;
  std::optional<lef58_spacing_eol_endtoend> _end_to_end;

  std::optional<double> _max_length;
  std::optional<double> _min_length;
  std::string _two_sides;

  std::string _equal_rect_width;
  std::optional<lef58_spacing_eol_paralleledge> _parallel_edge;
  std::optional<lef58_spacing_eol_enclosecut> _enclose_cut;
  std::optional<lef58_spacing_eol_toconcavecorner> _toconcavecorner;

  std::optional<double> _notch_length;
};

////////////////////////////////////////
// LEF58_SPACINGTABLE JOGTOJOGSPACING //
////////////////////////////////////////
struct lef58_spacingtable_jogtojog_width
{
  double _width;
  double _par_length;
  double _par_within;
  std::optional<double> _low_exclude_spacing;
  std::optional<double> _high_exclude_spacing;
  double _long_jog_spacing;
  std::optional<double> _width_short_jog_spacing;
};

struct lef58_spacingtable_jogtojog
{
  double _jog2jog_spacing;
  double _jog_width;
  double _short_jog_spacing;
  std::vector<lef58_spacingtable_jogtojog_width> _width;
};

}  // namespace idb::routinglayer_property

BOOST_FUSION_ADAPT_STRUCT(idb::routinglayer_property::double_pair, (double, first)(double, second))
BOOST_FUSION_ADAPT_STRUCT(idb::routinglayer_property::lef58_area_exceptedgelength,
                          (double, _min_edge_length)(std::optional<double>, _max_edge_length))

BOOST_FUSION_ADAPT_STRUCT(idb::routinglayer_property::lef58_area,
                          (double, _min_area)(std::optional<int>, _mask_num)(std::optional<double>, _except_min_width)(
                              std::optional<idb::routinglayer_property::lef58_area_exceptedgelength>,
                              _exceptedgelength)(std::vector<idb::routinglayer_property::double_pair>,
                                                 _except_min_size)(std::optional<idb::routinglayer_property::double_pair>, _except_step)(
                              std::optional<double>, _rect_width)(std::string, _exceptrectangle)(std::string,
                                                                                                 _trim_layer)(std::optional<int>, _overlap))

BOOST_FUSION_ADAPT_STRUCT(idb::routinglayer_property::lef58_cornerfillspacing,
                          (double, _spacing)(double, _length1)(double, _length2)(double, _eol_width))

BOOST_FUSION_ADAPT_STRUCT(idb::routinglayer_property::name_cuts, (std::string, _class_name)(int, _num_cuts))
BOOST_FUSION_ADAPT_STRUCT(idb::routinglayer_property::lef58_minimumcut,
                          (std::optional<int>, _num_cuts)(std::vector<idb::routinglayer_property::name_cuts>,
                                                          _cuts)(double, _width)(std::optional<double>, _cut_distance)(
                              std::string, _direction)(std::optional<double>, _length)(std::optional<double>, _length_within)(
                              std::optional<double>, _area)(std::optional<double>,
                                                            _area_within)(std::string, _samemetal_overlap)(std::string, _fully_enclosed))

BOOST_FUSION_ADAPT_STRUCT(
    idb::routinglayer_property::lef58_minstep,
    (double, _min_step_length)(std::string, _type)(std::optional<double>, _max_length)(std::optional<int>, _max_edges)(
        std::optional<double>, _min_adj_length)(std::optional<double>, _min_adj_length2)(std::string, _convex_corner)(std::optional<double>,
                                                                                                                      _except_within)(
        std::string, _concave_corner)(std::string, _three_concave_corners)(std::optional<double>, _center_width)(std::optional<double>,
                                                                                                                 _min_between_length)(
        std::string, _except_same_corners)(std::optional<double>, _no_adjacent_eol)(std::optional<double>, _except_adjacent_length)(
        std::optional<double>, _min_adjacent_length)(std::string, _concavecorners)(std::optional<double>, _no_between_eol))

BOOST_FUSION_ADAPT_STRUCT(idb::routinglayer_property::lef58_spacing_notchlength,
                          (double, _min_spacing)(double, _min_notch_length)(std::optional<double>, _low_exclude_spacing)(
                              std::optional<double>, _high_exclude_spacing)(std::optional<double>, _within)(std::optional<double>,
                                                                                                            _side_of_notch_span_length)(
                              std::string, _side_type)(std::optional<double>, _side_of_notch_width)(std::optional<double>, _notch_width))

BOOST_FUSION_ADAPT_STRUCT(idb::routinglayer_property::lef58_spacing_eol_withcut,
                          (std::string, _cutclass)(std::string, _above)(double, _with_cut_space)(
                              std::optional<double>, _enclosure_end_width)(std::optional<double>, _enclosure_end_within))

BOOST_FUSION_ADAPT_STRUCT(idb::routinglayer_property::lef58_spacing_eol_endprlspacing, (double, _end_prl_space)(double, _end_prl))

BOOST_FUSION_ADAPT_STRUCT(idb::routinglayer_property::lef58_spacing_eol_endtoend,
                          (double, _end_to_end_space)(std::optional<double>, _one_cut_space)(std::optional<double>, _two_cut_space)(
                              std::optional<double>, _extension)(std::optional<double>, _wrong_dir_extension)(std::optional<double>,
                                                                                                              _other_end_width))

BOOST_FUSION_ADAPT_STRUCT(idb::routinglayer_property::lef58_spacing_eol_paralleledge,
                          (std::string, _subtract_eol_width)(double, _par_space)(double, _par_within)(std::optional<double>, _prl)(
                              std::optional<double>, _min_length)(std::string, _two_edgs)(std::string, _same_metal)(
                              std::string, _non_eol_corner_only)(std::string, _parallel_same_mask))

BOOST_FUSION_ADAPT_STRUCT(idb::routinglayer_property::lef58_spacing_eol_enclosecut,
                          (std::string, _direction)(double, _enclose_dist)(double, _cut_to_metal_space)(std::string, _all_cuts))
BOOST_FUSION_ADAPT_STRUCT(idb::routinglayer_property::lef58_spacing_eol_toconcavecorner,
                          (std::optional<double>, _min_length)(std::optional<double>, _min_adj_length1)(std::optional<double>,
                                                                                                        _min_adj_length2))

BOOST_FUSION_ADAPT_STRUCT(
    idb::routinglayer_property::lef58_spacing_eol,
    (double, _eol_space)(double, _eol_width)(std::string, _exact_width)(std::optional<double>, _wrong_dir_space)(
        std::optional<double>, _opposite_width)(std::optional<double>, _eol_within)(std::optional<double>, _wrong_dir_within)(
        std::string, _same_mask)(std::optional<idb::routinglayer_property::double_pair>, _except_exact_width)(std::optional<double>,
                                                                                                              _fill_triangle)(
        std::optional<idb::routinglayer_property::lef58_spacing_eol_withcut>,
        _withcut)(std::optional<idb::routinglayer_property::lef58_spacing_eol_endprlspacing>,
                  _end_prl_spacing)(std::optional<idb::routinglayer_property::lef58_spacing_eol_endtoend>, _end_to_end)

        (std::optional<double>, _max_length)(std::optional<double>, _min_length)(std::string, _two_sides)

            (std::string, _equal_rect_width)(std::optional<idb::routinglayer_property::lef58_spacing_eol_paralleledge>, _parallel_edge)(
                std::optional<idb::routinglayer_property::lef58_spacing_eol_enclosecut>,
                _enclose_cut)(std::optional<idb::routinglayer_property::lef58_spacing_eol_toconcavecorner>, _toconcavecorner)

                (std::optional<double>, _notch_length)

)

BOOST_FUSION_ADAPT_STRUCT(idb::routinglayer_property::lef58_spacingtable_jogtojog_width,
                          (double, _width)(double, _par_length)(double, _par_within)(std::optional<double>, _low_exclude_spacing)(
                              std::optional<double>, _high_exclude_spacing)(double, _long_jog_spacing)(std::optional<double>,
                                                                                                       _width_short_jog_spacing))
BOOST_FUSION_ADAPT_STRUCT(idb::routinglayer_property::lef58_spacingtable_jogtojog,
                          (double, _jog2jog_spacing)(double, _jog_width)(double, _short_jog_spacing)(
                              std::vector<idb::routinglayer_property::lef58_spacingtable_jogtojog_width>, _width))