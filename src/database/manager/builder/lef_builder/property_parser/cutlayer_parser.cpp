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
#include "cutlayer_parser.h"

#include <optional>
#include <string>

#include "db_property/IdbCutLayerLef58Property.h"
#include "lef58_property/cutlayer_property_parser.h"
#include "property_parser/lef58_property/cutlayer_property.h"

namespace idb {
bool CutLayerParser::parse(const std::string& name, const std::string& value, IdbLayerCut* data)
{
  if (name == "LEF58_CUTCLASS") {
    return parse_lef58_cutclass(value, data);
  }
  if (name == "LEF58_ENCLOSURE") {
    return parse_lef58_enclosure(value, data);
  }
  if (name == "LEF58_ENCLOSUREEDGE") {
    return parse_lef58_enclosureedge(value, data);
  }
  if (name == "LEF58_EOLENCLOSURE") {
    return parse_lef58_eolenclosure(value, data);
  }
  if (name == "LEF58_EOLSPACING") {
    return parse_lef58_eolspacing(value, data);
  }
  if (name == "LEF58_SPACINGTABLE") {
    return parse_lef58_spacingtable(value, data);
  }
  std::cout << "Unhandled PROPERTY: " << name << " \"" << value << "\"" << std::endl;
  return false;
}
}  // namespace idb

/*
 * **********************************************************************
 *  Property Parsers & Parser Rules
 * **********************************************************************
 */

namespace idb {
bool CutLayerParser::parse_lef58_cutclass(const std::string& value, IdbLayerCut* data)
{
  std::vector<cutlayer_property::lef58_cutclass> vec;
  bool parse_ok = cutlayer_property::parse_lef58_cutclass(value.begin(), value.end(), vec);
  if (not parse_ok) {
    return false;
  }
  for (auto& item : vec) {
    auto cutclass = std::make_shared<cutlayer::Lef58Cutclass>();
    cutclass->set_class_name(std::move(item._classname));
    cutclass->set_via_width(this->transUnitDB(item._via_width));
    if (item._via_length) {
      cutclass->set_via_length(this->transUnitDB(item._via_length.value()));
    }
    if (item._num_cut) {
      cutclass->set_num_cut(item._num_cut.value());
    }
    cutclass->set_orient(item._orient);
    data->add_lef58_cutclass(std::move(cutclass));
  }
  return true;
}

bool CutLayerParser::parse_lef58_enclosure(const std::string& value, IdbLayerCut* data)
{
  std::vector<cutlayer_property::lef58_enclosure> vec;
  bool parse_ok = cutlayer_property::parse_lef58_enclosure(value.begin(), value.end(), vec);
  if (not parse_ok) {
    return false;
  }
  //
  for (auto& item : vec) {
    auto enclosure = std::make_shared<cutlayer::Lef58Enclosure> ();
    enclosure->set_class_name(std::move(item._classname));
    if (item._overhang1 && item._overhang2) {
      enclosure->set_overhang1(this->transUnitDB(item._overhang1.value()));
      enclosure->set_overhang2(this->transUnitDB(item._overhang2.value()));
    }
    if (item._end_overhang1 && item._side_overhang2) {
      enclosure->set_end_overhang1(this->transUnitDB(item._end_overhang1.value()));
      enclosure->set_side_overhang2(this->transUnitDB(item._side_overhang2.value()));
    }
    data->add_lef58_enclosure(std::move(enclosure));
  }
  return true;
}

bool CutLayerParser::parse_lef58_enclosureedge(const std::string& value, IdbLayerCut* data)
{
  std::vector<cutlayer_property::lef58_enclosureedge> vec;
  bool parse_ok = cutlayer_property::parse_lef58_enclosureedge(value.begin(), value.end(), vec);
  if (not parse_ok) {
    return false;
  }
  //
  for (auto& item : vec) {
    auto enclosure_edge = std::make_shared<cutlayer::Lef58EnclosureEdge>();
    enclosure_edge->set_class_name(std::move(item._classname));
    enclosure_edge->set_direction(item._direction);
    enclosure_edge->set_overhang(transUnitDB(item._overhang));
    if (item._width_convex.which() == 0) {
      auto& width = boost::get<cutlayer_property::lef58_enclosureedge_width>(item._width_convex);
      enclosure_edge->set_min_width(transUnitDB(width._min_width));
      enclosure_edge->set_par_length(transUnitDB(width._par_length));
      enclosure_edge->set_par_within(transUnitDB(width._par_within));
      if (width._except_extracut == "EXCEPTEXTRACUT") {
        enclosure_edge->set_except_extracut(true);
        if (width._cut_within) {
          enclosure_edge->set_except_extracut_cutwithin(transUnitDB(width._cut_within.value()));
        }
      }
      if (width._except_two_edges == "EXCEPTTWOEDGES") {
        enclosure_edge->set_except_twoedges(true);
        if (width._except_within) {
          enclosure_edge->set_except_within(transUnitDB(width._except_within.value()));
        }
      }

    } else if (item._width_convex.which() == 1) {
      auto& convex = boost::get<cutlayer_property::lef58_enclosureedge_convexcorners>(item._width_convex);
      cutlayer::Lef58EnclosureEdge::ConvexCorners convex_corners;
      convex_corners.set_convex_length(transUnitDB(convex._convex_length));
      convex_corners.set_adjacent_length(transUnitDB(convex._adjacent_length));
      convex_corners.set_par_within(transUnitDB(convex._par_within));
      convex_corners.set_length(transUnitDB(convex._length));
      enclosure_edge->set_convex_corners(convex_corners);
    }
    data->add_lef58_enclosure_edge(std::move(enclosure_edge));
  }
  return true;
}

bool CutLayerParser::parse_lef58_eolenclosure(const std::string& value, IdbLayerCut* data)
{
  cutlayer_property::lef58_eolenclosure enclosure_data;
  bool parse_ok = cutlayer_property::parse_lef58_eolenclosure(value.begin(), value.end(), enclosure_data);
  if (not parse_ok) {
    return false;
  }
  // dispose data
  auto enclosure = std::make_shared<cutlayer::Lef58EolEnclosure>();
  enclosure->set_eol_width(transUnitDB(enclosure_data._eol_width));
  if (enclosure_data._min_eol_width) {
    enclosure->set_min_eol_width(transUnitDB(enclosure_data._min_eol_width.value()));
  }
  enclosure->set_edge_direction(enclosure_data._edge_direction);
  enclosure->set_equal_rect_width(enclosure_data._equalrectwidth == "EQUALRECTWIDTH");
  enclosure->set_class_name(std::move(enclosure_data._classname));
  enclosure->set_direction(enclosure_data._direction);
  if (enclosure_data._overhang.which() == 0) {
    auto overhang = boost::get<cutlayer_property::lef58_eolenclosure_edgeoverhang>(enclosure_data._overhang);
    enclosure->set_application_type(overhang._applied_to);
    enclosure->set_overhang(transUnitDB(overhang._overhang));
  } else if (enclosure_data._overhang.which() == 1) {
    auto overhang = boost::get<cutlayer_property::lef58_eolenclosure_overhang>(enclosure_data._overhang);
    enclosure->set_overhang(transUnitDB(overhang._overhang));
    if (overhang._extract_overhang) {
      enclosure->set_extract_overhang(transUnitDB(overhang._extract_overhang.value()));
    }
    if (overhang._par_space) {
      enclosure->set_par_space(transUnitDB(overhang._par_space.value()));
      cutlayer::Lef58EolEnclosure::Extension ext;
      ext.set_backward_ext(transUnitDB(overhang._backward_ext.value()));
      ext.set_forward_ext(transUnitDB(overhang._forward_ext.value()));
      enclosure->set_extension(ext);
    }
    if (overhang._min_length) {
      enclosure->set_min_length(transUnitDB(overhang._min_length.value()));
    }
    enclosure->set_all_sides(overhang._allsides == "ALLSIDES");
  }
  data->set_lef58_eolenclosure(enclosure);
  return true;
}

bool CutLayerParser::parse_lef58_eolspacing(const std::string& value, IdbLayerCut* data)
{
  cutlayer_property::lef58_eolspacing eolspacing;
  bool parse_ok = cutlayer_property::parse_lef58_eolspacing(value.begin(), value.end(), eolspacing);
  if (not parse_ok) {
    return false;
  }
  // dispose data
  auto ptr = std::make_shared<cutlayer::Lef58EolSpacing>();
  ptr->set_cut_spacing1(transUnitDB(eolspacing._cut_spacing1));
  ptr->set_cut_spacing2(transUnitDB(eolspacing._cut_spacing2));
  ptr->set_class_name1(std::move(eolspacing._classname1));
  for (auto& toclass_item : eolspacing._to_classes) {
    cutlayer::Lef58EolSpacing::ToClass toclass;
    toclass.set_class_name(std::move(toclass_item._classname));
    toclass.set_cut_spacing1(transUnitDB(toclass_item._cut_spacing1));
    toclass.set_cut_spacing2(transUnitDB(toclass_item._cut_spacing2));
    ptr->add_to_class(std::move(toclass));
  }
  ptr->set_eol_width(transUnitDB(eolspacing._eol_width));
  ptr->set_prl(transUnitDB(eolspacing._prl));
  ptr->set_small_overhang(transUnitDB(eolspacing._smaller_overhang));
  ptr->set_equal_overhang(transUnitDB(eolspacing._equal_overhang));
  ptr->set_side_ext(transUnitDB(eolspacing._side_ext));
  ptr->set_backward_ext(transUnitDB(eolspacing._backward_ext));
  ptr->set_span_length(transUnitDB(eolspacing._span_length));

  data->set_lef58_eolspacing(ptr);
  return true;
}

bool CutLayerParser::parse_lef58_spacingtable(const std::string& value, IdbLayerCut* data)
{
  cutlayer_property::lef58_spacingtable spacingtable;
  bool parse_ok = cutlayer_property::parse_lef58_spacingtable(value.begin(), value.end(), spacingtable);
  if (not parse_ok) {
    return false;
  }
  // dispose data
  auto spacing_tbl = std::make_shared<cutlayer::Lef58SpacingTable>();
  if (spacingtable._layer) {
    cutlayer::Lef58SpacingTable::Layer layer;
    layer.set_second_layer_name(std::move(spacingtable._layer->_second_layername));
    spacing_tbl->set_layer(std::move(layer));
  }
  if (spacingtable._prl) {
    cutlayer::Lef58SpacingTable::Prl prl;
    prl.set_prl(transUnitDB(spacingtable._prl->_prl));
    prl.set_maxxy(!spacingtable._prl->_maxxy.empty());
    spacing_tbl->set_prl(prl);
  }

  {
    cutlayer::Lef58SpacingTable::CutClass cutclass;
    for (auto& classname1 : spacingtable._cutclass._classname1) {
      cutclass.add_class_name1(cutlayer::Lef58SpacingTable::ClassName(std::move(classname1._classname)));
    }
    for (auto& cut_row : spacingtable._cutclass._cuts) {
      cutclass.add_class_name2(cutlayer::Lef58SpacingTable::ClassName(std::move(cut_row._classname2._classname)));
      std::vector<cutlayer::Lef58SpacingTable::CutSpacing> cutspacing_row;
      for (auto& cut : cut_row._cutspacings) {
        std::optional<int32_t> cut_spacing1;
        std::optional<int32_t> cut_spacing2;
        if (cut._cut1) {
          cut_spacing1 = transUnitDB(cut._cut1.value());
        }
        if (cut._cut2) {
          cut_spacing2 = transUnitDB(cut._cut2.value());
        }
        cutspacing_row.emplace_back(cutlayer::Lef58SpacingTable::CutSpacing(cut_spacing1, cut_spacing2));
      }
      cutclass.add_cut_spacing_row(std::move(cutspacing_row));
    }
    spacing_tbl->set_cutclass(std::move(cutclass));
  }
  data->add_lef58_spacing_table(spacing_tbl);
  return true;
}

}  // namespace idb