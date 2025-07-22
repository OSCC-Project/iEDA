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
#include "io_placer.h"

#include "idm.h"

using namespace std;

namespace ifp {
/**
 * @brief
 *
 * @param edge
 * @return IdbOrient
 */
idb::IdbOrient IoPlacer::transferEdgeToOrient(Edge edge)
{
  if (edge == Edge::kBottom) {
    return IdbOrient::kN_R0;
  }
  if (edge == Edge::kRight) {
    return IdbOrient::kW_R90;
  }
  if (edge == Edge::kTop) {
    return IdbOrient::kS_R180;
  }
  if (edge == Edge::kLeft) {
    return IdbOrient::kE_R270;
  }
  return IdbOrient::kN_R0;
}

std::string IoPlacer::transferOrientToString(idb::IdbOrient orient)
{
  if (orient == idb::IdbOrient::kN_R0) {
    return "N";
  } else if (orient == idb::IdbOrient::kS_R180) {
    return "S";
  } else if (orient == idb::IdbOrient::kW_R90) {
    return "W";
  } else if (orient == idb::IdbOrient::kE_R270) {
    return "E";
  }
  return "NO Orient";
}

bool IoPlacer::edgeIsSameToOrient(Edge edge, idb::IdbOrient orient)
{
  if (edge == Edge::kBottom && orient == idb::IdbOrient::kN_R0) {
    return true;
  } else if (edge == Edge::kRight && orient == idb::IdbOrient::kW_R90) {
    return true;
  } else if (edge == Edge::kTop && orient == idb::IdbOrient::kS_R180) {
    return true;
  } else if (edge == Edge::kLeft && orient == idb::IdbOrient::kE_R270) {
    return true;
  }
  return false;
}
/**
 * @brief auto place io pins on specified layer & width height
 *
 * @param layer_name
 * @param width
 * @param height
 * @return true
 * @return false
 */
bool IoPlacer::autoPlacePins(std::string layer_name, int width, int height)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();

  auto idb_die = idb_layout->get_die();
  auto idb_core = idb_layout->get_core();
  auto layer = idb_layout->get_layers()->find_layer(layer_name);
  auto pin_list = idb_design->get_io_pin_list()->get_pin_list();

  /// calculate all the location
  int pin_num = pin_list.size();
  int edge_num = pin_num % 4 == 0 ? pin_num / 4 : pin_num / 4 + 1;
  int manufacture_grid = dmInst->get_idb_lef_service()->get_layout()->get_munufacture_grid();
  int width_step = idb_core->get_bounding_box()->get_width() / (edge_num + 1);
  int height_step = idb_core->get_bounding_box()->get_height() / (edge_num + 1);
  width_step = width_step / manufacture_grid * manufacture_grid;
  height_step = height_step / manufacture_grid * manufacture_grid;
  int pin_index = 0;
  /// left
  for (int i = 0; i < edge_num; ++i) {
    if (pin_index >= pin_num) {
      break;
    }

    int x = idb_die->get_llx() + width / 2;
    int y = idb_core->get_bounding_box()->get_low_y() + i * height_step;

    auto pin = pin_list[pin_index++];
    pin->set_location(x, y);

    auto io_term = pin->get_term();
    io_term->set_placement_status_place();
    auto port = io_term->add_port(nullptr);
    auto shape = port->add_layer_shape();
    shape->set_type_rect();
    
    // Calculate shape coordinates with left-bottom corner aligned to manufacture_grid
    int shape_llx = x - width / 2;
    int shape_lly = y - height / 2;
    shape_llx = (shape_llx / manufacture_grid) * manufacture_grid;
    shape_lly = (shape_lly / manufacture_grid) * manufacture_grid;
    int shape_urx = shape_llx + width;
    int shape_ury = shape_lly + height;
    
    shape->add_rect(shape_llx - x, shape_lly - y, shape_urx - x, shape_ury - y);
    shape->set_layer(layer);
  }

  /// right
  for (int i = 0; i < edge_num; ++i) {
    if (pin_index >= pin_num) {
      break;
    }

    int x = idb_die->get_urx() - width / 2;
    int y = idb_core->get_bounding_box()->get_low_y() + i * height_step;

    auto pin = pin_list[pin_index++];
    pin->set_location(x, y);

    auto io_term = pin->get_term();
    io_term->set_placement_status_place();
    auto port = io_term->add_port(nullptr);
    auto shape = port->add_layer_shape();
    shape->set_type_rect();
    
    // Calculate shape coordinates with left-bottom corner aligned to manufacture_grid
    int shape_llx = x - width / 2;
    int shape_lly = y - height / 2;
    shape_llx = (shape_llx / manufacture_grid) * manufacture_grid;
    shape_lly = (shape_lly / manufacture_grid) * manufacture_grid;
    int shape_urx = shape_llx + width;
    int shape_ury = shape_lly + height;
    
    shape->add_rect(shape_llx - x, shape_lly - y, shape_urx - x, shape_ury - y);
    shape->set_layer(layer);
  }

  /// bottom
  for (int i = 0; i < edge_num; ++i) {
    if (pin_index >= pin_num) {
      break;
    }

    int x = idb_core->get_bounding_box()->get_low_x() + i * width_step;
    int y = idb_die->get_lly() + height / 2;

    auto pin = pin_list[pin_index++];
    pin->set_location(x, y);

    auto io_term = pin->get_term();
    io_term->set_placement_status_place();
    auto port = io_term->add_port(nullptr);
    auto shape = port->add_layer_shape();
    shape->set_type_rect();
    
    // Calculate shape coordinates with left-bottom corner aligned to manufacture_grid
    int shape_llx = x - width / 2;
    int shape_lly = y - height / 2;
    shape_llx = (shape_llx / manufacture_grid) * manufacture_grid;
    shape_lly = (shape_lly / manufacture_grid) * manufacture_grid;
    int shape_urx = shape_llx + width;
    int shape_ury = shape_lly + height;
    
    shape->add_rect(shape_llx - x, shape_lly - y, shape_urx - x, shape_ury - y);
    shape->set_layer(layer);
  }

  /// top
  for (int i = 0; i < edge_num; ++i) {
    if (pin_index >= pin_num) {
      break;
    }

    int x = idb_core->get_bounding_box()->get_low_x() + i * width_step;
    int y = idb_die->get_ury() - height / 2;

    auto pin = pin_list[pin_index++];
    pin->set_location(x, y);

    auto io_term = pin->get_term();
    io_term->set_placement_status_place();
    auto port = io_term->add_port(nullptr);
    auto shape = port->add_layer_shape();
    shape->set_type_rect();
    
    // Calculate shape coordinates with left-bottom corner aligned to manufacture_grid
    int shape_llx = x - width / 2;
    int shape_lly = y - height / 2;
    shape_llx = (shape_llx / manufacture_grid) * manufacture_grid;
    shape_lly = (shape_lly / manufacture_grid) * manufacture_grid;
    int shape_urx = shape_llx + width;
    int shape_ury = shape_lly + height;
    
    shape->add_rect(shape_llx - x, shape_lly - y, shape_urx - x, shape_ury - y);
    shape->set_layer(layer);
  }

  return true;
}

bool IoPlacer::placePort(std::string pin_name, int32_t x_offset, int32_t y_offset, int32_t rect_width, int32_t rect_height,
                         std::string layer_name)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();

  ///
  auto layer = idb_layout->get_layers()->find_layer(layer_name);
  if (layer == nullptr) {
    std::cout << "Place Port Error : can not find Layer " << layer_name << std::endl;
    return false;
  }
  auto io_pin = idb_design->get_io_pin_list()->find_pin(pin_name);
  if (io_pin == nullptr) {
    std::cout << "Place Port Error : can not find IO Pin " << pin_name << std::endl;
    return false;
  }
  /// calculate rect for port
  auto io_cell = dmInst->getIoCellByIoPin(io_pin);
  io_cell->set_bounding_box();
  auto rect = io_cell->get_bounding_box();
  int32_t llx = rect->get_low_x();
  int32_t lly = rect->get_low_y();

  int32_t rect_llx = llx + x_offset;
  int32_t rect_lly = lly + y_offset;
  int32_t rect_urx = rect_llx + rect_width;
  int32_t rect_ury = rect_lly + rect_height;

  /// pin coordinate
  int32_t pin_x = (rect_llx + rect_urx) / 2;
  int32_t pin_y = (rect_lly + rect_ury) / 2;

  io_pin->set_average_coordinate(pin_x, pin_y);
  io_pin->set_orient();
  io_pin->set_location(pin_x, pin_y);

  /// set term attribute
  auto term = io_pin->get_term();
  if (term == nullptr) {
    std::cout << "Error : can not find IO Term." << std::endl;
    return false;
  }
  term->set_average_position(x_offset / 2, y_offset / 2);
  term->set_placement_status_fix();
  term->set_has_port(term->get_port_list().size() > 0 ? true : false);

  /// set port
  auto port = term->add_port();
  port->set_coordinate(pin_x, pin_y);

  auto layer_shape = port->add_layer_shape();
  layer_shape->set_layer(layer);
  layer_shape->add_rect(-rect_width / 2, -rect_height / 2, rect_width - rect_width / 2, rect_height - rect_height / 2);

  /// adjust the pin port coordinate
  io_pin->set_bounding_box();

  return true;
}

void IoPlacer::placeIOFiller(std::vector<idb::IdbCellMaster*>& fillers, const std::string prefix, PadCoordinate coord)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_inst_list = idb_design->get_instance_list();
  auto idb_io_inst_list = idb_inst_list->get_iopad_list();

  _iofiller_idx = -1;

  vector<Interval> used;
  vector<Interval> need_filler;
  for (auto io : idb_io_inst_list) {
    if ((io->is_placed() || io->is_fixed() || io->is_cover()) && edgeIsSameToOrient(coord.edge, io->get_orient())) {
      int32_t width = io->get_cell_master()->get_width();
      io->set_bounding_box();
      int32_t llx = io->get_bounding_box()->get_low_x();
      int32_t lly = io->get_bounding_box()->get_low_y();
      Interval inter = Interval();
      if (coord.edge == Edge::kLeft || coord.edge == Edge::kRight) {
        inter.set_edge(coord.edge);
        inter.set_begin_position(lly);
        inter.set_end_position(lly + width);
      } else {
        inter.set_edge(coord.edge);
        inter.set_begin_position(llx);
        inter.set_end_position(llx + width);
      }
      used.push_back(inter);
      if (io->get_cell_master()->get_type() == idb::CellMasterType::kPadSpacer) {
        ++_iofiller_idx;
      }
    }
  }

  if (used.empty()) {
    need_filler.push_back(Interval(coord.edge, coord.begin, coord.end));
  } else {
    sort(used.begin(), used.end(), [](Interval a, Interval b) { return a.get_begin_position() < b.get_begin_position(); });

    int32_t start_idx = 0, end_idx = 0;
    for (size_t i = 0; i != used.size(); ++i) {
      if (used[i].get_end_position() > coord.begin) {
        start_idx = i;
        break;
      }
    }
    for (ssize_t i = used.size() - 1; i != -1; --i) {
      if (used[i].get_begin_position() < coord.end) {
        end_idx = i;
        break;
      }
    }

    int32_t need_start = coord.begin;
    for (int i = start_idx; i <= end_idx; ++i) {
      if (used[i].get_begin_position() < coord.begin) {
        need_start = used[i].get_end_position();
        continue;
      }
      if (need_start == used[i].get_begin_position()) {
        need_start = used[i].get_end_position();
        continue;
      }
      Interval interval = Interval(coord.edge, need_start, used[i].get_begin_position());
      need_start = used[i].get_end_position();
      need_filler.push_back(interval);
    }

    if (used[end_idx].get_end_position() < coord.end) {
      Interval interval_end = Interval(coord.edge, used[end_idx].get_end_position(), coord.end);
      need_filler.push_back(interval_end);
    }
  }

  for (Interval fil : need_filler) {
    fillInterval(fil, fillers, prefix, coord);
  }
}

void IoPlacer::fillInterval(Interval interval, std::vector<idb::IdbCellMaster*> fillers, const std::string prefix, PadCoordinate coord)
{
  auto chooseFillerIndex = [](int32_t length, std::vector<idb::IdbCellMaster*> fillers) -> idb::IdbCellMaster* {
    for (size_t i = 0; i != fillers.size(); ++i) {
      if ((ssize_t) (fillers[i]->get_width()) <= (ssize_t) length) {
        return fillers[i];
      }
    }

    return nullptr;
  };

  auto build_inst_name = [](const std::string prefix, PadCoordinate coord, int idx) {
    string inst_name = "";

    switch (coord.orient) {
      case IdbOrient::kN_R0:
        inst_name = prefix + "_" + "S" + "_" + to_string(idx);
        break;
      case IdbOrient::kS_R180:
        inst_name = prefix + "_" + "N" + "_" + to_string(idx);
        break;
      case IdbOrient::kW_R90:
        inst_name = prefix + "_" + "E" + "_" + to_string(idx);
        break;
      case IdbOrient::kE_R270:
        inst_name = prefix + "_" + "W" + "_" + to_string(idx);
        break;
      default:
        inst_name = inst_name.substr(0, inst_name.length() - 2);
    }

    return inst_name;
  };

  int32_t begin_pos = interval.get_begin_position();
  int32_t length = interval.get_interval_length();

  while (length > 0) {
    auto* filler = chooseFillerIndex(length, fillers);
    if (filler == nullptr) {
      printf("IOFiller place error.Please Check IOCELL Place\n");
      printf("Edge is %s. llx or lly is %d\n", transferOrientToString(coord.orient).c_str(), begin_pos);
      return;
    }

    string inst_name = build_inst_name(prefix, coord, _iofiller_idx);

    int32_t llx = coord.orient == IdbOrient::kN_R0 || coord.orient == IdbOrient::kS_R180 ? begin_pos : coord.coord;
    int32_t lly = coord.orient == IdbOrient::kN_R0 || coord.orient == IdbOrient::kS_R180 ? coord.coord : begin_pos;

    dmInst->placeInst(inst_name, llx, lly, transferOrientToString(coord.orient), filler->get_name());

    ++_iofiller_idx;
    begin_pos += filler->get_width();
    length -= filler->get_width();
  }
}

/// calculate pad coordinate begin/end
void IoPlacer::set_pad_coords(vector<string> conner_masters)
{
  auto* idb_design = dmInst->get_idb_design();
  auto* idb_layout = idb_design->get_layout();
  auto* idb_die = idb_layout->get_die();
  auto* inst_list = idb_design->get_instance_list();
  auto* io_site = idb_layout->get_sites()->get_io_site();

  auto corner_list = inst_list->get_corner_list(conner_masters);
  int site_height = io_site->get_height();

  if (corner_list.size() == 0) {
    /// no corner, place pad in to the die edge
    auto die_box = idb_die->get_bounding_box();

    /// bottom
    _pad_coord[0].edge = Edge::kBottom;
    _pad_coord[0].begin = site_height;
    _pad_coord[0].end = die_box->get_width() - site_height;
    _pad_coord[0].coord = 0;
    /// lef
    _pad_coord[1].edge = Edge::kLeft;
    _pad_coord[1].begin = site_height;
    _pad_coord[1].end = die_box->get_height() - site_height;
    _pad_coord[1].coord = 0;
    /// top
    _pad_coord[2].edge = Edge::kTop;
    _pad_coord[2].begin = site_height;
    _pad_coord[2].end = die_box->get_width() - site_height;
    _pad_coord[2].coord = die_box->get_height() - site_height;
    /// right
    _pad_coord[3].edge = Edge::kRight;
    _pad_coord[3].begin = site_height;
    _pad_coord[3].end = die_box->get_height() - site_height;
    _pad_coord[3].coord = die_box->get_width() - site_height;
  } else {
    /// build range for x y
    std::set<int> list_x;
    std::set<int> list_y;
    for (auto* inst : corner_list) {
      auto bounding_box = inst->get_bounding_box();
      list_x.insert(bounding_box->get_low_x());
      list_x.insert(bounding_box->get_high_x());
      list_y.insert(bounding_box->get_low_y());
      list_y.insert(bounding_box->get_high_y());
    }
    std::vector<int> coord_x, coord_y;

    auto iter_x = list_x.begin();
    auto iter_y = list_y.begin();
    for (int i = 0; i < 4; i++) {
      coord_x.push_back(*iter_x++);
      coord_y.push_back(*iter_y++);
    }

    /// bottom
    _pad_coord[0].edge = Edge::kBottom;
    _pad_coord[0].orient = IdbOrient::kN_R0;
    _pad_coord[0].begin = coord_x[1];
    _pad_coord[0].end = coord_x[2];
    _pad_coord[0].coord = coord_y[0];
    /// lef
    _pad_coord[1].edge = Edge::kLeft;
    _pad_coord[1].orient = IdbOrient::kE_R270;
    _pad_coord[1].begin = coord_y[1];
    _pad_coord[1].end = coord_y[2];
    _pad_coord[1].coord = coord_x[0];
    /// top
    _pad_coord[2].edge = Edge::kTop;
    _pad_coord[2].orient = IdbOrient::kS_R180;
    _pad_coord[2].begin = coord_x[1];
    _pad_coord[2].end = coord_x[2];
    _pad_coord[2].coord = coord_y[2];
    /// right
    _pad_coord[3].edge = Edge::kRight;
    _pad_coord[3].orient = IdbOrient::kW_R90;
    _pad_coord[3].begin = coord_y[1];
    _pad_coord[3].end = coord_y[2];
    _pad_coord[3].coord = coord_x[2];
  }
}

/**
 * pad_masters : pad cell name list
 * conner_masters : corner cell name list
 */
bool IoPlacer::autoPlacePad(std::vector<std::string> pad_masters, std::vector<std::string> conner_masters)
{
  auto place_pad = [](std::vector<IdbInstance*>& pad_list, int& index_begin, PadCoordinate& pad_coord, int step) {
    int coord_offset = pad_coord.begin + step;
    for (; index_begin < (int) pad_list.size() && coord_offset < pad_coord.end; index_begin++) {
      int pad_witdh = pad_list[index_begin]->get_cell_master()->get_width();
      if (coord_offset + pad_witdh > pad_coord.end) {
        index_begin--;
        return;
      }

      if (pad_coord.edge == Edge::kBottom || pad_coord.edge == Edge::kTop) {
        int coord_x = coord_offset;
        int coord_y = pad_coord.coord;
        auto orient = pad_coord.orient;

        pad_list[index_begin]->set_coodinate(coord_x, coord_y, false);
        pad_list[index_begin]->set_orient(orient);
        pad_list[index_begin]->set_status_placed();

      } else {
        int coord_x = pad_coord.coord;
        int coord_y = coord_offset;
        auto orient = pad_coord.orient;

        pad_list[index_begin]->set_coodinate(coord_x, coord_y, false);
        pad_list[index_begin]->set_orient(orient);
        pad_list[index_begin]->set_status_placed();
      }

      coord_offset = coord_offset + pad_witdh + step;
    }
  };

  auto* idb_design = dmInst->get_idb_design();
  auto* idb_layout = idb_design->get_layout();
  int io_site_width = idb_layout->get_sites()->get_io_site()->get_width();
  auto* inst_list = idb_design->get_instance_list();

  auto pad_list = inst_list->get_iopad_list(pad_masters);
  if (pad_list.size() <= 0) {
    return false;
  }

  set_pad_coords(conner_masters);

  /// calculate average interval between pads
  int range_total_len = 0;
  for (int i = 0; i < 4; i++) {
    range_total_len += (_pad_coord[i].end - _pad_coord[i].begin);
  }
  int pad_total_len = 0;
  for (auto* inst : pad_list) {
    pad_total_len += inst->get_cell_master()->get_width();
  }
  int site_step = (range_total_len - pad_total_len) / (pad_list.size() + 8) / io_site_width * io_site_width;

  int pad_index = 0;
  for (int i = 0; i < 4; i++) {
    place_pad(pad_list, pad_index, _pad_coord[i], site_step);
  }

  return true;
}

bool IoPlacer::autoIOFiller(std::vector<std::string> filler_name_list, std::string prefix)
{
  auto* idb_design = dmInst->get_idb_design();
  auto* idb_layout = idb_design->get_layout();
  auto* idb_cell_masters = idb_layout->get_cell_master_list();
  if (idb_cell_masters == nullptr) {
    std::cout << "Error : cell master not exist!" << std::endl;
    return false;
  }

  auto pad_fillers = idb_cell_masters->getIOFillers(filler_name_list);
  if (pad_fillers.size() <= 0) {
    return false;
  }
  /// sort pad filler
  sort(pad_fillers.begin(), pad_fillers.end(),
       [](idb::IdbCellMaster* fill_1, idb::IdbCellMaster* fill_2) { return fill_1->get_width() > fill_2->get_width(); });

  set_pad_coords();

  for (int i = 0; i < 4; i++) {
    placeIOFiller(pad_fillers, prefix, _pad_coord[i]);
  }

  return true;
}

}  // namespace ifp