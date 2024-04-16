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

int32_t IoPlacer::chooseFillerIndex(int32_t length, std::vector<idb::IdbCellMaster*> fillers)
{
  for (size_t i = 0; i != fillers.size(); ++i) {
    if ((ssize_t) (fillers[i]->get_width()) <= (ssize_t) length) {
      return i;
    }
  }
  return -1;
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
  int width_step = idb_core->get_bounding_box()->get_width() / (edge_num + 1);
  int height_step = idb_core->get_bounding_box()->get_height() / (edge_num + 1);

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
    shape->add_rect(-(width / 2), -(height / 2), width / 2, height / 2);
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
    shape->add_rect(-(width / 2), -(height / 2), width / 2, height / 2);
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
    shape->add_rect(-(width / 2), -(height / 2), width / 2, height / 2);
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
    shape->add_rect(-(width / 2), -(height / 2), width / 2, height / 2);
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

bool IoPlacer::placeIOFiller(std::vector<std::string> filler_name_list, std::string prefix, std::string orient, double begin, double end,
                             std::string source)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();
  auto idb_die = idb_layout->get_die();

  int32_t dbu = idb_layout->get_units()->get_micron_dbu();
  double die_llx = ((double) idb_die->get_llx()) / dbu;
  double die_lly = ((double) idb_die->get_lly()) / dbu;
  double die_urx = ((double) idb_die->get_urx()) / dbu;
  double die_ury = ((double) idb_die->get_ury()) / dbu;

  if (orient.empty()) {
    placeIOFiller(filler_name_list, prefix, Edge::kBottom, die_llx, die_urx, source);
    placeIOFiller(filler_name_list, prefix, Edge::kTop, die_llx, die_urx, source);
    placeIOFiller(filler_name_list, prefix, Edge::kLeft, die_lly, die_ury, source);
    placeIOFiller(filler_name_list, prefix, Edge::kRight, die_lly, die_ury, source);
    return true;
  }

  double begin_new = begin;
  double end_new = end;
  Edge edge;
  if (orient == "bottom") {
    edge = Edge::kBottom;
    begin_new = die_llx;
    end_new = die_urx;
  } else if (orient == "top") {
    edge = Edge::kTop;
    begin_new = die_llx;
    end_new = die_urx;
  } else if (orient == "left") {
    edge = Edge::kLeft;
    begin_new = die_lly;
    end_new = die_ury;
  } else {
    edge = Edge::kRight;
    begin_new = die_lly;
    end_new = die_ury;
  }

  placeIOFiller(filler_name_list, prefix, edge, begin_new, end_new, source);

  return true;
}

void IoPlacer::placeIOFiller(std::vector<std::string> filler_names, const std::string prefix, Edge edge, double begin_pos, double end_pos,
                             std::string source)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();
  auto idb_inst_list = idb_design->get_instance_list();
  auto idb_io_inst_list = idb_inst_list->get_io_cell_list();
  auto idb_die = idb_layout->get_die();
  auto idb_cellmaster_list = idb_layout->get_cell_master_list();

  _iofiller_idx = -1;
  vector<idb::IdbCellMaster*> fillers;

  int32_t dbu = idb_layout->get_units()->get_micron_dbu();
  int32_t iocell_height = idb_io_inst_list[0]->get_cell_master()->get_height();

  int32_t die_llx = idb_die->get_llx();
  int32_t die_lly = idb_die->get_lly();
  int32_t die_urx = idb_die->get_urx();
  int32_t die_ury = idb_die->get_ury();
  int32_t begin_position = 0, end_position = 0;

  begin_position = dbu * begin_pos;
  end_position = dbu * end_pos;
  if (edge == Edge::kLeft || edge == Edge::kRight) {
    if (end_position > die_ury - iocell_height) {
      end_position = die_ury - iocell_height;
    }
    if (begin_position < iocell_height + die_lly) {
      begin_position = iocell_height + die_lly;
    }
  } else if (edge == Edge::kBottom || edge == Edge::kTop) {
    if (end_position > die_urx - iocell_height) {
      end_position = die_urx - iocell_height;
    }
    if (begin_position < iocell_height + die_llx) {
      begin_position = iocell_height + die_llx;
    }
  }

  for (std::string fill : filler_names) {
    auto filler = idb_cellmaster_list->find_cell_master(fill);
    fillers.push_back(filler);
  }
  sort(fillers.begin(), fillers.end(),
       [](idb::IdbCellMaster* fill_fir, idb::IdbCellMaster* fill_sec) { return fill_fir->get_width() > fill_sec->get_width(); });

  vector<Interval> used;
  vector<Interval> need_filler;
  for (auto io : idb_io_inst_list) {
    if ((io->is_placed() || io->is_fixed() || io->is_cover()) && edgeIsSameToOrient(edge, io->get_orient())) {
      int32_t width = io->get_cell_master()->get_width();
      io->set_bounding_box();
      int32_t llx = io->get_bounding_box()->get_low_x();
      int32_t lly = io->get_bounding_box()->get_low_y();
      Interval inter = Interval();
      if (edge == Edge::kLeft || edge == Edge::kRight) {
        inter.set_edge(edge);
        inter.set_begin_position(lly);
        inter.set_end_position(lly + width);
      } else {
        inter.set_edge(edge);
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
    need_filler.push_back(Interval(edge, begin_position, end_position));
  } else {
    sort(used.begin(), used.end(), [](Interval a, Interval b) { return a.get_begin_position() < b.get_begin_position(); });

    int32_t start_idx = 0, end_idx = 0;
    for (size_t i = 0; i != used.size(); ++i) {
      if (used[i].get_end_position() > begin_position) {
        start_idx = i;
        break;
      }
    }
    for (ssize_t i = used.size() - 1; i != -1; --i) {
      if (used[i].get_begin_position() < end_position) {
        end_idx = i;
        break;
      }
    }

    int32_t need_start = begin_position;
    for (int i = start_idx; i <= end_idx; ++i) {
      if (used[i].get_begin_position() < begin_position) {
        need_start = used[i].get_end_position();
        continue;
      }
      if (need_start == used[i].get_begin_position()) {
        need_start = used[i].get_end_position();
        continue;
      }
      Interval interval = Interval(edge, need_start, used[i].get_begin_position());
      need_start = used[i].get_end_position();
      need_filler.push_back(interval);
    }

    if (used[end_idx].get_end_position() < end_position) {
      Interval interval_end = Interval(edge, used[end_idx].get_end_position(), end_position);
      need_filler.push_back(interval_end);
    }
  }

  for (Interval fil : need_filler) {
    fillInterval(fil, fillers, prefix, source);
  }
}

void IoPlacer::fillInterval(Interval interval, std::vector<idb::IdbCellMaster*> fillers, const std::string prefix, std::string source)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();
  auto idb_die = idb_layout->get_die();

  idb::IdbOrient orient = transferEdgeToOrient(interval.get_edge());
  int32_t begin_pos = interval.get_begin_position();
  int32_t length = interval.get_interval_length();

  int32_t die_llx = idb_die->get_llx();
  int32_t die_urx = idb_die->get_urx();
  int32_t die_lly = idb_die->get_lly();
  int32_t die_ury = idb_die->get_ury();
  int32_t llx = 0, lly = 0;
  int32_t width, height;
  while (length > 0) {
    int32_t index = chooseFillerIndex(length, fillers);
    if (index < 0) {
      printf("IOFiller place error.Please Check IOCELL Place\n");
      printf("Edge is %s. llx or lly is %d\n", transferOrientToString(orient).c_str(), begin_pos);
      return;
    }
    width = fillers[index]->get_width();
    height = fillers[index]->get_height();
    // string inst_name, int32_t x, int32_t y, string orient_name,string
    // cell_master_name
    string inst_name;

    if (orient == IdbOrient::kN_R0) {
      llx = begin_pos;
      lly = die_lly;
      inst_name = prefix + "_" + "S" + "_" + to_string(_iofiller_idx);
    }
    if (orient == IdbOrient::kS_R180) {
      llx = begin_pos;
      lly = die_ury - height;
      inst_name = prefix + "_" + "N" + "_" + to_string(_iofiller_idx);
    }
    if (orient == IdbOrient::kW_R90) {
      llx = die_urx - height;
      lly = begin_pos;
      inst_name = prefix + "_" + "E" + "_" + to_string(_iofiller_idx);
    }
    if (orient == IdbOrient::kE_R270) {
      llx = die_llx;
      lly = begin_pos;
      inst_name = prefix + "_" + "W" + "_" + to_string(_iofiller_idx);
    }
    if (_iofiller_idx == -1) {
      inst_name = inst_name.substr(0, inst_name.length() - 2);
    }

    dmInst->placeInst(inst_name, llx, lly, transferOrientToString(orient), fillers[index]->get_name(), source);

    ++_iofiller_idx;
    begin_pos += width;
    length -= width;
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
    _pad_coord[0].begin = coord_x[1];
    _pad_coord[0].end = coord_x[2];
    _pad_coord[0].coord = coord_y[0];
    /// lef
    _pad_coord[1].edge = Edge::kLeft;
    _pad_coord[1].begin = coord_y[1];
    _pad_coord[1].end = coord_y[2];
    _pad_coord[1].coord = coord_x[0];
    /// top
    _pad_coord[2].edge = Edge::kTop;
    _pad_coord[2].begin = coord_x[1];
    _pad_coord[2].end = coord_x[2];
    _pad_coord[2].coord = coord_y[2];
    /// right
    _pad_coord[3].edge = Edge::kRight;
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
        auto orient = pad_coord.edge == Edge::kBottom ? IdbOrient::kN_R0 : IdbOrient::kS_R180;

        pad_list[index_begin]->set_coodinate(coord_x, coord_y, false);
        pad_list[index_begin]->set_orient(orient);
        pad_list[index_begin]->set_status_placed();

      } else {
        int coord_x = pad_coord.coord;
        int coord_y = coord_offset;
        auto orient = pad_coord.edge == Edge::kLeft ? IdbOrient::kE_R270 : IdbOrient::kW_R90;

        pad_list[index_begin]->set_coodinate(coord_x, coord_y, false);
        pad_list[index_begin]->set_orient(orient);
        pad_list[index_begin]->set_status_placed();
      }

      coord_offset = coord_offset + pad_witdh + step;
    }
  };

  auto* idb_design = dmInst->get_idb_design();
  auto* idb_layout = idb_design->get_layout();
  auto* inst_list = idb_design->get_instance_list();

  auto pad_list = inst_list->get_iopad_list(pad_masters);
  if (pad_list.size() <= 0) {
    return false;
  }

  set_pad_coords(conner_masters);

  int range_total_len = 0;
  for (int i = 0; i < 4; i++) {
    range_total_len += (_pad_coord[i].end - _pad_coord[i].begin);
  }

  /// calculate average interval between pads
  int pad_total_len = 0;
  for (auto* inst : pad_list) {
    pad_total_len += inst->get_cell_master()->get_width();
  }

  int site_step = (range_total_len - pad_total_len) / (pad_list.size() + 8);

  int pad_index = 0;
  /// bottom = kN_R0
  place_pad(pad_list, pad_index, _pad_coord[0], site_step);

  /// left = kE_R270
  place_pad(pad_list, pad_index, _pad_coord[1], site_step);

  /// top = kS_R180
  place_pad(pad_list, pad_index, _pad_coord[2], site_step);

  /// right = kW_R90
  place_pad(pad_list, pad_index, _pad_coord[3], site_step);

  return true;
}

bool IoPlacer::autoIOFiller(std::vector<std::string> filler_name_list, std::string prefix)
{
  auto* idb_design = dmInst->get_idb_design();
  auto* idb_layout = idb_design->get_layout();

  auto pad_fillers = idb_layout->get_cell_master_list()->getIOFillers(filler_name_list);
  if (pad_fillers.size() <= 0) {
    return false;
  }
  /// sort pad filler
  sort(pad_fillers.begin(), pad_fillers.end(),
       [](idb::IdbCellMaster* fill_1, idb::IdbCellMaster* fill_2) { return fill_1->get_width() > fill_2->get_width(); });

  return true;
}

}  // namespace ifp