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
#include "idbsetup.h"
IdbSetup::IdbSetup(const std::vector<std::string>& lef_paths, const std::string& def_path, GuiGraphicsScene* scene)
    : DbSetup(lef_paths, def_path, scene) {
  set_type(DbSetupType::kDetailPlace);
  initDB();
  createChip();
  //   TestGrGui();
}

IdbSetup::IdbSetup(IdbBuilder* idb_builder, GuiGraphicsScene* scene) : DbSetup(scene) {
  _db_builder = idb_builder;
  set_type(DbSetupType::kDetailRouting);

  _design = _db_builder->get_def_service()->get_design();
  _layout = _db_builder->get_def_service()->get_layout();
  createChip();
}

IdbSetup::~IdbSetup() {
  if (_db_builder != nullptr) {
    delete _db_builder;
    _db_builder = nullptr;
  }

  if (_db_builder != nullptr) {
    delete _db_builder;
    _db_builder = nullptr;
  }

  if (_db_builder != nullptr) {
    delete _db_builder;
    _db_builder = nullptr;
  }
}

// GuiItem::Orientation IdbSetup::orientationTransform(IdbOrient orient_type)
//{
//    switch (orient_type) {
//    case IdbOrient::kN_R0:
//        return GuiItem::R0;
//    case IdbOrient::kW_R90:
//        return GuiItem::M90;
//    case IdbOrient::kS_R180:
//        return GuiItem::R180;
//    case IdbOrient::kE_R270:
//        return GuiItem::R270;
//    case IdbOrient::kFN_MY:
//        return GuiItem::MY;
//    case IdbOrient::kFW_MX90:
//        return GuiItem::MY90;
//    case IdbOrient::kFS_MX:
//        return GuiItem::MX;
//    case IdbOrient::kFE_MY90:
//        return GuiItem::MX90;
//    default:
//        return GuiItem::NoOrientation;
//    }
//}

void IdbSetup::initDB() {
  _db_builder = new IdbBuilder();
  _db_builder->buildLef(_lef_paths);

  DbSetupType type = get_type();
  //   _db_builder->buildDef(def_file);
  switch (type) {
    case DbSetupType::kChip: _db_builder->buildDef(_def_path); break;
    case DbSetupType::kFloorplan: _db_builder->buildDefFloorplan(_def_path); break;
    case DbSetupType::kGlobalPlace: _db_builder->buildDef(_def_path); break;
    case DbSetupType::kDetailPlace: _db_builder->buildDef(_def_path); break;
    case DbSetupType::kGlobalRouting: _db_builder->buildDef(_def_path); break;
    case DbSetupType::kDetailRouting: _db_builder->buildDef(_def_path); break;
    default: break;
  }

  _design = _db_builder->get_def_service()->get_design();
  _layout = _db_builder->get_lef_service()->get_layout();
}

/// fit the design visible to view
void IdbSetup::fitView(double width, double height) {
  double this_width  = width;
  double this_height = height;
  if (width == 0 || height == 0) {
    IdbDie* db_die = _layout->get_die();
    if (db_die == nullptr) {
      return;
    }

    this_width  = _transform.db_to_guidb(_layout->get_die()->get_width());
    this_height = _transform.db_to_guidb(_layout->get_die()->get_height());
  }

  DbSetup::fitView(this_width, this_height);
}

void IdbSetup::createChip() {
  std::cout << "Begin to create chip..." << std::endl;
  DbSetupType type = get_type();
  switch (type) {
    case DbSetupType::kChip:
      createDbu();
      createDie();
      createCore();
      createIO();
      createRow();
      createInstance();
      createSpecialNet();
      createNet();
      break;
    case DbSetupType::kFloorplan:
      createDbu();
      createDie();
      createCore();
      createIO();
      createRow();
      createInstance();
      //   createSpecialNet();
      //   createNet();
      break;
    case DbSetupType::kGlobalPlace:
      createDbu();
      createDie();
      createCore();
      createIO();
      createRow();
      createInstance();
      //   createSpecialNet();
      //   createNet();
      break;
    case DbSetupType::kDetailPlace:
      createDbu();
      createDie();
      createCore();
      createIO();
      createRow();
      createInstance();
      // createSpecialNet();
      //   createNet();
      break;
    case DbSetupType::kGlobalRouting:
      createDbu();
      createDie();
      createCore();
      createIO();
      createRow();
      createInstance();
      //   createSpecialNet();
      //   createNet();
      break;
    case DbSetupType::kDetailRouting:
      createDbu();
      createDie();
      createCore();
      createIO();
      createRow();
      createInstance();
      createSpecialNet();
      createNet();
      break;
    default: break;
  }

  fitView();

  std::cout << "Success to create chip..." << std::endl;
}

void IdbSetup::createDbu() {
  std::cout << "Start to create DBU..." << std::endl;
  IdbUnits* db_unit = _layout->get_units();
  _unit             = db_unit != nullptr ? db_unit->get_micron_dbu() : 1000;

  _transform.set_micron(_unit);
  std::cout << "Success to create DBU..." << std::endl;
}

void IdbSetup::createDie() {
  std::cout << "Start to create DIE..." << std::endl;
  IdbDie* db_die = _layout->get_die();
  _transform.set_die_height(db_die->get_height());

  GuiDie* die = new GuiDie();
  die->set_rect(QRectF(_transform.db_to_guidb(db_die->get_llx()), _transform.db_to_guidb(db_die->get_lly()),
                       _transform.db_to_guidb(db_die->get_width()), _transform.db_to_guidb(db_die->get_height())));
  die->setZValue(0);

  addItem(die);
  std::cout << "Success to create DIE..." << std::endl;
}

void IdbSetup::createCore() {
  std::cout << "Start to create CORE..." << std::endl;
  IdbCore* db_core = _layout->get_core();

  GuiCore* gui_core = new GuiCore();
  gui_core->set_rect(_transform.db_to_guidb_rect(db_core->get_bounding_box()));
  gui_core->setZValue(0);
  addItem(gui_core);
  std::cout << "Success to create CORE..." << std::endl;
}

void IdbSetup::createIO() {
  std::cout << "Start to create IO..." << std::endl;

  IdbRect* rect      = _layout->get_core()->get_bounding_box();
  int32_t row_height = _design->get_io_pin_list()->getIOPortWidth();

  IdbPins* pins = _design->get_io_pin_list();
  for (IdbPin* pin : pins->get_pin_list()) {
    IdbTerm* term = pin->get_term();
    if (term->is_port_exist()) {
      for (IdbPort* port : pin->get_term()->get_port_list()) {
        GuiPin* gui_pin                    = new GuiPin();
        IdbCoordinate<int32_t>* coordinate = port->get_io_average_coordinate();
        IdbConnectDirection direction      = term->get_direction();
        RectEdgePosition rect_edge         = rect->findCoordinateEdgePosition(*coordinate);

        gui_pin->set_IOPin(_transform.db_to_guidb(coordinate->get_x()), _transform.db_to_guidb_rotate(coordinate->get_y()),
                           _transform.db_to_guidb(row_height * 0.8), direction, rect_edge);
        gui_pin->setZValue(port->findZOrderTop());

        // gui_pin->setFlag(QGraphicsItem::ItemIsSelectable, true);

        addItem(gui_pin);
      }
    } else {
      GuiPin* gui_pin                    = new GuiPin();
      IdbCoordinate<int32_t>* coordinate = pin->get_average_coordinate();
      IdbConnectDirection direction      = term->get_direction();
      RectEdgePosition rect_edge         = rect->findCoordinateEdgePosition(*coordinate);

      gui_pin->set_IOPin(_transform.db_to_guidb(coordinate->get_x()), _transform.db_to_guidb_rotate(coordinate->get_y()),
                         _transform.db_to_guidb(row_height * 0.8), direction, rect_edge);
      uint8_t z_order = term->get_port_number() > 0 ? term->get_top_order() : 0;
      gui_pin->setZValue(z_order);

      gui_pin->setFlag(QGraphicsItem::ItemIsSelectable, true);

      addItem(gui_pin);
      //    gui_pin->setFlag(QGraphicsItem::ItemIgnoresTransformations,true);
    }
  }

  createIOPinPortShape(pins->get_pin_list());

  std::cout << "Success to create IO..." << std::endl;
}

void IdbSetup::createIOPinPortShape(vector<IdbPin*>& pin_list) {
  for (IdbPin* pin : pin_list) {
    if (pin != nullptr && pin->get_term()->is_placed()) {
      for (IdbLayerShape* layer_shape : pin->get_port_box_list()) {
        createLayerShape(*layer_shape);
      }
    }
  }
}

void IdbSetup::createRow() {
  std::cout << "Begin to create Row..." << std::endl;
  IdbRows* rows            = _layout->get_rows();
  vector<IdbRow*> row_list = rows->get_row_list();

  for (IdbRow* db_row : row_list) {
    GuiRow* gui_row = new GuiRow();

    gui_row->set_rect(_transform.db_to_guidb_rect(db_row->get_bounding_box()));
    gui_row->setZValue(0);

    addItem(gui_row);
  }

  std::cout << "Success to create Row..." << std::endl;
}

void IdbSetup::createInstancePin(vector<IdbPin*>& pin_list, GuiInstance* gui_instance) {
  for (IdbPin* pin : pin_list) {
    if (pin->get_term()->get_name() == "VDD" || pin->get_term()->get_name() == "VSS")
      continue;

    if (pin != nullptr && pin->get_term()->get_port_number() > 0) {
      if (pin->get_instance()->get_cell_master()->is_core()) {
        IdbCoordinate<int32_t>* coordinate = pin->get_grid_coordinate();
        gui_instance->add_pin(_transform.db_to_guidb(coordinate->get_x()),
                              _transform.db_to_guidb_rotate(coordinate->get_y()),
                              _transform.db_to_guidb(_layout->get_rows()->get_row_height() / 20));

      } else {
        IdbCoordinate<int32_t>* coordinate = pin->get_average_coordinate();
        gui_instance->add_pin(_transform.db_to_guidb(coordinate->get_x()),
                              _transform.db_to_guidb_rotate(coordinate->get_y()),
                              _transform.db_to_guidb(_layout->get_rows()->get_row_height() / 20));
      }
    }
  }
}

void IdbSetup::createPinPortShape(vector<IdbPin*>& pin_list) {
  for (IdbPin* pin : pin_list) {
    //    if (pin->get_term()->get_name() == "VDD"
    //        || pin->get_term()->get_name() == "VSS")
    //      continue;

    if (pin != nullptr && pin->get_term()->is_instance_pin()) {
      for (IdbLayerShape* layer_shape : pin->get_port_box_list()) {
        createLayerShape(*layer_shape);
      }
    }
  }
}

void IdbSetup::createInstance() {
  std::cout << "Begin to create Instance..." << std::endl;
  for (IdbInstance* instance : _design->get_instance_list()->get_instance_list()) {
    createInstanceCore(instance);
    createInstancePad(instance);
    createInstanceBlock(instance);
  }

  std::cout << "Success to create Instance..." << std::endl;
}

void IdbSetup::createInstanceCore(IdbInstance* instance) {
  IdbCellMaster* cell_master = instance->get_cell_master();

  if ((cell_master->is_core() && (!is_floorplan())) ||
      ((is_floorplan() && (cell_master->is_core_filler() || cell_master->is_endcap())))) {
    IdbRect* bounding_box        = instance->get_bounding_box();
    GuiStandardCell* gui_stdCell = new GuiStandardCell();

    gui_stdCell->set_rect(_transform.db_to_guidb_rect(bounding_box), instance->get_orient());
    gui_stdCell->setZValue(0);
    gui_stdCell->setFlag(QGraphicsItem::ItemIsSelectable, true);

    addItem(gui_stdCell);

    createInstancePin(instance->get_pin_list()->get_pin_list(), gui_stdCell);

    //   createPinPortShape(instance->get_pin_list()->get_pin_list());
  }
}

void IdbSetup::createInstancePad(IdbInstance* instance) {
  IdbCellMaster* cell_master = instance->get_cell_master();
  if (cell_master->is_pad()) {
    IdbRect* bounding_box = instance->get_bounding_box();
    GuiPad* gui_pad       = new GuiPad();
    gui_pad->set_rect(_transform.db_to_guidb_rect(bounding_box), instance->get_orient());

    addItem(gui_pad);

    createInstancePin(instance->get_pin_list()->get_pin_list(), gui_pad);

    createPinPortShape(instance->get_pin_list()->get_pin_list());
  }
}

void IdbSetup::createInstanceBlock(IdbInstance* instance) {
  IdbCellMaster* cell_master = instance->get_cell_master();
  if (cell_master->is_block()) {
    IdbRect* bounding_box = instance->get_bounding_box();

    GuiBlock* gui_block = new GuiBlock();
    gui_block->set_rect(_transform.db_to_guidb_rect(bounding_box), instance->get_orient());

    IdbHalo* halo = instance->get_halo();
    if (halo != nullptr) {
      gui_block->set_halo_rect(_transform.db_to_guidb_rect(halo->get_bounding_box()));
    }

    addItem(gui_block);

    createInstancePin(instance->get_pin_list()->get_pin_list(), gui_block);

    createPinPortShape(instance->get_pin_list()->get_pin_list());
  }
}

void IdbSetup::createLayerShape(IdbLayerShape& layer_shape) {
  if (layer_shape.is_via()) {
    for (IdbRect* cut_rect : layer_shape.get_rect_list()) {
      if (layer_shape.get_layer() == nullptr) {
        std::cout << "Error...createLayerShape : Via Layer not exist :  " << std::endl;
        return;
      }

      GuiVia* via = new GuiVia();
      via->set_rect(_transform.db_to_guidb_rect(cut_rect));
      via->set_layer(layer_shape.get_layer()->get_name());
      via->setZValue(layer_shape.get_layer()->get_order());

      addItem(via);
    }
  } else {
    for (IdbRect* rect : layer_shape.get_rect_list()) {
      if (layer_shape.get_layer() == nullptr) {
        std::cout << "Error...createLayerShape : Rect Layer not exist :  " << std::endl;
        return;
      }
      GuiPower* gui_rect = new GuiPower();
      gui_rect->set_rect(_transform.db_to_guidb_rect(rect));
      gui_rect->set_layer(layer_shape.get_layer()->get_name());
      gui_rect->setZValue(layer_shape.get_layer()->get_order());

      addItem(gui_rect);
    }
  }
}

void IdbSetup::createSpecialNetVia(IdbSpecialWireSegment* segment) {
  if (segment->is_via()) {
    IdbVia* via                      = segment->get_via();
    IdbLayerShape bottom_layer_shape = via->get_bottom_layer_shape();
    createLayerShape(bottom_layer_shape);
    IdbLayerShape cut_layer_shape = via->get_cut_layer_shape();
    createLayerShape(cut_layer_shape);
    IdbLayerShape top_layer_shape = via->get_top_layer_shape();
    createLayerShape(top_layer_shape);
  }
}
void IdbSetup::createSpecialNetPoints(IdbSpecialWireSegment* segment) {
  /// ensure there are 2 point in a segment
  if (segment->get_point_list().size() >= 2) {
    GuiPower* power = new GuiPower();

    IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(segment->get_layer());
    int32_t routing_width = segment->get_route_width() == 0 ? routing_layer->get_width() : segment->get_route_width();

    IdbCoordinate<int32_t>* point_1 = segment->get_point_start();
    IdbCoordinate<int32_t>* point_2 = segment->get_point_second();

    int32_t ll_x = 0;
    int32_t ll_y = 0;
    int32_t ur_x = 0;
    int32_t ur_y = 0;

    /// conpasate the 1/2 width for segment
    // if (point_1->get_y() == point_2->get_y()) {
    //   // horizontal
    //   ll_x = std::min(point_1->get_x(), point_2->get_x()) - routing_width /
    //   2; ll_y = std::min(point_1->get_y(), point_2->get_y()) - routing_width
    //   / 2; ur_x = std::max(point_1->get_x(), point_2->get_x()) +
    //   routing_width / 2; ur_y = ll_y + routing_width;
    // } else if (point_1->get_x() == point_2->get_x()) {
    //   // vertical
    //   ll_x = std::min(point_1->get_x(), point_2->get_x()) - routing_width /
    //   2; ll_y = std::min(point_1->get_y(), point_2->get_y()) - routing_width
    //   / 2; ur_x = ll_x + routing_width; ur_y = std::max(point_1->get_y(),
    //   point_2->get_y()) + routing_width / 2;
    // }
    /// do not conpasate the 1/2 width for segment
    if (point_1->get_y() == point_2->get_y()) {
      // horizontal
      ll_x = std::min(point_1->get_x(), point_2->get_x());
      ll_y = std::min(point_1->get_y(), point_2->get_y()) - routing_width / 2;
      ur_x = std::max(point_1->get_x(), point_2->get_x());
      ur_y = ll_y + routing_width;
    } else if (point_1->get_x() == point_2->get_x()) {
      // vertical
      ll_x = std::min(point_1->get_x(), point_2->get_x()) - routing_width / 2;
      ll_y = std::min(point_1->get_y(), point_2->get_y());
      ur_x = ll_x + routing_width;
      ur_y = std::max(point_1->get_y(), point_2->get_y());
    }

    IdbRect* rect = new IdbRect(ll_x, ll_y, ur_x, ur_y);
    power->set_rect(_transform.db_to_guidb_rect(rect));

    power->set_layer(routing_layer->get_name());
    power->setZValue(routing_layer->get_order());

    addItem(power);
    delete rect;
    rect = nullptr;
  } else {
    // std::cout << "Error...Power segment only use layer, layer = "
    //           << segment->get_layer()->get_name() << std::endl;
  }
}

void IdbSetup::createSpecialNet() {
  std::cout << "Begin to create PDN..." << std::endl;

  IdbSpecialNetList* special_net_list = _design->get_special_net_list();
  int number                          = 0;
  for (IdbSpecialNet* special_net : special_net_list->get_net_list()) {
    for (IdbSpecialWire* special_wire : special_net->get_wire_list()->get_wire_list()) {
      for (IdbSpecialWireSegment* segment : special_wire->get_segment_list()) {
        if (segment->is_via()) {
          continue;
          createSpecialNetVia(segment);
          number++;
        } else {
          createSpecialNetPoints(segment);
          number++;
        }

        if (number % 10000 == 0) {
          std::cout << "-";
        }
      }
    }
  }

  std::cout << std::endl << "Success to create PDN... Total number = " << number << std::endl;
}

void IdbSetup::createNetVia(IdbRegularWireSegment* segment) {
  for (IdbVia* via : segment->get_via_list()) {
    IdbLayerShape bottom_layer_shape = via->get_bottom_layer_shape();
    createLayerShape(bottom_layer_shape);
    IdbLayerShape cut_layer_shape = via->get_cut_layer_shape();
    createLayerShape(cut_layer_shape);
    IdbLayerShape top_layer_shape = via->get_top_layer_shape();
    createLayerShape(top_layer_shape);
  }
}

void IdbSetup::createNetRect(IdbRegularWireSegment* segment) {
  IdbCoordinate<int32_t>* coordinate = segment->get_point_start();
  IdbRect* rect_delta                = segment->get_delta_rect();

  if (coordinate->get_x() < 0 || coordinate->get_y() < 0) {
    std::cout << "Error...Coordinate error...x = " << coordinate->get_x() << " y = " << coordinate->get_y() << std::endl;
  }

  IdbLayer* layer = segment->get_layer();
  if (layer == nullptr) {
    std::cout << "Error...createNetRect : Layer not exist :  " << std::endl;
    return;
  }

  GuiPower* power = new GuiPower();
  IdbRect* rect   = new IdbRect(rect_delta);
  rect->moveByStep(coordinate->get_x(), coordinate->get_y());
  power->set_rect(_transform.db_to_guidb_rect(rect));
  power->set_layer(layer->get_name());
  power->setZValue(layer->get_order());

  addItem(power);
  delete rect;
  rect = nullptr;
}

void IdbSetup::createNetPoints(IdbRegularWireSegment* segment) {
  if (segment->get_point_list().size() >= 2)  // ensure the point number >= 2
  {
    if (segment->get_layer() == nullptr) {
      std::cout << "Error...createNetPoints : Layer not exist :  " << std::endl;
      return;
    }

    IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(segment->get_layer());
    int32_t routing_width          = routing_layer->get_width();

    IdbCoordinate<int32_t>* point_1 = segment->get_point_start();
    IdbCoordinate<int32_t>* point_2 = segment->get_point_second();

    int32_t ll_x = 0;
    int32_t ll_y = 0;
    int32_t ur_x = 0;
    int32_t ur_y = 0;
    if (point_1->get_y() == point_2->get_y()) {
      // horizontal
      ll_x = std::min(point_1->get_x(), point_2->get_x()) - routing_width / 2;
      ll_y = std::min(point_1->get_y(), point_2->get_y()) - routing_width / 2;
      ur_x = std::max(point_1->get_x(), point_2->get_x()) + routing_width / 2;
      ur_y = ll_y + routing_width;
    } else if (point_1->get_x() == point_2->get_x()) {
      // vertical
      ll_x = std::min(point_1->get_x(), point_2->get_x()) - routing_width / 2;
      ll_y = std::min(point_1->get_y(), point_2->get_y()) - routing_width / 2;
      ur_x = ll_x + routing_width;
      ur_y = std::max(point_1->get_y(), point_2->get_y()) + routing_width / 2;
    } else {
      // only support horizontal & vertical direction
      std::cout << "Error...Regular segment only support horizontal & "
                   "vertical direction... "
                << segment->get_layer()->get_name() << std::endl;
    }

    IdbRect* rect   = new IdbRect(ll_x, ll_y, ur_x, ur_y);
    GuiPower* power = new GuiPower();
    power->set_rect(_transform.db_to_guidb_rect(rect));
    power->set_layer(routing_layer->get_name());
    power->setZValue(routing_layer->get_order());

    addItem(power);
    delete rect;
    rect = nullptr;
  } else {
    // std::cout << "Error...Regular segment only use layer, layer = "
    //           << segment->get_layer()->get_name() << std::endl;
  }
}

void IdbSetup::createNet() {
  std::cout << "Begin to create NET..." << std::endl;

  IdbNetList* net_list = _design->get_net_list();
  for (IdbNet* net : net_list->get_net_list()) {
    for (IdbRegularWire* wire : net->get_wire_list()->get_wire_list()) {
      for (IdbRegularWireSegment* segment : wire->get_segment_list()) {
        if (segment->is_via()) {
          createNetVia(segment);
          createNetPoints(segment);
        } else if (segment->is_rect()) {
          createNetRect(segment);
        } else {
          createNetPoints(segment);
        }
      }
    }
  }

  std::cout << "Success to create NET..." << std::endl;
}

int32_t IdbSetup::getLayerCount() {
  IdbLayers* layer_list = _layout->get_layers();
  return layer_list->get_layers_num();
}

std::vector<std::string> IdbSetup::getLayer() {
  IdbLayers* layers = _layout->get_layers();

  return layers->get_all_layer_name();
}

void IdbSetup::TestGrGui() {
  vector<GrInfo> info_list;
  //   for (int i = 0; i < 1000; i++) {
  //     for (int j = 0; j < 1000; j++) {
  //       GrInfo gr_info;
  //       gr_info.x          = i * 100;
  //       gr_info.y          = j * 100;
  //       gr_info.layer_name = "Metal" + std::to_string(i % 5 + 1);

  //       gr_info._h_supply   = double(i) / 1000;
  //       gr_info._v_supply   = double(j) / 1000;
  //       gr_info._h_demand   = double(i) / 1000;
  //       gr_info._v_demand   = double(j) / 1000;
  //       gr_info._via_demand = double(i) / 1000;
  //       info_list.emplace_back(gr_info);
  //     }
  //   }
  analysisResource("/home/zengzhisheng/iEDA/build/space_resource.txt", info_list);

  createGrGui(info_list);
}

void IdbSetup::createGrGui(vector<GrInfo>& info_list) {
  createDbu();
  IdbDie* db_die = _layout->get_die();
  _transform.set_die_height(db_die->get_height());
  createGrCongestionMap(info_list);
}
/// Test
void IdbSetup::createGrCongestionMap(vector<GrInfo>& info_list) {
  int n = 0;
  for (GrInfo gr_info : info_list) {
    GuiGrRect* rect        = new GuiGrRect();
    std::string coordinate = std::to_string(gr_info.x) + " " + std::to_string(gr_info.y);
    rect->set_item_info(coordinate);
    rect->add_info("h_s: " + std::to_string(gr_info._h_supply));
    rect->add_info("v_s: " + std::to_string(gr_info._v_supply));
    rect->add_info("h_d: " + std::to_string(gr_info._h_demand));
    rect->add_info("v_d: " + std::to_string(gr_info._v_demand));
    rect->add_info("via: " + std::to_string(gr_info._via_demand));

    IdbRect* idb_rect = new IdbRect(gr_info.x * 100, gr_info.y * 100, gr_info.ur_x * 100, gr_info.ur_y * 100);
    rect->set_rect(gr_info.x, gr_info.y, gr_info.ur_x, gr_info.ur_y);
    rect->set_layer(gr_info.layer_name);
    IdbLayer* layer = _layout->get_layers()->find_layer(gr_info.layer_name);
    rect->setZValue(layer != nullptr ? layer->get_order() : 0);

    addItem(rect);
    delete idb_rect;

    std::cout << gr_info.layer_name << " " << gr_info.x << " , " << gr_info.y << " ) (" << gr_info.ur_x << " , "
              << gr_info.ur_y << " )";

    n++;
  }
  //   int n = 0;
  //   for (GrInfo gr_info : info_list) {
  //     GuiGrRect*  rect = new GuiGrRect();
  //     std::string coordinate
  //         = std::to_string(_transform.guidb_to_db(gr_info.x)) + " "
  //           + std::to_string(_transform.guidb_to_db(gr_info.y));
  //     rect->set_item_info(coordinate);
  //     rect->add_info("h_supply   : "
  //                    +
  //                    std::to_string(_transform.guidb_to_db(gr_info._h_supply)));
  //     rect->add_info("v_supply   : "
  //                    +
  //                    std::to_string(_transform.guidb_to_db(gr_info._v_supply)));
  //     rect->add_info("h_demand   : "
  //                    +
  //                    std::to_string(_transform.guidb_to_db(gr_info._h_demand)));
  //     rect->add_info("v_demand   : "
  //                    +
  //                    std::to_string(_transform.guidb_to_db(gr_info._v_demand)));
  //     rect->add_info(
  //         "via_demand : "
  //         + std::to_string(_transform.guidb_to_db(gr_info._via_demand)));

  //     IdbRect* idb_rect
  //         = new IdbRect(gr_info.x, gr_info.y, gr_info.ur_x, gr_info.ur_y);
  //     rect->set_rect(gr_info.x, gr_info.y, gr_info.ur_x, gr_info.ur_y);
  //     rect->set_layer(gr_info.layer_name);
  //     IdbLayer* layer =
  //     _layout->get_layers()->find_layer(gr_info.layer_name);
  //     rect->setZValue(layer != nullptr ? layer->get_order() : 0);
  //     addItem(rect);
  //     delete idb_rect;

  //     // if (n % 1000 == 0) {
  //     std::cout << gr_info.layer_name << " " << gr_info.x << " , " <<
  //     gr_info.y
  //               << " ) (" << gr_info.ur_x << " , " << gr_info.ur_y << " )"
  //               << std::endl;
  //     // }
  //     n++;
  //   }
}

#include <fstream>
#include <sstream>
void IdbSetup::analysisResource(const std::string& filename, vector<GrInfo>& info_list) {
  std::ifstream resource_file(filename);

  std::string temp;
  int lb_x, lb_y, rt_x, rt_y;
  std::string layer_name;
  double _h_supply, _v_supply, _h_demand, _v_demand, _via_demand;

  if (resource_file.is_open()) {
    while (getline(resource_file, temp)) {
      std::istringstream str(temp);
      while (!str.eof()) {
        str >> lb_x >> lb_y >> rt_x >> rt_y >> layer_name >> _h_supply >> _v_supply >> _h_demand >> _v_demand >> _via_demand;
      }
      // do some thing
      GrInfo gr_info;
      gr_info.x          = lb_x;
      gr_info.y          = lb_y;
      gr_info.ur_x       = rt_x;
      gr_info.ur_y       = rt_y;
      gr_info.layer_name = layer_name;

      gr_info._h_supply   = _h_supply;
      gr_info._v_supply   = _v_supply;
      gr_info._h_demand   = _h_demand;
      gr_info._v_demand   = _v_demand;
      gr_info._via_demand = _via_demand;
      info_list.emplace_back(gr_info);
      //   std::cout << lb_x << lb_y << rt_x << rt_y << layer_name << _h_supply
      //             << _v_supply << _h_demand << _v_demand << _via_demand
      //             << std::endl;
    }

  } else {
    std::cout << "[GR Error] Failed to open resource file '" << filename << "'!" << std::endl;
  }
  resource_file.close();
}
