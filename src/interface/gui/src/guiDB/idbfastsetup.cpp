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
#include "idbfastsetup.h"

#include "guiConfig.h"
#include "omp.h"

IdbSpeedUpSetup::IdbSpeedUpSetup(const std::vector<std::string>& lef_paths, const std::string& def_path,
                                 GuiGraphicsScene* scene)
    : DbSetup(lef_paths, def_path, scene) {
  //   set_type(DbSetupType::kChip);

  _gui_design = new GuiSpeedupDesign(scene, _type);

  initDB();
  createChip();
  //   TestGrGui();
}

IdbSpeedUpSetup::IdbSpeedUpSetup(IdbBuilder* idb_builder, GuiGraphicsScene* scene, DbSetupType type) : DbSetup(scene) {
  std::cout << "idb_builder" << std::endl;
  _db_builder = idb_builder;
  set_type(type);

  _gui_design = new GuiSpeedupDesign(scene, type);

  _design = _db_builder->get_def_service()->get_design();
  _layout = _db_builder->get_def_service()->get_layout();
  createChip();
}

IdbSpeedUpSetup::~IdbSpeedUpSetup() {
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

  _gui_design->clear();
}
////get layer number
int32_t IdbSpeedUpSetup::getLayerCount() {
  IdbLayers* layer_list = _layout->get_layers();
  return layer_list->get_layers_num();
}
////get layer string list
std::vector<std::string> IdbSpeedUpSetup::getLayer() {
  IdbLayers* layers = _layout->get_layers();

  return layers->get_all_layer_name();
}

void IdbSpeedUpSetup::initDB() {
  //   std::list<std::string> lef_files = _fileMap["lef"];
  //   const char* def_file             = _fileMap["def"].begin()->data();

  //   std::vector<std::string> file_list;
  //   std::list<std::string>::iterator it = lef_files.begin();
  //   for (; it != lef_files.end(); ++it) {
  //     std::string file = *it;
  //     file_list.push_back(file);
  //   }

  _db_builder = new IdbBuilder();
  _db_builder->buildLef(_lef_paths);

  IdbDefService* def_service = nullptr;
  DbSetupType type           = get_type();
  switch (type) {
    case DbSetupType::kChip: def_service = _db_builder->buildDef(_def_path); break;
    case DbSetupType::kFloorplan: def_service = _db_builder->buildDefFloorplan(_def_path); break;
    case DbSetupType::kGlobalPlace: def_service = _db_builder->buildDef(_def_path); break;
    case DbSetupType::kDetailPlace: def_service = _db_builder->buildDef(_def_path); break;
    case DbSetupType::kGlobalRouting: def_service = _db_builder->buildDef(_def_path); break;
    case DbSetupType::kDetailRouting: def_service = _db_builder->buildDef(_def_path); break;
    case DbSetupType::kClockTree: def_service = _db_builder->buildDef(_def_path); break;
    case DbSetupType::kCellMaster: break;
    default: break;
  }

  if (def_service == nullptr) {
    /// release
  }

  /// set top db
  _design = def_service->get_design();
  _layout = def_service->get_layout();

  /// update config layer
  guiConfig->UpdateLayerTree(_layout->get_layers()->get_all_layer_name());
}

void IdbSpeedUpSetup::initGuiDB() {
  _gui_design->set_idb_layout(_layout);
  IdbDie* db_die = _layout->get_die();

  _gui_design->init((QRectF(_transform.db_to_guidb(db_die->get_llx()), _transform.db_to_guidb(db_die->get_lly()),
                            _transform.db_to_guidb(db_die->get_width()), _transform.db_to_guidb(db_die->get_height()))));

  addItem(new QGraphicsRectItem(
      QRectF(_transform.db_to_guidb(db_die->get_llx()) - 50, _transform.db_to_guidb(db_die->get_lly()) - 50,
             _transform.db_to_guidb(db_die->get_width()) + 300, _transform.db_to_guidb(db_die->get_height()) + 100)));
}

void IdbSpeedUpSetup::initTransformer() {
  IdbDie* db_die = _layout->get_die();
  _transform.set_die_height(db_die->get_height());

  IdbUnits* db_unit = _layout->get_units();
  _unit             = db_unit != nullptr ? db_unit->get_micron_dbu() : 1000;

  _transform.set_micron(_unit);
}

/// fit the design visible to view
void IdbSpeedUpSetup::fitView(double width, double height) {
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

void IdbSpeedUpSetup::createChip() {
  std::cout << "Begin to create chip..." << std::endl;
  initTransformer();
  initGuiDB();
  /// basic ui info

  /// depends on set up type
  switch (_type) {
    case DbSetupType::kChip:
      createDie();
      createCore();
      createIO();
      createRow();
      createInstance();
      createSpecialNet();
      createNet();
      createTrackGrid();
      createBlockage();
      _gui_design->finishCreateItem();
      break;
    case DbSetupType::kFloorplan:
      createDie();
      createCore();
      createIO();
      createRow();
      createInstance();
      createSpecialNet();
      //   createNet();
      createBlockage();
      _gui_design->finishCreateItem();
      break;
    case DbSetupType::kGlobalPlace:
      createDie();
      createCore();
      createIO();
      createRow();
      createInstance();
      //   createSpecialNet();
      //   createBlockage();
      //   createNet();
      _gui_design->finishCreateItem();
      break;
    case DbSetupType::kDetailPlace:
      createDie();
      createCore();
      createIO();
      createRow();
      createInstance();
      createSpecialNet();
      createBlockage();
      //   createNet();
      _gui_design->finishCreateItem();
      break;
    case DbSetupType::kGlobalRouting:
      createDie();
      createCore();
      createIO();
      createRow();
      createInstance();
      createSpecialNet();
      createNet();
      _gui_design->finishCreateItem();
      break;
    case DbSetupType::kDetailRouting:
      createDie();
      createCore();
      createIO();
      createRow();
      //   createInstance();
      //   createSpecialNet();
      createNet();
      createTrackGrid();
      createBlockage();
      _gui_design->finishCreateItem();
      break;
    case DbSetupType::kClockTree: break;
    case DbSetupType::kCellMaster:
      createDie();
      createCore();
      createRow();
      createTrackGrid();
      showCellMasters();
      _gui_design->finishCreateItem();
      break;
    default: break;
  }

  fitView();

  std::cout << "Success to create chip..." << std::endl;
}

void IdbSpeedUpSetup::createDie() {
  std::cout << "Start to create DIE..." << std::endl;
  IdbDie* db_die = _layout->get_die();

  GuiDie* die = new GuiDie();
  /// init die height, do not use db_to_guidb_rect interface
  die->set_rect(QRectF(_transform.db_to_guidb(db_die->get_llx()), _transform.db_to_guidb(db_die->get_lly()),
                       _transform.db_to_guidb(db_die->get_width()), _transform.db_to_guidb(db_die->get_height())));
  die->setZValue(0);

  addItem(die);

  std::cout << "Success to create DIE..." << std::endl;
}

void IdbSpeedUpSetup::createCore() {
  std::cout << "Start to create CORE..." << std::endl;
  IdbCore* db_core = _layout->get_core();

  GuiCore* gui_core = new GuiCore();
  gui_core->set_rect(_transform.db_to_guidb_rect(db_core->get_bounding_box()));
  gui_core->setZValue(0);
  addItem(gui_core);
  std::cout << "Success to create CORE..." << std::endl;
}

void IdbSpeedUpSetup::createIO() {
  std::cout << "Start to create IO..." << std::endl;
  IdbRect* rect      = _layout->get_core()->get_bounding_box();
  int32_t row_height = _design->get_io_pin_list()->getIOPortWidth();

  IdbPins* pins = _design->get_io_pin_list();
  for (IdbPin* pin : pins->get_pin_list()) {
    IdbTerm* term = pin->get_term();
    if (term->is_port_exist()) {
      /// there are "port" key word
      for (IdbPort* port : pin->get_term()->get_port_list()) {
        GuiPin* gui_pin                    = new GuiPin();
        IdbCoordinate<int32_t>* coordinate = port->get_io_average_coordinate();
        IdbConnectDirection direction      = term->get_direction();
        RectEdgePosition rect_edge         = rect->findCoordinateEdgePosition(*coordinate);

        gui_pin->set_IOPin(_transform.db_to_guidb(coordinate->get_x()), _transform.db_to_guidb_rotate(coordinate->get_y()),
                           _transform.db_to_guidb(row_height * 0.8), direction, rect_edge);
        gui_pin->setZValue(port->findZOrderTop());

        gui_pin->setFlag(QGraphicsItem::ItemIsSelectable, true);

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

void IdbSpeedUpSetup::createIOPinPortShape(vector<IdbPin*>& pin_list) {
  for (IdbPin* pin : pin_list) {
    if (pin != nullptr && pin->get_term()->is_placed()) {
      for (IdbLayerShape* layer_shape : pin->get_port_box_list()) {
        createInstanceLayerShape(*layer_shape);
      }
    }
  }
}

void IdbSpeedUpSetup::createRow() {
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

void IdbSpeedUpSetup::createInstanceCorePin(vector<IdbPin*>& pin_list, GuiSpeedupItem* item) {
  if (item == nullptr) {
    return;
  }
  /// use wire width in metal 1 as pin shape width
  /// if metal 1 not exist, using row height / 20 as pin shape width
  //   IdbLayerRouting* layer_routing = dynamic_cast<IdbLayerRouting*>(_layout->get_layers()->find_layer("METAL1"));
  auto layer_routing = _layout->get_layers()->get_bottom_routing_layer();
  int32_t pin_width = layer_routing != nullptr ? layer_routing->get_width() / 4 : _layout->get_rows()->get_row_height() / 10;

  for (IdbPin* pin : pin_list) {
    if (pin->get_term()->is_pdn()) {
      continue;
    }

    if (pin != nullptr && pin->get_term()->get_port_number() > 0 && pin->get_instance()->get_cell_master()->is_core()) {
      if (pin->get_term()->get_pa_list().size() > 0) {
        for (auto pa : pin->get_term()->get_pa_list()) {
          (dynamic_cast<GuiSpeedupInstance*>(item))
              ->add_pin(_transform.db_to_guidb(pa->get_x()), _transform.db_to_guidb_rotate(pa->get_y()),
                        _transform.db_to_guidb(pin_width));
        }
      } else {
        if (pin->get_net() != nullptr) {
          IdbCoordinate<int32_t>* coordinate = pin->get_grid_coordinate();

          (dynamic_cast<GuiSpeedupInstance*>(item))
              ->add_pin(_transform.db_to_guidb(coordinate->get_x()), _transform.db_to_guidb_rotate(coordinate->get_y()),
                        _transform.db_to_guidb(pin_width));
        }
      }
    }
  }
}

void IdbSpeedUpSetup::createInstanceCoreObs(vector<idb::IdbLayerShape*>& obs_list, GuiSpeedupItem* item) {
  if (item == nullptr) {
    return;
  }

  GuiSpeedupInstance* instance_item = dynamic_cast<GuiSpeedupInstance*>(item);
  for (idb::IdbLayerShape* layer_shape : obs_list) {
    IdbLayer* layer = layer_shape->get_layer();
    GuiSpeedupWire* obs_shape =
        instance_item->add_shape(attributeInst->getLayerColorDark(layer->get_name()), layer->get_order(), true);
    createLayerShape(*layer_shape, obs_shape);
  }
}

void IdbSpeedUpSetup::createInstanceMacroPin(vector<IdbPin*>& pin_list, GuiInstance* gui_instance) {
  for (IdbPin* pin : pin_list) {
    if (pin->get_term()->get_name() == "VDD" || pin->get_term()->get_name() == "VSS" || pin->get_net() == nullptr) {
      continue;
    }

    if (pin != nullptr && pin->get_term()->get_port_number() > 0) {
      if (!pin->get_instance()->get_cell_master()->is_core()) {
        IdbCoordinate<int32_t>* coordinate = pin->get_average_coordinate();
        gui_instance->add_pin(_transform.db_to_guidb(coordinate->get_x()),
                              _transform.db_to_guidb_rotate(coordinate->get_y()),
                              _transform.db_to_guidb(_layout->get_rows()->get_row_height() / 20));
      }
    }
  }
}

void IdbSpeedUpSetup::createPinPortShape(vector<IdbPin*>& pin_list, GuiSpeedupItem* item) {
  if (item != nullptr) {
    /// speedup mode
    GuiSpeedupInstance* instance_item = dynamic_cast<GuiSpeedupInstance*>(item);
    for (IdbPin* pin : pin_list) {
      //    if (pin->get_term()->get_name() == "VDD"
      //        || pin->get_term()->get_name() == "VSS")
      //      continue;

      if (pin != nullptr && pin->get_term()->is_instance_pin()) {
        for (IdbLayerShape* layer_shape : pin->get_port_box_list()) {
          IdbLayer* layer = layer_shape->get_layer();
          GuiSpeedupWire* pin_shape =
              instance_item->add_shape(attributeInst->getLayerColorLight(layer->get_name()), layer->get_order());
          createLayerShape(*layer_shape, pin_shape);
        }
      }
    }
  } else {
    /// normal mode
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
}

void IdbSpeedUpSetup::createInstance(IdbInstanceList* inst_list) {
  std::cout << "Begin to create Instance..." << std::endl;

  auto insts = inst_list == nullptr ? _design->get_instance_list()->get_instance_list() : inst_list->get_instance_list();
  if (inst_list == nullptr) {
    std::cout << "inst_list == nullptr" << std::endl;
  }
  for (IdbInstance* instance : insts) {
    if (instance == nullptr || instance->get_cell_master() == nullptr) {
      continue;
    }
    createInstanceCore(instance);
    createInstancePad(instance);
    createInstanceBlock(instance);
  }

  std::cout << "Success to create Instance..." << std::endl;
}

void IdbSpeedUpSetup::createInstanceCore(IdbInstance* instance) {
  IdbCellMaster* cell_master = instance->get_cell_master();
  if ((cell_master->is_core() && (!is_floorplan())) ||
      ((is_floorplan() && (cell_master->is_core_filler() || cell_master->is_endcap())))) {
    IdbRect* bounding_box    = instance->get_bounding_box();
    QRectF rect              = _transform.db_to_guidb_rect(bounding_box);
    GuiSpeedupInstance* item = _gui_design->get_instance_list()->findItem(rect.center());
    if (item == nullptr) {
      std::cout << "Error : can not find Instance item in die" << std::endl;
      return;
    }
    item->add_rect(rect);
    item->set_type(GuiSpeedupItemType::kInstStandarCell);

    if (DbSetupType::kGlobalPlace != _type) {
      createInstanceCorePin(instance->get_pin_list()->get_pin_list(), item);
      createPinPortShape(instance->get_pin_list()->get_pin_list(), item);
      createInstanceCoreObs(instance->get_obs_box_list(), item);
    }
  }
}

void IdbSpeedUpSetup::createInstancePad(IdbInstance* instance) {
  IdbCellMaster* cell_master = instance->get_cell_master();
  if (cell_master->is_pad() || cell_master->is_cover()) {
    IdbRect* bounding_box = instance->get_bounding_box();
    GuiPad* gui_pad       = new GuiPad();
    gui_pad->set_rect(_transform.db_to_guidb_rect(bounding_box), instance->get_orient());

    addItem(gui_pad);

    createInstanceMacroPin(instance->get_pin_list()->get_pin_list(), gui_pad);

    createPinPortShape(instance->get_pin_list()->get_pin_list());
  }
}

void IdbSpeedUpSetup::createInstanceBlock(IdbInstance* instance) {
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

    createInstanceMacroPin(instance->get_pin_list()->get_pin_list(), gui_block);

    createPinPortShape(instance->get_pin_list()->get_pin_list());
  }
}

GuiSpeedupItem* IdbSpeedUpSetup::findViaItem(IdbLayerShape& layer_shape, GuiSpeedupItemType gui_type) {
  GuiSeedupViaContainer* container = nullptr;
  switch (gui_type) {
    case GuiSpeedupItemType::kSignal: {
      container = _gui_design->get_signal_via_container();
    } break;
    case GuiSpeedupItemType::kSignalClock: {
      container = _gui_design->get_clock_via_container();
    } break;
    case GuiSpeedupItemType::kSignalPower:
    case GuiSpeedupItemType::kPdnPower: {
      container = _gui_design->get_power_via_container();
    } break;
    case GuiSpeedupItemType::kSignalGround:
    case GuiSpeedupItemType::kPdnGround: {
      container = _gui_design->get_ground_via_container();
    } break;
    default: {
      container = _gui_design->get_signal_via_container();
    } break;
  }

  GuiSpeedupViaList* via_list = container->findViaList(layer_shape.get_layer()->get_name());
  if (via_list == nullptr) {
    std::cout << "Error : can not find via list container..." << std::endl;
    return nullptr;
  }

  IdbCoordinate mid_coord = layer_shape.get_bounding_box().get_middle_point();
  return via_list->findItem(QPointF(_transform.db_to_guidb(mid_coord.get_x()), _transform.db_to_guidb(mid_coord.get_y())));
}

void IdbSpeedUpSetup::createLayerShape(IdbLayerShape& layer_shape, GuiSpeedupItem* item) {
  if (layer_shape.get_layer() == nullptr) {
    std::cout << "Error...createLayerShape : Layer not exist :  " << std::endl;
    return;
  }

  if (layer_shape.is_via()) {
    for (IdbRect* cut_rect : layer_shape.get_rect_list()) {
      if (item == nullptr) {
        GuiVia* via = new GuiVia();
        via->set_rect(_transform.db_to_guidb_rect(cut_rect));
        via->set_layer(layer_shape.get_layer()->get_name());
        via->setZValue(layer_shape.get_layer()->get_order());

        addItem(via);
        std::cout << "Error : pseedup via do not exist..." << std::endl;
      } else {
        GuiSpeedupVia* via_item = (dynamic_cast<GuiSpeedupVia*>(item));
        via_item->add_rect(_transform.db_to_guidb_rect(cut_rect));
      }
    }
  } else {
    for (IdbRect* rect : layer_shape.get_rect_list()) {
      if (item == nullptr) {
        GuiPower* gui_rect = new GuiPower();
        gui_rect->set_rect(_transform.db_to_guidb_rect(rect));
        gui_rect->set_layer(layer_shape.get_layer()->get_name());
        gui_rect->setZValue(layer_shape.get_layer()->get_order());

        addItem(gui_rect);
      } else {
        (dynamic_cast<GuiSpeedupWire*>(item))->add_rect(_transform.db_to_guidb_rect(rect));
      }
    }
  }
}

void IdbSpeedUpSetup::createInstanceLayerShape(IdbLayerShape& layer_shape) {
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

void IdbSpeedUpSetup::createSpecialNetVia(IdbSpecialWireSegment* segment, bool b_vdd) {
  if (segment == nullptr) {
    return;
  }

  if (segment->is_via()) {
    GuiSpeedupItemType type = b_vdd ? GuiSpeedupItemType::kPdnPower : GuiSpeedupItemType::kPdnGround;

    IdbVia* via                   = segment->get_via();
    IdbLayerShape cut_layer_shape = via->get_cut_layer_shape();
    GuiSpeedupItem* item          = findViaItem(cut_layer_shape, type);
    if (item == nullptr) {
      return;
    }
    item->set_type(type);
    createLayerShape(cut_layer_shape, item);

    GuiSpeedupWire* enclosure_top = dynamic_cast<GuiSpeedupVia*>(item)->get_enclosure_top();
    enclosure_top->set_type(type);
    IdbLayerShape top_layer_shape = via->get_top_layer_shape();
    createLayerShape(top_layer_shape, enclosure_top);

    GuiSpeedupWire* enclosure_bottom = dynamic_cast<GuiSpeedupVia*>(item)->get_enclosure_bottom();
    enclosure_bottom->set_type(type);
    IdbLayerShape bottom_layer_shape = via->get_bottom_layer_shape();
    createLayerShape(bottom_layer_shape, enclosure_bottom);
  }
}

void IdbSpeedUpSetup::createSpecialNetPoints(IdbSpecialWireSegment* segment, GuiSpeedupItem* item) {
  if (segment == nullptr) {
    return;
  }

  /// ensure there are 2 point in a segment
  if (segment->get_point_num() >= 2) {
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

    /// single one
    if (item == nullptr) {
      IdbRect* rect   = new IdbRect(ll_x, ll_y, ur_x, ur_y);
      GuiPower* power = new GuiPower();
      power->set_rect(_transform.db_to_guidb_rect(rect));
      power->set_layer(routing_layer->get_name());
      power->setZValue(routing_layer->get_order());

      addItem(power);
      delete rect;
      rect = nullptr;
    } else {
      (dynamic_cast<GuiSpeedupWire*>(item))->add_rect(_transform.db_to_guidb_rect(ll_x, ll_y, ur_x, ur_y));
    }
  } else {
    // std::cout << "Error...Power segment only use layer, layer = "
    //           << segment->get_layer()->get_name() << std::endl;
  }
}

GuiSpeedupItem* IdbSpeedUpSetup::findSpecialNetItem(GuiSpeedupWireContainer* container, IdbSpecialWireSegment* segment) {
  if (segment == nullptr || segment->get_point_num() < 2) {
    // std::cout << "Error : pdn points number = " << segment->get_point_num() << std::endl;
    return nullptr;
  }

  IdbLayer* layer = segment->get_layer();
  if (layer == nullptr) {
    std::cout << "Error : illegal pdn segment layer." << std::endl;
    return nullptr;
  }

  GuiSpeedupWireList* wire_list = container->findWireList(layer->get_name());

  return wire_list->findItem(QPointF(_transform.db_to_guidb(segment->get_point_start()->get_x()),
                                     _transform.db_to_guidb(segment->get_point_start()->get_y())),
                             QPointF(_transform.db_to_guidb(segment->get_point_second()->get_x()),
                                     _transform.db_to_guidb(segment->get_point_second()->get_y())));
}

void IdbSpeedUpSetup::createSpecialNet() {
  std::cout << "Begin to create PDN..." << std::endl;

  int number = 0;

  IdbSpecialNetList* special_net_list = _design->get_special_net_list();
  for (IdbSpecialNet* special_net : special_net_list->get_net_list()) {
    /// set vdd or vss parameters
    GuiSpeedupWireContainer* container =
        special_net->is_vdd() ? _gui_design->get_power_container() : _gui_design->get_ground_container();
    GuiSpeedupItemType type = special_net->is_vdd() ? GuiSpeedupItemType::kPdnPower : GuiSpeedupItemType::kPdnGround;

    for (IdbSpecialWire* special_wire : special_net->get_wire_list()->get_wire_list()) {
      for (IdbSpecialWireSegment* segment : special_wire->get_segment_list()) {
        if (segment == nullptr) {
          continue;
        }

        if (segment->is_via()) {
          /// if point >=2 means wire + via
          if (segment->get_point_list().size() >= 2) {
            GuiSpeedupItem* this_gui_wire = findSpecialNetItem(container, segment);
            if (this_gui_wire == nullptr) {
              continue;
            }
            this_gui_wire->set_type(type);
            createSpecialNetPoints(segment, this_gui_wire);
          }
          createSpecialNetVia(segment, special_net->is_vdd());
          number++;
        } else {
          /// find gui wire list ptr
          GuiSpeedupItem* this_gui_wire = findSpecialNetItem(container, segment);
          if (this_gui_wire == nullptr) {
            continue;
          }
          this_gui_wire->set_type(type);
          createSpecialNetPoints(segment, this_gui_wire);
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

void IdbSpeedUpSetup::createNetVia(IdbRegularWireSegment* segment, GuiSpeedupItemType gui_type) {
  if (segment == nullptr) {
    return;
  }

  for (IdbVia* via : segment->get_via_list()) {
    /// cut
    IdbLayerShape cut_layer_shape = via->get_cut_layer_shape();
    GuiSpeedupItem* item_cut      = findViaItem(cut_layer_shape, gui_type);
    if (item_cut == nullptr) {
      continue;
    }
    item_cut->set_type(gui_type);
    createLayerShape(cut_layer_shape, item_cut);
    /// bottom
    IdbLayerShape bottom_layer_shape = via->get_bottom_layer_shape();
    GuiSpeedupItem* item_bottom      = (dynamic_cast<GuiSpeedupVia*>(item_cut))->get_enclosure_bottom();
    item_bottom->set_type(gui_type);
    createLayerShape(bottom_layer_shape, item_bottom);
    /// top
    IdbLayerShape top_layer_shape = via->get_top_layer_shape();
    GuiSpeedupItem* item_top      = (dynamic_cast<GuiSpeedupVia*>(item_cut))->get_enclosure_top();
    item_top->set_type(gui_type);
    createLayerShape(top_layer_shape, item_top);
  }
}

void IdbSpeedUpSetup::createNetRect(IdbRegularWireSegment* segment, GuiSpeedupItem* item) {
  if (item == nullptr) {
    std::cout << "Error...createNetRect : GuiSpeedupItem not exist :  " << std::endl;
    return;
  }

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

  IdbRect* rect = new IdbRect(rect_delta);
  rect->moveByStep(coordinate->get_x(), coordinate->get_y());
  (dynamic_cast<GuiSpeedupWire*>(item))->add_rect(_transform.db_to_guidb_rect(rect));
  delete rect;
  rect = nullptr;
}

void IdbSpeedUpSetup::createNetPoints(IdbRegularWireSegment* segment, GuiSpeedupItem* item) {
  if (segment->get_point_number() >= 2)  // ensure the point number >= 2
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

    if (item == nullptr) {
      IdbRect* rect   = new IdbRect(ll_x, ll_y, ur_x, ur_y);
      GuiPower* power = new GuiPower();
      power->set_rect(_transform.db_to_guidb_rect(rect));
      power->set_layer(routing_layer->get_name());
      power->setZValue(routing_layer->get_order());

      addItem(power);
      delete rect;
      rect = nullptr;
    } else {
      (dynamic_cast<GuiSpeedupWire*>(item))->add_rect(_transform.db_to_guidb_rect(ll_x, ll_y, ur_x, ur_y));
    }
    wire_number++;
  } else {
    // std::cout << "Error...Regular segment only use layer, layer = "
    //           << segment->get_layer()->get_name() << std::endl;
  }
}

GuiSpeedupItemType IdbSpeedUpSetup::getNetGuiType(IdbNet* net) {
  if (net->is_signal()) {
    return GuiSpeedupItemType::kSignal;
  }
  if (net->is_clock()) {
    return GuiSpeedupItemType::kSignalClock;
  }
  if (net->is_power()) {
    return GuiSpeedupItemType::kSignalPower;
  }
  if (net->is_ground()) {
    return GuiSpeedupItemType::kSignalGround;
  }

  //   return GuiSpeedupItemType::kNet;
  return GuiSpeedupItemType::kSignal;
}

void IdbSpeedUpSetup::createNet() {
  std::cout << "Begin to create NET..." << std::endl;

  IdbNetList* net_list = _design->get_net_list();

  // #pragma omp parallel for

  int net_id = 0;
  for (IdbNet* net : net_list->get_net_list()) {
    // if ("FE_OFN5472_FE_OFN426_n686" != net->get_net_name()) {
    //   continue;
    // }
    GuiSpeedupItemType gui_type = getNetGuiType(net);

    for (IdbRegularWire* wire : net->get_wire_list()->get_wire_list()) {
      for (IdbRegularWireSegment* segment : wire->get_segment_list()) {
        if (segment->is_via()) {
          /// find gui via list ptr
          createNetVia(segment, gui_type);

          /// if point >=2 means wire + via
          if (segment->get_point_number() >= 2) {
            GuiSpeedupItem* this_gui_wire = findNetItem(segment, gui_type);
            if (this_gui_wire == nullptr) {
              continue;
            }
            this_gui_wire->set_type(gui_type);

            createNetPoints(segment, this_gui_wire);
          }
        } else if (segment->is_rect()) {
          /// find gui wire list ptr
          GuiSpeedupItem* this_gui_wire = findNetItem(segment, gui_type);
          if (this_gui_wire == nullptr) {
            continue;
          }
          this_gui_wire->set_type(gui_type);

          createNetRect(segment, this_gui_wire);
        } else {
          /// find gui wire list ptr
          GuiSpeedupItem* this_gui_wire = findNetItem(segment, gui_type);
          if (this_gui_wire == nullptr) {
            continue;
          }
          this_gui_wire->set_type(gui_type);

          createNetPoints(segment, this_gui_wire);
        }
      }
    }
  }
  std::cout << "create net rect number =  :  " << wire_number << std::endl;
  std::cout << "Success to create NET..." << std::endl;
}

GuiSpeedupItem* IdbSpeedUpSetup::findNetItem(IdbRegularWireSegment* segment, GuiSpeedupItemType gui_type) {
  if (segment == nullptr) {
    return nullptr;
  }
  /// grid
  GuiSpeedupItem* this_gui_wire =
      findNetItem(gui_type == GuiSpeedupItemType::kSignalClock ? _gui_design->get_clock_grid_container()
                                                               : _gui_design->get_net_grid_container(),
                  segment);
  /// find prefer panel
  if (this_gui_wire == nullptr) {
    this_gui_wire =
        findNetItem(gui_type == GuiSpeedupItemType::kSignalClock ? _gui_design->get_clock_panel_prefer_container()
                                                                 : _gui_design->get_net_panel_prefer_container(),
                    segment);
  }
  /// find non prefer panel
  if (this_gui_wire == nullptr) {
    this_gui_wire =
        findNetItem(gui_type == GuiSpeedupItemType::kSignalClock ? _gui_design->get_clock_panel_nonprefer_container()
                                                                 : _gui_design->get_net_panel_nonprefer_container(),
                    segment);
  }

  return this_gui_wire;
}

GuiSpeedupItem* IdbSpeedUpSetup::findNetItem(GuiSpeedupWireContainer* container, IdbRegularWireSegment* segment) {
  if (segment == nullptr || container == nullptr) {
    return nullptr;
  }

  IdbLayer* layer               = segment->get_layer();
  GuiSpeedupWireList* wire_list = container->findWireList(layer->get_name());

  if (wire_list == nullptr) {
    return nullptr;
  }

  if (segment->get_point_list().size() < 2) {
    return wire_list->findItem(QPointF(_transform.db_to_guidb(segment->get_point_start()->get_x()),
                                       _transform.db_to_guidb(segment->get_point_start()->get_y())));
  } else {
    return wire_list->findItem(QPointF(_transform.db_to_guidb(segment->get_point_start()->get_x()),
                                       _transform.db_to_guidb(segment->get_point_start()->get_y())),
                               QPointF(_transform.db_to_guidb(segment->get_point_end()->get_x()),
                                       _transform.db_to_guidb(segment->get_point_end()->get_y())));
  }
}

void IdbSpeedUpSetup::createTrackGrid() {
  std::cout << "Begin to create Track Grid..." << std::endl;

  GuiSeedupGridContainer* prefer_grid_container    = _gui_design->get_track_grid_prefer_container();
  GuiSeedupGridContainer* nonprefer_grid_container = _gui_design->get_track_grid_nonprefer_container();

  qreal width  = _transform.db_to_guidb(_layout->get_die()->get_width());
  qreal height = _transform.db_to_guidb(_layout->get_die()->get_height());

  IdbTrackGridList* track_grid_list = _layout->get_track_grid_list();

  for (IdbTrackGrid* track_grid : track_grid_list->get_track_grid_list()) {
    IdbTrack* track                        = track_grid->get_track();
    int32_t start                          = track->get_start();
    int32_t pitch                          = track->get_pitch();
    GuiSeedupGridContainer* this_container = nullptr;

    for (IdbLayer* layer : track_grid->get_layer_list()) {
      IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(layer);
      if (track->is_track_direction_y()) {
        this_container = routing_layer->is_horizontal() ? prefer_grid_container : nonprefer_grid_container;
        GuiSpeedupItemType gui_type =
            routing_layer->is_horizontal() ? GuiSpeedupItemType::kTrackGridPrefer : GuiSpeedupItemType::kTrackGridNonPrefer;
        for (uint i = 0; i < track_grid->get_track_num(); ++i) {
          qreal y = _transform.db_to_guidb_rotate(start + pitch * i);

          GuiSpeedupGrid* item = findGridItem(this_container, layer->get_name(), QPointF(0, y), QPointF(width, y));
          if (item == nullptr) {
            std::cout << "Error : failed to find Track Grid..." << std::endl;
            continue;
          }
          item->add_point(QPointF(0, y), QPointF(width, y));
          item->set_type(gui_type);
        }

      } else {
        this_container = routing_layer->is_vertical() ? prefer_grid_container : nonprefer_grid_container;
        GuiSpeedupItemType gui_type =
            routing_layer->is_vertical() ? GuiSpeedupItemType::kTrackGridPrefer : GuiSpeedupItemType::kTrackGridNonPrefer;
        for (uint i = 0; i < track_grid->get_track_num(); ++i) {
          qreal x = _transform.db_to_guidb(start + pitch * i);

          GuiSpeedupGrid* item = findGridItem(this_container, layer->get_name(), QPointF(x, 0), QPointF(x, height));
          if (item == nullptr) {
            std::cout << "Error : failed to find Track Grid..." << std::endl;
            continue;
          }
          item->add_point(QPointF(x, 0), QPointF(x, height));
          item->set_type(gui_type);
        }
      }
    }
  }
  std::cout << "Success to Track Grid..." << std::endl;
}

void IdbSpeedUpSetup::createTrackGridPreferDirection() { }

void IdbSpeedUpSetup::createTrackGridNonPreferDirection() { }

GuiSpeedupGrid* IdbSpeedUpSetup::findGridItem(GuiSeedupGridContainer* container, std::string layer_name, QPointF pt1,
                                              QPointF pt2) {
  GuiSpeedupGridList* grid_list = container->findGridList(layer_name);
  if (grid_list == nullptr) {
    return nullptr;
  }

  return grid_list->findItem(pt1, pt2);
}

void IdbSpeedUpSetup::createBlockage() {
  std::cout << "Begin to create Blockage..." << std::endl;

  for (auto idb_blockage : _design->get_blockage_list()->get_blockage_list()) {
    GuiBlock* gui_block = new GuiBlock();

    int llx = INT_MAX;
    int lly = INT_MAX;
    int urx = 0;
    int ury = 0;
    for (auto idb_rect : idb_blockage->get_rect_list()) {
      llx = min(llx, idb_rect->get_low_x());
      lly = min(lly, idb_rect->get_low_y());
      urx = max(urx, idb_rect->get_high_x());
      ury = max(ury, idb_rect->get_high_y());
      gui_block->set_halo_rect(_transform.db_to_guidb_rect(idb_rect));
    }

    gui_block->set_rect(_transform.db_to_guidb_rect(llx, lly, urx, ury), idb::IdbOrient::kN_R0);

    addItem(gui_block);
  }

  std::cout << "Success to create Blockage..." << std::endl;
}