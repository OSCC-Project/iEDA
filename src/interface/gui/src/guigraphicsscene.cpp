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
#include "guigraphicsscene.h"

#include <QApplication>
#include <QImage>
#include <QString>
#include <iostream>

#include "dbsetup.h"
#include "guiConfig.h"
#include "gui_io.h"
#include "guigraphicsview.h"
#include "idbfastsetup.h"
#include "idbsetup.h"

// TODO: remove sample later

GuiGraphicsScene::GuiGraphicsScene(QObject* parent) : QGraphicsScene(parent) { _graphic_mode = GraphicMode::kFast; }

GuiGraphicsScene::~GuiGraphicsScene() {
  //   _ruler_list.clear();
}

void GuiGraphicsScene::fitView(qreal width, qreal height) {
  for (QGraphicsView* gui_view : views()) {
    (dynamic_cast<GuiGraphicsView*>(gui_view))->set_scale_fit_design(width, height);
  }
}

void GuiGraphicsScene::viewRect(QRectF rect) {
  for (QGraphicsView* gui_view : views()) {
    (dynamic_cast<GuiGraphicsView*>(gui_view))->viewRect(rect);
  }
}

void GuiGraphicsScene::onDbChanged(DbSetup* db_setup) {
  if (_db_setup != nullptr) {
    delete _db_setup;
    _db_setup = nullptr;
  }

  _db_setup = db_setup;

  switch (_graphic_mode) {
    case GraphicMode::kFast:
      for (QGraphicsView* gui_view : views()) {
        (dynamic_cast<GuiGraphicsView*>(gui_view))->set_transform(((IdbSpeedUpSetup*)_db_setup)->get_transform());
      }
      break;
    case GraphicMode::kNormal:
      for (QGraphicsView* gui_view : views()) {
        (dynamic_cast<GuiGraphicsView*>(gui_view))->set_transform(((IdbSetup*)_db_setup)->get_transform());
      }
      break;
    default:
      /// do nothing
      break;
  }
}

void GuiGraphicsScene::createChip(std::map<std::string, std::list<std::string>> map) {
  std::list<std::string> lef_files = map["lef"];
  const char* def_file             = map["def"].begin()->data();

  std::vector<std::string> file_list;
  std::list<std::string>::iterator it = lef_files.begin();
  for (; it != lef_files.end(); ++it) {
    std::string file = *it;
    file_list.push_back(file);
  }

  createChip(file_list, def_file);
}

void GuiGraphicsScene::createChip(std::vector<std::string> lef_paths, std::string def_path) {
  empty = false;
  if (_db_setup != nullptr) {
    delete _db_setup;
  }

  switch (_graphic_mode) {
    case GraphicMode::kFast:
      _db_setup = new IdbSpeedUpSetup(lef_paths, def_path, this);

      for (QGraphicsView* gui_view : views()) {
        (dynamic_cast<GuiGraphicsView*>(gui_view))->set_transform(((IdbSpeedUpSetup*)_db_setup)->get_transform());
      }
      break;
    case GraphicMode::kNormal:
      _db_setup = new IdbSetup(lef_paths, def_path, this);

      for (QGraphicsView* gui_view : views()) {
        (dynamic_cast<GuiGraphicsView*>(gui_view))->set_transform(((IdbSetup*)_db_setup)->get_transform());
      }
      break;
    default:
      /// do nothing
      break;
  }
}

void GuiGraphicsScene::createChip(IdbBuilder* builder, std::string type) {
  if (builder == nullptr) {
    return;
  }

  empty = false;
  if (_db_setup != nullptr) {
    delete _db_setup;
  }

  /// update config layer
  auto layers          = builder->get_lef_service()->get_layout()->get_layers();
  auto layer_name_list = layers->get_all_layer_name();
  /// init color list
  attributeInst->updateColorByLayers(layer_name_list);
  /// init tree
  guiConfig->UpdateLayerTree(layer_name_list);

  /// update GUI
  DbSetupType gui_type = translate_type(type);
  guiConfig->enableClockTree(gui_type == DbSetupType::kClockTree ? true : false);
  switch (_graphic_mode) {
    case GraphicMode::kFast:
      _db_setup = new IdbSpeedUpSetup(builder, this, gui_type);

      for (QGraphicsView* gui_view : views()) {
        (dynamic_cast<GuiGraphicsView*>(gui_view))->set_transform(((IdbSpeedUpSetup*)_db_setup)->get_transform());
      }
      break;
    case GraphicMode::kNormal:
      _db_setup = new IdbSetup(builder, this);

      for (QGraphicsView* gui_view : views()) {
        (dynamic_cast<GuiGraphicsView*>(gui_view))->set_transform(((IdbSetup*)_db_setup)->get_transform());
      }
      break;
    default:
      /// do nothing
      break;
  }
}

void GuiGraphicsScene::createDrc(std::map<std::string, std::map<std::string, std::vector<ids::Violation>>>& drc_db, int max_num) {
  ((IdbSpeedUpSetup*)_db_setup)->showDrc(drc_db, max_num);
}

void GuiGraphicsScene::createGraph(std::map<int, ivec::VecNet> net_map) {
  ((IdbSpeedUpSetup*)_db_setup)->showGraph(net_map);
}

void GuiGraphicsScene::createClockTree(std::vector<iplf::CtsTreeNodeMap*>& node_list) {
  ((IdbSpeedUpSetup*)_db_setup)->showClockTree(node_list);
}

void GuiGraphicsScene::updateInstanceInFastMode(std::vector<iplf::FileInstance>& file_inst_list) {
  ((IdbSpeedUpSetup*)_db_setup)->updateInstanceInFastMode(file_inst_list);
}

void GuiGraphicsScene::mousePressEvent(QGraphicsSceneMouseEvent* event) {
  //   if (event->button() == Qt::LeftButton) {
  //     switch (_shape_code) {
  //       case Shape::None:
  //         _selected_item = nullptr;
  //         foreach (QGraphicsItem* item, items(event->scenePos())) {
  //           item->setFlags(QGraphicsItem::ItemIsSelectable | QGraphicsItem::ItemIsMovable);
  //           item->setSelected(true);
  //           _selected_item = item;
  //           //   qDebug() << _selected_item->scenePos();
  //         }
  //         break;
  //       case Shape::Line: {
  //         Line* _line = new Line;
  //         _shape      = _line;
  //         addItem(_line);
  //         if (_shape) {
  //           _line->set_start(event->scenePos());
  //           _shape->startDraw(event);
  //           bDraw = false;
  //         }
  //         break;
  //       }
  //       case Shape::Ruler:
  //         // Ruler* _ruler = new Ruler;
  //         // _shape        = _ruler;
  //         // addItem(_ruler);
  //         // _ruler_list << _ruler;
  //         // if (_shape) {
  //         //   _ruler->set_start(event->scenePos());
  //         //   _ruler->set_end(event->scenePos());
  //         //   bDraw = false;
  //         // }
  //         break;
  //     }
  //   }
  QGraphicsScene::mousePressEvent(event);
}

void GuiGraphicsScene::mouseReleaseEvent(QGraphicsSceneMouseEvent* event) {
  //   switch (_shape_code) {
  //     case Shape::None:
  //       if (_selected_item != nullptr) {
  //         _selected_item->setSelected(true);
  //       }
  //       break;
  //     case Shape::Line: {
  //       break;
  //     }
  //     case Shape::Ruler: {
  //       //   bDraw = true;
  //       break;
  //     }
  //   }
  QGraphicsScene::mouseReleaseEvent(event);
}

void GuiGraphicsScene::mouseMoveEvent(QGraphicsSceneMouseEvent* event) {
  //   if (_shape && !bDraw) {
  //     _shape->drawing(event);
  //   }
  QGraphicsScene::mouseMoveEvent(event);
}

void GuiGraphicsScene::search(QString str) {
  if (_db_setup != nullptr) {
    IdbSpeedUpSetup* speedup_setup = dynamic_cast<IdbSpeedUpSetup*>(_db_setup);
    speedup_setup->search(str.toStdString());
  }
}

void GuiGraphicsScene::updateScene(std::string node_name, std::string parent_name) {
  if (_db_setup != nullptr) {
    IdbSpeedUpSetup* speedup_setup = dynamic_cast<IdbSpeedUpSetup*>(_db_setup);
    speedup_setup->update(node_name, parent_name);
  }
}

void GuiGraphicsScene::timerOut() {
  timerStop();

  if (_db_setup != nullptr) {
    IdbSpeedUpSetup* speedup_setup = dynamic_cast<IdbSpeedUpSetup*>(_db_setup);
    if (_update_mode == 0) {
      if (guiInst->guiUpdateInstanceInFastMode()) {
        timerStart(_ms);
        return;
      } else {
        guiInst->guiUpdateInstanceInFastMode("", true);
        timerStart(_ms);
        return;
      }
      //   if (!speedup_setup->updateInstance()) {
      //     timerStart(100);
      //     return;
    } else {
      speedup_setup->resetIndex();
      _update_mode = 1;
    }

    if (_update_mode == 1) {
      if (!speedup_setup->updateNet()) {
        timerStart(100);
      } else {
        speedup_setup->resetIndex();
        _update_mode = 2;
      }
    }
  }
}
