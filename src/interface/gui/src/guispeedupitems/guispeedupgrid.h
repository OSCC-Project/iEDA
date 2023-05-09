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
 * @file GuiSpeedupGrid.h
 * @author Yell

 * @brief
 * @version 0.1
 * @date 2021-11-29(V0.1)
 *
 *
 *
 */

#ifndef GUI_SPEEDUP_GRID
#define GUI_SPEEDUP_GRID

#include "guispeedupitem.h"

enum class GuiSpeedupGridType { kNone, kTrackGrid, kGcellGrid, kMax };

class GuiSpeedupGrid : public GuiSpeedupItem {
 public:
  explicit GuiSpeedupGrid(QColor color, int32_t z_order, GuiSpeedupItem* parent = nullptr) : GuiSpeedupItem(parent) {
    setPen(color);

    setBrush(QBrush(color, Qt::BrushStyle::NoBrush));

    _brush = brush();
    _pen   = pen();
    setZValue(z_order);
  }

  virtual ~GuiSpeedupGrid() = default;

  void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

  /// getter
  virtual bool has_capacity() override { return get_rect_number() + get_points_number() < GUI_GRID_MAX ? true : false; }
  virtual bool is_visible() override;

  /// setter

  /// opreator
  virtual GuiSpeedupItem* clone() override {
    GuiSpeedupGrid* new_item =
        new GuiSpeedupGrid(this->pen().color(), this->zValue(), dynamic_cast<GuiSpeedupItem*>(this->parentItem()));
    new_item->set_bounding_rect(get_bounding_rect());
    new_item->set_type(get_type());

    return dynamic_cast<GuiSpeedupItem*>(new_item);
  }

  /// painter
  virtual void paintScaleTop(QPainter* painter, qreal lod);
  virtual void paintScale_1st(QPainter* painter, qreal lod);
  virtual void paintScale_2nd(QPainter* painter, qreal lod);
  virtual void paintScale_3rd(QPainter* painter, qreal lod);

 private:
  QPen _pen;
  QBrush _brush;
};

class GuiSpeedupGridList : public GuiSpeedupItemList {
 public:
  GuiSpeedupGridList(GuiGraphicsScene* scene, GuiSpeedupItemType type) : GuiSpeedupItemList(scene, type) { }
  ~GuiSpeedupGridList() = default;

  /// getter
  QString get_layer_name() { return _layer_name; }
  GuiSpeedupGrid* findItem(QPointF pt) { return dynamic_cast<GuiSpeedupGrid*>(get_item(pt)); }
  GuiSpeedupGrid* findItem(QPointF pt1, QPointF pt2) { return dynamic_cast<GuiSpeedupGrid*>(get_item(pt1, pt2)); }

  /// setter
  void set_color(QColor color) { _color = color; }
  void set_layer_name(QString name) { _layer_name = name; }
  void set_order(int32_t order) { _z_order = order; }
  GuiSpeedupGrid* addCurrentItem(GuiSpeedupItem* parent = nullptr) {
    GuiSpeedupGrid* new_item = new GuiSpeedupGrid(_color, _z_order, parent);
    new_item->set_type(get_type());

    return dynamic_cast<GuiSpeedupGrid*>(add_current_item(new_item));
  }

  /// operator
  void init(QRectF boundingbox, IdbLayerDirection direction);
  void initPanel(QRectF boundingbox, IdbLayerDirection direction);

 private:
  QColor _color;
  QString _layer_name;
  int32_t _z_order;
};

class GuiSeedupGridContainer {
 public:
  GuiSeedupGridContainer(GuiGraphicsScene* scene, GuiSpeedupItemType type) : _scene(scene), _grid_type(type) { }
  ~GuiSeedupGridContainer() = default;

  /// getter
  GuiSpeedupItemType get_grid_type() { return _grid_type; }
  GuiSpeedupGridList* findGridList(std::string layer) {
    QString layer_name = QString::fromStdString(layer);
    for (auto item : _grid_container) {
      if (item->get_layer_name().toUpper() == layer_name.toUpper()) {
        return item;
      }
    }
    std::cout << "Warning : can not find gui grid for " << layer << std::endl;
    return nullptr;
  }

  /// setter
  void set_grid_type(GuiSpeedupItemType grid_type) { _grid_type = grid_type; }
  GuiSpeedupGridList* addGridList(GuiSpeedupItemType type) {
    GuiSpeedupGridList* grid_list = new GuiSpeedupGridList(_scene, type);
    _grid_container.emplace_back(grid_list);

    return grid_list;
  }

  ////operator
  virtual void finishCreateItem() {
    for (auto grid_list : _grid_container) {
      grid_list->finishCreateItem();
    }
  }

  int32_t number_create() {
    int32_t number = 0;
    for (auto grid_list : _grid_container) {
      number += grid_list->get_number_create();
    }
    return number;
  }

  int32_t number_not_find() {
    int32_t number = 0;
    for (auto gird_list : _grid_container) {
      number += gird_list->get_number_not_find();
    }
    return number;
  }

  void clear() {
    for (auto grid_list : _grid_container) {
      if (grid_list != nullptr) {
        delete grid_list;
        grid_list = nullptr;
      }
    }
  }

  void update() {
#pragma omp parallel for
    for (auto grid_list : _grid_container) {
      grid_list->update();
    }
  }

 private:
  GuiGraphicsScene* _scene;
  GuiSpeedupItemType _grid_type;
  std::vector<GuiSpeedupGridList*> _grid_container;
};

#endif  // GUI_SPEEDUP_GRID
