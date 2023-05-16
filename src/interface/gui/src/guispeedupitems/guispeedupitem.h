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
 * @file GuiSpeedupItem.h
 * @author Yell

 * @brief
 * @version 0.1
 * @date 2021-11-29(V0.1)
 *
 *
 *
 */

#ifndef GUI_SPEEDUP_ITEM
#define GUI_SPEEDUP_ITEM
#include <QBrush>
#include <QColor>
#include <QGraphicsItem>
#include <QGraphicsPixmapItem>
#include <QGraphicsRectItem>
#include <QPainter>
#include <QPen>
#include <QString>
#include <QStyleOptionGraphicsItem>
#include <set>

#include "IdbEnum.h"
#include "IdbGeometry.h"
#include "guiattribute.h"
#include "guigraphicsscene.h"
#include "omp.h"

enum class GuiSpeedupItemType : int8_t {
  kNone,
  kNet,
  kSignal,
  kSignalClock,
  kSignalPower,
  kSignalGround,
  kPdn,
  kPdnPower,
  kPdnGround,
  kVia,
  kInstance,
  kInstStandarCell,
  kInstIoCell,
  kInstBlock,
  kInstPad,
  kTrackGrid,
  kTrackGridPrefer,
  kTrackGridNonPrefer,
  kGCellGrid,
  kDrc,
  kDrcCutEOL,
  kDrcCutSpacing,
  kDrcCutEnclosure,
  kDrcEOL,
  kDrcMetalShort,
  kDrcPRL,
  kDrcNotchSpacing,
  kDrcMinStep,
  kDrcMinArea,
  kMax
};

class GuiSpeedupItem : public QGraphicsRectItem {
 public:
  explicit GuiSpeedupItem(QGraphicsRectItem* parent = nullptr, GuiSpeedupItemType type = GuiSpeedupItemType::kNone)
      : QGraphicsRectItem(parent) {
    if (type > GuiSpeedupItemType::kNone && type < GuiSpeedupItemType::kMax) {
      _type = type;
    }
    //   setCacheMode(QGraphicsItem::ItemCoordinateCache);
  }
  virtual ~GuiSpeedupItem() {
    clear();

    if (_image_top != nullptr) {
      delete _image_top;
      _image_top = nullptr;
    }

    if (_image_scale_1 != nullptr) {
      delete _image_scale_1;
      _image_scale_1 = nullptr;
    }

    if (_image_scale_2 != nullptr) {
      delete _image_scale_2;
      _image_scale_2 = nullptr;
    }

    if (_image_scale_3 != nullptr) {
      delete _image_scale_3;
      _image_scale_3 = nullptr;
    }
  }
  virtual QRectF boundingRect() const override;
  virtual QPainterPath shape() const override;
  virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

  /// getter
  QRectF get_bounding_rect() { return _bounding_box; }
  virtual std::vector<QRectF>& get_rect_list() { return _rect_list; }
  virtual int32_t get_rect_number() { return _rect_list.size(); }
  virtual std::vector<std::pair<QPointF, QPointF>>& get_points() { return _point_list; }
  virtual int32_t get_points_number() { return _point_list.size(); }
  virtual GuiSpeedupItemType get_type() { return _type; }

  virtual bool is_visible() = 0;
  bool is_clock_tree_visible();
  virtual bool has_capacity() = 0;

  /// setter
  virtual void set_bounding_rect(QRectF bounding_box) { _bounding_box = bounding_box; }
  virtual void set_type(GuiSpeedupItemType type) { _type = type; }
  virtual void adjust_bouding_rect(QRectF rect);
  virtual void add_rect(qreal ll_x, qreal ll_y, qreal witdh, qreal height) {
    _rect_list.emplace_back(QRectF(ll_x, ll_y, witdh, height));
    adjust_bouding_rect(QRectF(ll_x, ll_y, witdh, height));
  }

  virtual void add_rect(QRectF rect) {
    _rect_list.emplace_back(rect);
    adjust_bouding_rect(rect);
  }
  virtual void add_point(QPointF point_1, QPointF point_2) { _point_list.emplace_back(std::make_pair(point_1, point_2)); }

  virtual void add_point(qreal ll_x, qreal ll_y, qreal ur_x, qreal ur_y) {
    _point_list.emplace_back(std::make_pair(QPointF(ll_x, ll_y), QPointF(ur_x, ur_y)));
  }

  virtual GuiSpeedupItem* clone() = 0;

  /// operator
  virtual void clear() {
    _rect_list.clear();
    _point_list.clear();
  }

  /// paiter
  virtual void paintScaleTop(QPainter* painter, qreal lod);
  virtual void paintScale_1st(QPainter* painter, qreal lod);
  virtual void paintScale_2nd(QPainter* painter, qreal lod);
  virtual void paintScale_3rd(QPainter* painter, qreal lod);

  virtual void drawText(QPainter* painter, QRectF rect, QString str, const qreal lod);

  virtual void paintImageTop(qreal lod) {
    if (_image_top != nullptr) {
      delete _image_top;
      _image_top = nullptr;
    }
    // _image_top = new QPixmap(_bounding_box.width(), _bounding_box.height());
    _image_top = new QImage(static_cast<int>(boundingRect().width()), static_cast<int>(boundingRect().height()),
                            QImage::Format::Format_RGB32);

    _image_top->fill(Qt::transparent);

    QPainter painter;
    painter.begin(_image_top);
    paintScaleTop(&painter, lod);
    painter.end();
    // _image_top->save("./0.png");
  }

  virtual void paintImageSacel_1st(qreal lod) {
    if (_image_scale_1 != nullptr) {
      delete _image_scale_1;
      _image_scale_1 = nullptr;
    }
    // _image_scale_1 = new QPixmap(_bounding_box.width(), _bounding_box.height());
    _image_scale_1 = new QImage(static_cast<int>(boundingRect().width()), static_cast<int>(boundingRect().height()),
                                QImage::Format::Format_RGB32);

    _image_scale_1->fill(Qt::transparent);

    QPainter painter;
    painter.begin(_image_scale_1);
    paintScaleTop(&painter, lod);
    painter.end();
    // _image_scale_1->save("./1.png");
  }

  virtual void paintImageScale_2nd(qreal lod) {
    if (_image_scale_2 != nullptr) {
      delete _image_scale_2;
      _image_scale_2 = nullptr;
    }
    // _image_scale_2 = new QPixmap(_bounding_box.width(), _bounding_box.height());
    _image_scale_2 = new QImage(static_cast<int>(boundingRect().width()), static_cast<int>(boundingRect().height()),
                                QImage::Format::Format_RGB32);
    _image_scale_2->fill(Qt::transparent);

    QPainter painter;
    painter.begin(_image_scale_2);
    paintScaleTop(&painter, lod);
    painter.end();
    // _image_scale_2->save("./2.png");
  }

  virtual void paintImageScale_3rd(qreal lod) {
    if (_image_scale_3 != nullptr) {
      delete _image_scale_3;
      _image_scale_3 = nullptr;
    }
    // _image_scale_3 = new QPixmap(_bounding_box.width(), _bounding_box.height());
    _image_scale_3 = new QImage(static_cast<int>(boundingRect().width()), static_cast<int>(boundingRect().height()),
                                QImage::Format::Format_RGB32);
    _image_scale_3->fill(Qt::transparent);

    QPainter painter;
    painter.begin(_image_scale_3);
    paintScaleTop(&painter, lod);
    painter.end();
    // _image_scale_3->save("./3.png");
  }

 private:
  QRectF _bounding_box;
  GuiSpeedupItemType _type = GuiSpeedupItemType::kNone;
  QImage* _image_top       = nullptr;
  QImage* _image_scale_1   = nullptr;
  QImage* _image_scale_2   = nullptr;
  QImage* _image_scale_3   = nullptr;

  std::vector<QRectF> _rect_list;
  std::vector<std::pair<QPointF, QPointF>> _point_list;
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class GuiSpeedupItemList {
 public:
  GuiSpeedupItemList(GuiGraphicsScene* scene, GuiSpeedupItemType type) {
    _scene = scene;
    _type  = type;
  }
  virtual ~GuiSpeedupItemList() { clear(); }

  /// getter
  GuiGraphicsScene* get_scene() { return _scene; }
  virtual GuiSpeedupItemType get_type() { return _type; }
  virtual std::vector<GuiSpeedupItem*>& get_item_list() { return _item_list; }
  virtual std::vector<GuiSpeedupItem*>& get_current_item_list() { return _current_item_list; }
  virtual int32_t get_current_item_number() { return _current_item_list.size(); }

  /// find item by coordinate
  GuiSpeedupItem* get_item(qreal x, qreal y) { return get_item(QPointF(x, y)); }
  GuiSpeedupItem* get_item(QPointF pt) {
    int index_x = pt.x() / _step_x;
    int index_y = pt.y() / _step_y;
    if (index_x < _number_x && index_y < _number_y) {
      int index = index_y * _number_x + index_x;

      if (_current_item_list[index]->has_capacity()) {
        return _current_item_list[index];
      } else {
        /// add to scene
        // addSceneItem(_current_item_list[index]);
        /// new a item and set to current list
        auto item                 = _current_item_list[index]->clone();
        _current_item_list[index] = item;
        add_item(item);
        return item;
      }
    }
    number_not_find++;
    return nullptr;
  }

  GuiSpeedupItem* findCurrentItem(qreal x, qreal y) { return findCurrentItem(QPointF(x, y)); }

  GuiSpeedupItem* findCurrentItem(QPointF pt) {
    int index_x = pt.x() / _step_x;
    int index_y = pt.y() / _step_y;
    if (index_x < _number_x && index_y < _number_y) {
      int index = index_y * _number_x + index_x;

      return _current_item_list[index];
    }

    return nullptr;
  }

  /// find item by wire

  GuiSpeedupItem* get_item(qreal x1, qreal y1, qreal x2, qreal y2) { return get_item(QPointF(x1, y1), QPointF(x2, y2)); }

  GuiSpeedupItem* get_item(QPointF pt1, QPointF pt2) {
    int index_x_1 = pt1.x() / _step_x;
    int index_y_1 = pt1.y() / _step_y;
    int index_x_2 = pt2.x() / _step_x;
    int index_y_2 = pt2.y() / _step_y;
    if (index_x_1 < _number_x && index_y_1 < _number_y && index_x_2 < _number_x && index_y_2 < _number_y) {
      int index_1 = index_y_1 * _number_x + index_x_1;
      int index_2 = index_y_2 * _number_x + index_x_2;

      if (index_1 == index_2) {
        if (_current_item_list[index_1]->has_capacity()) {
          return _current_item_list[index_1];
        } else {
          /// add to scene
          //   addSceneItem(_current_item_list[index_1]);
          /// new a item and set to current list
          auto item                   = _current_item_list[index_1]->clone();
          _current_item_list[index_1] = item;
          add_item(item);

          return item;
        }
      }
    }

    number_not_find++;
    return nullptr;
  }

  /// setter
  virtual void set_type(GuiSpeedupItemType type) { _type = type; }

  virtual GuiSpeedupItem* add_item(GuiSpeedupItem* item) {
    if (item != nullptr) {
      //   _current_item_list.emplace_back(item);
      item->set_type(_type);
      _item_list.emplace_back(item);  // backup
      addSceneItem(item);

      number_create++;

      return item;
    }
    return nullptr;
  }

  virtual GuiSpeedupItem* add_current_item(GuiSpeedupItem* item) {
    if (item != nullptr) {
      item->set_type(_type);
      _current_item_list.emplace_back(item);
      _item_list.emplace_back(item);  // backup
      addSceneItem(item);

      number_create++;

      return item;
    }
    return nullptr;
  }

  virtual void clear() {
    for (GuiSpeedupItem* item : _item_list) {
      if (item != nullptr) {
        delete item;
        item = nullptr;
      }
    }

    _current_item_list.clear();
    _item_list.clear();
  }

  void set_number(int32_t num_x, int32_t num_y) {
    _number_x = num_x;
    _number_y = num_y;
  }

  void set_step(qreal step_x, qreal step_y) {
    _step_x = step_x;
    _step_y = step_y;
  }

  /// operator
  virtual void init(QRectF boundingbox) { _bounding_box = boundingbox; }
  virtual void initPanel(QRectF boundingbox, IdbLayerDirection direction) {
    _bounding_box = boundingbox;
    _direction    = direction;
  }
  virtual void finishCreateItem() {
    // for (GuiSpeedupItem* item : _item_list) {
    //   if (item != nullptr) {
    //     _scene->addItem(item);
    //   }
    // }
  }

  virtual void addSceneItem(GuiSpeedupItem* item) { _scene->addItem(item); }

  /// test
  int32_t get_number_not_find() { return number_not_find; }
  int32_t get_number_create() { return number_create; }

  /// gui
  virtual void update() {
#pragma omp parallel for
    for (auto item : _item_list) {
      item->update();
    }
  }

 private:
  GuiSpeedupItemType _type;
  GuiGraphicsScene* _scene;
  std::vector<GuiSpeedupItem*> _current_item_list;
  std::vector<GuiSpeedupItem*> _item_list;
  QRectF _bounding_box;
  IdbLayerDirection _direction = IdbLayerDirection::kNone;

  int32_t _number_x = 0;
  int32_t _number_y = 0;
  qreal _step_x     = 0;
  qreal _step_y     = 0;

  /// test
  int number_not_find = 0;
  int number_create   = 0;
};

#endif  // GUI_SPEEDUP_ITEM
