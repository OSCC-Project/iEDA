/**
 * @file transform.h
 * @author Yell
 * @brief Gui data transform method
 * @version 0.1
 * @date 2021-09-22(V0.1)
 *
 *
 *
 */

#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <QPointF>
#include <QRectF>

#include "IdbGeometry.h"

using namespace idb;

class Transform {
 public:
  Transform() : _micron(0), _die_height(0) { }
  Transform(int micron, int die_heigth) : _micron(micron), _die_height(die_heigth) { }
  ~Transform() = default;

  // setter
  void set_micron(int micron) { _micron = micron; }
  void set_die_height(int die_height) { _die_height = die_height; }

  // Operator
  double db_to_guidb(int value) { return ((double)value) / _micron; }

  double db_to_guidb_rotate(int value) { return ((double)(_die_height - value)) / _micron; }

  QRectF db_to_guidb_rect(IdbRect* db_rect) {
    qreal ll_x   = db_rect->get_low_x();
    qreal ll_y   = _die_height - db_rect->get_low_y() - db_rect->get_height();
    qreal width  = db_rect->get_width();
    qreal height = db_rect->get_height();

    return QRectF(ll_x / _micron, ll_y / _micron, width / _micron, height / _micron);
  }

  QRectF db_to_guidb_rect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y) {
    qreal width  = std::abs(ur_x - ll_x);
    qreal height = std::abs(ur_y - ll_y);

    /// transform y-axis
    qreal left_x   = ll_x;
    qreal bottom_y = _die_height - ll_y - height;

    return QRectF(left_x / _micron, bottom_y / _micron, width / _micron, height / _micron);
  }

  int guidb_to_db(double value) { return (int)value * _micron; }
  int guidb_to_db_rotate(double value) { return _die_height - ((int)value * _micron); }

  QPointF guidb_calculate_real_coordinate(QPointF point) {
    return QPointF(point.x(), ((double)_die_height) / _micron - point.y());
  }

  qreal guidb_reverse_y_value(qreal y_value) { return ((qreal)_die_height) / _micron - y_value; }

 private:
  int _micron;
  int _die_height;
};

#endif  // TRANSFORM_H
