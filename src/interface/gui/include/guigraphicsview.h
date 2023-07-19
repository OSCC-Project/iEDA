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
#ifndef GUIGRAPHICSVIEW_H
#define GUIGRAPHICSVIEW_H

#include <QGraphicsView>
#include <QTimer>
#include <QWheelEvent>

#include "transform.h"

class GuiGraphicsView : public QGraphicsView {
  Q_OBJECT
 public:
  GuiGraphicsView(QWidget* parent = nullptr);
  virtual ~GuiGraphicsView() { timerKill(); };

  /// getter
  qreal get_scale() { return _scale; }

  /// setter
  void set_transform(Transform* tranform) { _tranform = tranform; }
  void set_scale(qreal scale) { _scale = scale; }
  void set_scale_fit_design(qreal width, qreal height);

  /// operator
  void zoom(qreal scale = 1);
  void zoomUp();
  void zoomDown();
  void zoomFit();

  bool captureDesign(std::string path = "");

  /// Timer
  bool is_need_timer() { return _scale < _scale_fit * 5 ? true : false; }
  void timerCreate() {
    _timer = new QTimer();
    connect(_timer, SIGNAL(timeout()), this, SLOT(timerOut()));
  }

  void timerKill() {
    if (_timer) {
      timerStop();
      _timer = nullptr;
    }
  }

  void timerStart() {
    if (_timer) {
      _timer->start(300);
    }
  }

  void timerStop() {
    if (_timer) {
      _timer->stop();
    }
  }
  void timerReset() {
    if (_timer) {
      _timer->stop();
      _timer->start(300);
    }
  };

  bool remainingTime() { return _timer->remainingTime() > 0 ? true : false; }

  void updateView() {
    if (_timer) {
      _timer->stop();
    }

    QTransform matrix;
    matrix.scale(_scale, _scale);
    setTransform(matrix);
  }

  void viewRect(QRectF rect_item);

 public slots:
  void timerOut();

 protected:
#if QT_CONFIG(wheelevent)
  void wheelEvent(QWheelEvent*) override;
#endif
  void keyPressEvent(QKeyEvent* event) override;
  void keyReleaseEvent(QKeyEvent* event) override;
  void mousePressEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void moveByPoint(QPointF);

  virtual void paintEvent(QPaintEvent* event);

 signals:
  void coordinateChange(const QString&);

 private:
  //   int         _zoom_value    = 0;
  qreal _scale             = 1;
  qreal _scale_fit         = 1;
  qreal _width             = 0;
  qreal _height            = 0;
  bool _handScrolling      = false;
  int _hand_scroll_motions = 0;
  Transform* _tranform     = nullptr;

  ///
  QMouseEvent _last_mouse_event;
  QTimer* _timer;
  bool _b_update = false;
};

#endif  // GUIGRAPHICSVIEW_H
