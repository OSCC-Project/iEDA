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
#include "guigraphicsview.h"

// #include <QGLWidget>
#include <QtMath>

#include "mainwindow.h"

GuiGraphicsView::GuiGraphicsView(QWidget* parent)
    : QGraphicsView(parent), _last_mouse_event(QEvent::None, QPointF(), Qt::NoButton, Qt::NoButton, Qt::NoModifier) {
  timerCreate();

  setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));

  //   setBackgroundBrush(QBrush(Qt::black));
  QBrush brush(Qt::black);
  setBackgroundBrush(brush);
  setMouseTracking(true);
  setRubberBandSelectionMode(Qt::ContainsItemShape);
  setDragMode(QGraphicsView::RubberBandDrag);
  // TODO: Use OpenGL to creat viewport
  //  setViewport(new QGLWidget(QGLFormat(QGL::SampleBuffers)));
  //  QSurfaceFormat fmt;
  //  fmt.setSamples(4);
  //  fmt.setSwapInterval(0);
  //  QSurfaceFormat::setDefaultFormat(fmt);
  //   setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
  setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
  setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
  setResizeAnchor(QGraphicsView::AnchorUnderMouse);

  zoom();
}

void GuiGraphicsView::paintEvent(QPaintEvent* event) {
  QGraphicsView::paintEvent(event);
  if (_b_update) {
    QApplication::restoreOverrideCursor();
    _b_update = false;
  }

  std::cout << "paintEvent...." << std::endl;
}

#if QT_CONFIG(wheelevent)
void GuiGraphicsView::wheelEvent(QWheelEvent* e) {
  if (e->modifiers() & Qt::ControlModifier) {
    if (_b_update) {
      e->ignore();
      return;
    }
    _b_update = true;

    QApplication::setOverrideCursor(Qt::WaitCursor);
    if (e->angleDelta().y() > 0) {
      zoomUp();
    } else {
      zoomDown();
    }
    // centerOn(e->x(), e->y());
    e->accept();
  } else {
    e->accept();
    QGraphicsView::wheelEvent(e);
  }
}
#endif

void GuiGraphicsView::keyPressEvent(QKeyEvent* event) {
  if (event->key() == Qt::Key_F) {
    zoomFit();
    std::cout << "Press F" << std::endl;
  }
  return QGraphicsView::keyPressEvent(event);
}

void GuiGraphicsView::keyReleaseEvent(QKeyEvent* event) { return QGraphicsView::keyReleaseEvent(event); }

void GuiGraphicsView::mousePressEvent(QMouseEvent* event) {
  _last_mouse_event = *event;
  _last_mouse_event.setAccepted(false);

  if (Qt::MiddleButton == event->button()) {
    setDragMode(QGraphicsView::ScrollHandDrag);
  }

  //   QGraphicsView::mousePressEvent(event);
  _last_mouse_event.setAccepted(event->isAccepted());
  if (dragMode() == QGraphicsView::ScrollHandDrag && event->button() == Qt::MiddleButton) {
    // Left-button press in scroll hand mode initiates hand scrolling.
    event->accept();
    _handScrolling       = true;
    _hand_scroll_motions = 0;
#ifndef QT_NO_CURSOR
    viewport()->setCursor(Qt::ClosedHandCursor);
#endif
  }
  QGraphicsView::mousePressEvent(event);
}

void GuiGraphicsView::mouseReleaseEvent(QMouseEvent* event) {
  if (dragMode() == QGraphicsView::ScrollHandDrag && event->button() == Qt::MiddleButton) {
#ifndef QT_NO_CURSOR
    viewport()->setCursor(Qt::OpenHandCursor);
#endif
    _handScrolling = false;
    if (scene() && _last_mouse_event.isAccepted() && _hand_scroll_motions <= 6) {
      scene()->clearSelection();
    }
  }
  _last_mouse_event = *event;
  if (Qt::MiddleButton == event->button()) {
    setDragMode(QGraphicsView::RubberBandDrag);
  }

  QGraphicsView::mouseReleaseEvent(event);
  _last_mouse_event.setAccepted(event->isAccepted());
}

void GuiGraphicsView::mouseMoveEvent(QMouseEvent* event) {
  if (dragMode() == QGraphicsView::ScrollHandDrag) {
    if (_handScrolling) {
      QScrollBar* hBar = horizontalScrollBar();
      QScrollBar* vBar = verticalScrollBar();
      QPointF delta    = event->pos() - _last_mouse_event.pos();
      hBar->setValue(hBar->value() + (isRightToLeft() ? delta.x() : -delta.x()));
      vBar->setValue(vBar->value() - delta.y());
      ++_hand_scroll_motions;
    }
  }
  _last_mouse_event = *event;
  _last_mouse_event.setAccepted(false);
  QPointF pos_scene =
      _tranform == nullptr ? mapToScene(event->pos()) : _tranform->guidb_calculate_real_coordinate(mapToScene(event->pos()));

  QString msg_pos_x = QString::number(pos_scene.x(), 'f', 3).append(QString("00"));
  QString msg_pos_y = QString::number(pos_scene.y(), 'f', 3).append(QString("00"));
  QString msg_pos   = QString("positon = (%1, %2)").arg(msg_pos_x).arg(msg_pos_y);
  // msg_pos.sprintf("positon = (%5f , %5f)", pos_scene.x(), pos_scene.y());

  coordinateChange(msg_pos);  // emit CoordinateChange signal

  return QGraphicsView::mouseMoveEvent(event);
}

void GuiGraphicsView::zoom(qreal scale) {
  _scale = scale;
  QTransform matrix;
  matrix.scale(_scale, _scale);
  setTransform(matrix);

  std::cout << " scale = " << _scale << std::endl;
}

void GuiGraphicsView::zoomUp() {
  _scale = _scale * 2;

  timerReset();

  std::cout << "zoomUp scale = " << _scale << std::endl;
}

void GuiGraphicsView::zoomDown() {
  _scale = _scale * 0.5;

  timerReset();

  std::cout << "zoomDown scale = " << _scale << std::endl;
}

void GuiGraphicsView::set_scale_fit_design(qreal width, qreal height) {
  _width  = width * 1.1;
  _height = height * 1.1;

  //   setSceneRect(-_width / 2, -_height / 2, _width * 2, _height * 2);
  zoomFit();
}

void GuiGraphicsView::zoomFit() {
  if (_width > 0 && _height > 0) {
    qreal scale_width  = rect().width() / _width;
    qreal scale_height = rect().height() / _height;

    _scale_fit = std::min(scale_width, scale_height);
    zoom(_scale_fit);

    centerOn(_width / 3, _height / 3);
    // ensureVisible(-50, -50, _width + 50, _height + 50);
  }
}

void GuiGraphicsView::viewRect(QRectF rect_item) {
  if (rect_item.width() > 0 && rect_item.height() > 0 && _width > 0 && _height > 0) {
    qreal scale_width  = rect().width() / rect_item.width();
    qreal scale_height = rect().height() / rect_item.height();

    qreal scale_fit = std::min(scale_width, scale_height);
    zoom(scale_fit * 0.9);

    centerOn(rect_item.center().x(), rect_item.center().y());
    // ensureVisible(-50, -50, _width + 50, _height + 50);
  }
}

void GuiGraphicsView::timerOut() { updateView(); }

bool GuiGraphicsView::captureDesign(std::string path) {
  if (path.empty()) {
    path = "./result/capture/capture.png";
  }

  QPixmap img = grab();
  img.save(QString::fromStdString(path));
  return true;
}