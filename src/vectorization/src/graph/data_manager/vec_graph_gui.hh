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
 * @file vec_graph_gui.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2025-02-20
 * @brief the graph gui for vectorization
 */

#pragma once

#include <QApplication>
#include <QCheckBox>
#include <QColor>
#include <QGroupBox>
#include <QHash>
#include <QHeaderView>
#include <QLineEdit>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QPainter>
#include <QPoint>
#include <QResizeEvent>
#include <QSortFilterProxyModel>
#include <QStandardItemModel>
#include <QStyledItemDelegate>
#include <QTableView>
#include <QTimer>
#include <QToolButton>
#include <QVector>
#include <QWheelEvent>
#include <QWidget>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace ivec {

class VecShapeTableView : public QTableView
{
  Q_OBJECT

 public:
  VecShapeTableView(QWidget* parent = nullptr) : QTableView(parent)
  {
    horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
    _setColumnWidths();
  }

 protected:
  void resizeEvent(QResizeEvent* event) override;

 private:
  void _setColumnWidths();
};

class VecColorItemDelegate : public QStyledItemDelegate
{
  Q_OBJECT
 public:
  using QStyledItemDelegate::QStyledItemDelegate;

  void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;

  bool editorEvent(QEvent* event, QAbstractItemModel* model, const QStyleOptionViewItem& option, const QModelIndex& index) override;
};

class VecGraphWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
  Q_OBJECT
 public:
  enum ShapeType
  {
    Wire,
    Rect,
    Via
  };

  struct Shape
  {
    ShapeType type;
    float x1, y1, z1;
    float x2, y2, z2;
    std::string comment;
    float width;
    std::string shape_class;
  };

 public:
  explicit VecGraphWidget(QWidget* parent = nullptr);
  ~VecGraphWidget() override {}

  void addWire(float x1, float y1, float z, float x2, float y2, float /*z2*/, const std::string& comment,
               const std::string& shape_class = "Component 1", const QVector3D& color = QVector3D(0.92f, 0.62f, 0.15f));
  void addRect(float x1, float y1, float z, float x2, float y2, float /*z2*/, const std::string& comment,
               const std::string& shape_class = "Component 1", const QVector3D& color = QVector3D(0.7f, 0.04f, 0.0f));
  void addVia(float x, float y, float z1, float z2, const std::string& comment, const std::string& shape_class = "Component 1",
              const QVector3D& color = QVector3D(0.75f, 0.75f, 0.75f));

  void autoScale();
  void showAxes();
  void initView();

 protected:
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;
  void mousePressEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void wheelEvent(QWheelEvent* event) override;
  bool event(QEvent* e) override;
  void resizeEvent(QResizeEvent* event) override;

 private slots:
  void toggleRotationMode(bool checked);
  void resetView();
  void onHideAllChecked(int state);
  void onShowAllChecked(int state);
  void onUnifiedColorClicked();
  void onSearchTimeout();
  void onClassModelDataChanged(const QModelIndex& top_left, const QModelIndex& bottom_right, const QVector<int>& roles);

 private:
  void _ensureClassInModel(const QString& class_name, const QVector3D& default_color);
  void _drawAxes();
  QVector3D _project(const QVector3D& obj, const QMatrix4x4& model_view, const QMatrix4x4& proj);

 private:
  bool _axes_visible;
  bool _rotation_mode;
  QPoint _last_mouse_pos;
  float _zoom_factor;
  float _pan_x, _pan_y;
  float _rotation_x, _rotation_y, _rotation_z;
  QMatrix4x4 _projection;
  QWidget* _control_panel;
  QToolButton* _rotation_button;
  QToolButton* _reset_view_button;
  QGroupBox* _filtering_group_box;
  QLineEdit* _search_line_edit;
  QToolButton* _unified_color_button;
  QCheckBox* _hide_all_check_box;
  QCheckBox* _show_all_check_box;
  QStandardItemModel* _class_model;
  QSortFilterProxyModel* _proxy_model;
  VecShapeTableView* _table_view;
  QHash<QString, int> _class_indices;
  QHash<QString, QVector<Shape>> _shapes_by_class;
  QHash<QString, QVector3D> _class_colors;
  QHash<QString, bool> _class_visibility;
  QTimer* _search_timer;
  QString _previous_filter_text;
};

}  // namespace ivec
