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
 * @file lm_graph_gui.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2025-02-20
 * @brief the graph gui for large model
 */

#include "lm_graph_gui.hh"

#include <QColorDialog>
#include <QDesktopWidget>
#include <QHBoxLayout>
#include <QHelpEvent>
#include <QRect>
#include <QStandardItem>
#include <QToolTip>
#include <QVBoxLayout>

#include "lm_net_graph_gen.hh"
namespace ilm {

void LmShapeTableView::resizeEvent(QResizeEvent* event)
{
  QTableView::resizeEvent(event);
  _setColumnWidths();
}
void LmShapeTableView::_setColumnWidths()
{
  int total_width = viewport()->width();
  int class_column_width = total_width * 5.5 / 10;
  int visible_column_width = total_width * 2.5 / 10;
  int color_column_width = total_width * 2 / 10;
  setColumnWidth(0, class_column_width);
  setColumnWidth(1, visible_column_width);
  setColumnWidth(2, color_column_width);
}
void LmColorItemDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
  QString color_string = index.data(Qt::DisplayRole).toString();
  QColor color(color_string);
  if (color.isValid()) {
    painter->save();
    painter->setRenderHint(QPainter::Antialiasing, true);
    int side = qMin(option.rect.width(), option.rect.height()) - 8;
    int left = option.rect.left() + (option.rect.width() - side) / 2;
    int top = option.rect.top() + (option.rect.height() - side) / 2;
    QRect square_rect(left, top, side, side);
    painter->setBrush(color);
    painter->setPen(Qt::NoPen);
    painter->drawRoundedRect(square_rect, 4, 4);
    QColor border_color = color.darker(150);
    painter->setBrush(Qt::NoBrush);
    painter->setPen(QPen(border_color, 1));
    painter->drawRoundedRect(square_rect, 4, 4);
    painter->restore();
  } else {
    QStyledItemDelegate::paint(painter, option, index);
  }
}
bool LmColorItemDelegate::editorEvent(QEvent* event, QAbstractItemModel* model, const QStyleOptionViewItem& option,
                                      const QModelIndex& index)
{
  if (event->type() == QEvent::MouseButtonPress) {
    QMouseEvent* mouse_event = static_cast<QMouseEvent*>(event);
    if ((mouse_event->button() == Qt::LeftButton) && (option.rect.contains(mouse_event->pos()))) {
      QString color_string = index.data(Qt::DisplayRole).toString();
      QColor current_color(color_string);
      if (!current_color.isValid()) {
        current_color = QColor("#888");
      }
      QPoint global_mouse_pos = mouse_event->globalPos();
      QRect screen_geometry = QApplication::desktop()->screenGeometry(global_mouse_pos);
      QColorDialog dlg;
      dlg.setCurrentColor(current_color);
      dlg.setOption(QColorDialog::DontUseNativeDialog, true);
      QPoint centered_pos = global_mouse_pos - QPoint(dlg.sizeHint().width() / 2, dlg.sizeHint().height() / 2);
      if (!screen_geometry.contains(QRect(centered_pos, dlg.sizeHint()))) {
        centered_pos = screen_geometry.center() - QPoint(dlg.sizeHint().width() / 2, dlg.sizeHint().height() / 2);
      }
      dlg.move(centered_pos);
      if (dlg.exec() == QDialog::Accepted) {
        QColor new_color = dlg.selectedColor();
        if (new_color.isValid()) {
          model->setData(index, new_color.name(QColor::HexRgb), Qt::EditRole);
        }
      }
      return true;
    }
  }
  return QStyledItemDelegate::editorEvent(event, model, option, index);
}
LmGraphWidget::LmGraphWidget(QWidget* parent)
    : QOpenGLWidget(parent),
      _axes_visible(false),
      _rotation_mode(false),
      _zoom_factor(1.0f),
      _pan_x(0.0f),
      _pan_y(0.0f),
      _rotation_x(45.0f),
      _rotation_y(45.0f),
      _rotation_z(45.0f),
      _previous_filter_text("")
{
  setFocusPolicy(Qt::StrongFocus);

  _control_panel = new QWidget(this);
  _control_panel->setObjectName("control_panel");
  _control_panel->setStyleSheet("background-color: #222; color: #eee;");

  QVBoxLayout* panel_layout = new QVBoxLayout(_control_panel);
  panel_layout->setContentsMargins(10, 10, 10, 10);
  panel_layout->setSpacing(10);

  QHBoxLayout* button_layout = new QHBoxLayout();
  _rotation_button = new QToolButton(_control_panel);
  _rotation_button->setText("Rotation Mode (Off)");
  _rotation_button->setCheckable(true);

  _reset_view_button = new QToolButton(_control_panel);
  _reset_view_button->setText("Reset View");

  button_layout->addWidget(_rotation_button);
  button_layout->addWidget(_reset_view_button);
  panel_layout->addLayout(button_layout);

  QString button_style = R"(
          QToolButton {
              background-color: #444;
              color: #eee;
              border: 1px solid #222;
              border-radius: 4px;
              padding: 4px;
          }
          QToolButton:checked {
              background-color: #666;
          }
          QToolButton:hover {
              background-color: #555;
          }
        )";
  _rotation_button->setStyleSheet(button_style);
  _reset_view_button->setStyleSheet(button_style);

  _search_line_edit = new QLineEdit(_control_panel);
  _search_line_edit->setPlaceholderText("Search filter...");
  _search_line_edit->setStyleSheet("padding: 4px; border: 1px solid #444; border-radius: 4px; background-color: #333; color: #eee;");

  _hide_all_check_box = new QCheckBox("Hide All", _control_panel);
  _show_all_check_box = new QCheckBox("Show All", _control_panel);

  QHBoxLayout* search_layout = new QHBoxLayout();
  search_layout->addWidget(_search_line_edit);
  search_layout->addStretch();
  search_layout->addWidget(_hide_all_check_box);
  search_layout->addWidget(_show_all_check_box);

  _unified_color_button = new QToolButton(_control_panel);
  _unified_color_button->setStyleSheet(
      "QToolButton {"
      "    border: 2px solid gray;"
      "    border-radius: 15px;"
      "    padding: 5px 10px;"
      "    color: white;"
      "    background: qlineargradient("
      "        x1: 0, y1: 0, x2: 1, y2: 0,"
      "        stop: 0 red, stop: 0.17 orange,"
      "        stop: 0.33 yellow, stop: 0.5 green,"
      "        stop: 0.67 cyan, stop: 0.83 blue,"
      "        stop: 1 purple"
      "    );"
      "}"
      "QToolButton:hover {"
      "    background: qlineargradient("
      "        x1: 0, y1: 0, x2: 1, y2: 0,"
      "        stop: 0 purple, stop: 0.17 blue,"
      "        stop: 0.33 cyan, stop: 0.5 green,"
      "        stop: 0.67 yellow, stop: 0.83 orange,"
      "        stop: 1 red"
      "    );"
      "}"
      "QToolButton:pressed {"
      "    background: qlineargradient("
      "        x1: 0, y1: 0, x2: 1, y2: 0,"
      "        stop: 0 black, stop: 1 white"
      "    );"
      "}");
  _unified_color_button->setFixedSize(20, 20);
  search_layout->addWidget(_unified_color_button);

  panel_layout->addLayout(search_layout);

  _filtering_group_box = new QGroupBox("Data Filtering", _control_panel);
  _filtering_group_box->setStyleSheet(
      "QGroupBox { background-color: #333; color: #eee; border: 1px solid #222; border-radius: 4px; margin-top: 10px; }"
      "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }");

  QVBoxLayout* filtering_box_layout = new QVBoxLayout(_filtering_group_box);
  filtering_box_layout->setContentsMargins(5, 5, 5, 5);

  _class_model = new QStandardItemModel(this);
  _class_model->setColumnCount(3);
  _class_model->setHorizontalHeaderLabels({"Class", "Visible", "Color"});

  _proxy_model = new QSortFilterProxyModel(this);
  _proxy_model->setSourceModel(_class_model);
  _proxy_model->setFilterCaseSensitivity(Qt::CaseInsensitive);
  _proxy_model->setFilterKeyColumn(0);

  _table_view = new LmShapeTableView(_filtering_group_box);
  _table_view->setModel(_proxy_model);
  _table_view->setSelectionMode(QAbstractItemView::NoSelection);
  _table_view->setSelectionBehavior(QAbstractItemView::SelectRows);
  _table_view->horizontalHeader()->setStretchLastSection(true);
  auto* color_delegate = new LmColorItemDelegate(_table_view);
  _table_view->setItemDelegateForColumn(2, color_delegate);

  filtering_box_layout->addWidget(_table_view);
  panel_layout->addWidget(_filtering_group_box);
  panel_layout->addStretch();

  connect(_hide_all_check_box, &QCheckBox::stateChanged, this, &LmGraphWidget::onHideAllChecked);
  connect(_show_all_check_box, &QCheckBox::stateChanged, this, &LmGraphWidget::onShowAllChecked);
  connect(_unified_color_button, &QToolButton::clicked, this, &LmGraphWidget::onUnifiedColorClicked);

  _search_timer = new QTimer(this);
  _search_timer->setSingleShot(true);
  _search_timer->setInterval(300);
  connect(_search_timer, &QTimer::timeout, this, &LmGraphWidget::onSearchTimeout);
  connect(_search_line_edit, &QLineEdit::textChanged, this, [this](const QString&) { _search_timer->start(); });

  connect(_rotation_button, &QToolButton::clicked, this, &LmGraphWidget::toggleRotationMode);
  connect(_reset_view_button, &QToolButton::clicked, this, &LmGraphWidget::resetView);
  connect(_class_model, &QStandardItemModel::dataChanged, this, &LmGraphWidget::onClassModelDataChanged);

  _table_view->setColumnWidth(0, 100);
  _table_view->setColumnWidth(1, 60);
  _table_view->setColumnWidth(2, 60);
  _table_view->setAttribute(Qt::WA_NoMousePropagation);
}

void LmGraphWidget::addWire(float x1, float y1, float z, float x2, float y2, float, const std::string& comment,
                            const std::string& shape_class, const QVector3D& color)
{
  if (!((std::abs(x1 - x2) < 1e-6f) || (std::abs(y1 - y2) < 1e-6f))) {
    return;
  }
  Shape shape;
  shape.type = Wire;
  shape.x1 = x1;
  shape.y1 = y1;
  shape.z1 = z;
  shape.x2 = x2;
  shape.y2 = y2;
  shape.z2 = z;
  shape.comment = comment;
  shape.width = 5.0f;
  shape.shape_class = shape_class;
  QString class_name = QString::fromStdString(shape_class);
  _shapes_by_class[class_name].push_back(shape);
  _ensureClassInModel(class_name, color);
  update();
}

void LmGraphWidget::addRect(float x1, float y1, float z, float x2, float y2, float, const std::string& comment,
                            const std::string& shape_class, const QVector3D& color)
{
  Shape shape;
  shape.type = Rect;
  shape.x1 = x1;
  shape.y1 = y1;
  shape.z1 = z;
  shape.x2 = x2;
  shape.y2 = y2;
  shape.z2 = z;
  shape.comment = comment;
  shape.width = 0.0f;
  shape.shape_class = shape_class;
  QString class_name = QString::fromStdString(shape_class);
  _shapes_by_class[class_name].push_back(shape);
  _ensureClassInModel(class_name, color);
  update();
}

void LmGraphWidget::addVia(float x, float y, float z1, float z2, const std::string& comment, const std::string& shape_class,
                           const QVector3D& color)
{
  Shape shape;
  shape.type = Via;
  shape.x1 = x;
  shape.y1 = y;
  shape.z1 = z1;
  shape.x2 = x;
  shape.y2 = y;
  shape.z2 = z2;
  shape.comment = comment;
  shape.width = 5.0f;
  shape.shape_class = shape_class;
  QString class_name = QString::fromStdString(shape_class);
  _shapes_by_class[class_name].push_back(shape);
  _ensureClassInModel(class_name, color);
  update();
}

void LmGraphWidget::autoScale()
{
  if (_shapes_by_class.isEmpty()) {
    return;
  }
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float min_z = std::numeric_limits<float>::max();
  float max_x = -std::numeric_limits<float>::max();
  float max_y = -std::numeric_limits<float>::max();
  float max_z = -std::numeric_limits<float>::max();
  for (auto it = _shapes_by_class.begin(); it != _shapes_by_class.end(); ++it) {
    const QVector<Shape>& shape_list = it.value();
    for (const Shape& shape : shape_list) {
      min_x = std::min(min_x, std::min(shape.x1, shape.x2));
      min_y = std::min(min_y, std::min(shape.y1, shape.y2));
      min_z = std::min(min_z, std::min(shape.z1, shape.z2));
      max_x = std::max(max_x, std::max(shape.x1, shape.x2));
      max_y = std::max(max_y, std::max(shape.y1, shape.y2));
      max_z = std::max(max_z, std::max(shape.z1, shape.z2));
    }
  }
  float dx = max_x - min_x;
  float dy = max_y - min_y;
  float dz = (max_z - min_z) * 10.0f;
  if ((dx < 1e-6f) || (dy < 1e-6f) || (dz < 1e-6f)) {
    return;
  }
  for (auto it = _shapes_by_class.begin(); it != _shapes_by_class.end(); ++it) {
    QVector<Shape>& shape_list = it.value();
    for (Shape& shape : shape_list) {
      shape.x1 = ((shape.x1 - min_x) / dx) * 100.0f;
      shape.x2 = ((shape.x2 - min_x) / dx) * 100.0f;
      shape.y1 = ((shape.y1 - min_y) / dy) * 100.0f;
      shape.y2 = ((shape.y2 - min_y) / dy) * 100.0f;
      shape.z1 = ((shape.z1 - min_z) / dz) * 100.0f;
      shape.z2 = ((shape.z2 - min_z) / dz) * 100.0f;
    }
  }
  initView();
}

void LmGraphWidget::showAxes()
{
  _axes_visible = true;
  update();
}

void LmGraphWidget::initView()
{
  if (_shapes_by_class.isEmpty()) {
    _zoom_factor = 1.0f;
    _pan_x = 0.0f;
    _pan_y = 0.0f;
    _rotation_x = -45.0f;
    _rotation_y = 0.0f;
    _rotation_z = -45.0f;
    update();
    return;
  }
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float min_z = std::numeric_limits<float>::max();
  float max_x = -std::numeric_limits<float>::max();
  float max_y = -std::numeric_limits<float>::max();
  float max_z = -std::numeric_limits<float>::max();
  for (auto it = _shapes_by_class.begin(); it != _shapes_by_class.end(); ++it) {
    const QVector<Shape>& shape_list = it.value();
    for (const Shape& shape : shape_list) {
      min_x = std::min(min_x, std::min(shape.x1, shape.x2));
      min_y = std::min(min_y, std::min(shape.y1, shape.y2));
      min_z = std::min(min_z, std::min(shape.z1, shape.z2));
      max_x = std::max(max_x, std::max(shape.x1, shape.x2));
      max_y = std::max(max_y, std::max(shape.y1, shape.y2));
      max_z = std::max(max_z, std::max(shape.z1, shape.z2));
    }
  }
  float center_x = 0.5f * (min_x + max_x);
  float center_y = 0.5f * (min_y + max_y);
  _pan_x = -center_x;
  _pan_y = -center_y;
  float dx = max_x - min_x;
  float dy = max_y - min_y;
  float dz = (max_z - min_z) * 10.0f;
  float radius = std::sqrt(dx * dx + dy * dy + dz * dz) * 0.5f;
  float fov_rad = 45.0f * 3.14159265f / 180.0f;
  float d = radius / std::tan(fov_rad / 2.0f);
  _zoom_factor = d / 50.0f;
  _rotation_x = -45.0f;
  _rotation_y = 0.0f;
  _rotation_z = -45.0f;
  update();
}

void LmGraphWidget::initializeGL()
{
  initializeOpenGLFunctions();
  glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
}

void LmGraphWidget::resizeGL(int w, int h)
{
  glViewport(0, 0, w, h);
  _projection.setToIdentity();
  _projection.perspective(45.0f, GLfloat(w) / h, 0.1f, 1000.0f);
}

void LmGraphWidget::paintGL()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  QMatrix4x4 model_view;
  model_view.translate(_pan_x, _pan_y, -75.0f * _zoom_factor);
  model_view.rotate(_rotation_x, 1, 0, 0);
  model_view.rotate(_rotation_y, 0, 1, 0);
  model_view.rotate(_rotation_z, 0, 0, 1);
  glMatrixMode(GL_PROJECTION);
  glLoadMatrixf(_projection.constData());
  glMatrixMode(GL_MODELVIEW);
  glLoadMatrixf(model_view.constData());
  for (auto it = _shapes_by_class.begin(); it != _shapes_by_class.end(); ++it) {
    QString class_name = it.key();
    if ((!_class_visibility.contains(class_name)) || (!_class_visibility[class_name])) {
      continue;
    }
    QVector3D draw_color = _class_colors.value(class_name, QVector3D(1, 1, 1));
    glColor3f(draw_color.x(), draw_color.y(), draw_color.z());
    glLineWidth(5.0f);
    const QVector<Shape>& shape_list = it.value();
    for (const Shape& shape : shape_list) {
      switch (shape.type) {
        case Wire: {
          glBegin(GL_LINES);
          glVertex3f(shape.x1, shape.y1, shape.z1);
          glVertex3f(shape.x2, shape.y2, shape.z2);
          glEnd();
          break;
        }
        case Rect: {
          glBegin(GL_QUADS);
          glVertex3f(shape.x1, shape.y1, shape.z1);
          glVertex3f(shape.x2, shape.y1, shape.z1);
          glVertex3f(shape.x2, shape.y2, shape.z2);
          glVertex3f(shape.x1, shape.y2, shape.z2);
          glEnd();
          break;
        }
        case Via: {
          glBegin(GL_LINES);
          glVertex3f(shape.x1, shape.y1, shape.z1);
          glVertex3f(shape.x2, shape.y2, shape.z2);
          glEnd();
          break;
        }
      }
    }
  }
  if (_axes_visible) {
    _drawAxes();
  }
}

void LmGraphWidget::mousePressEvent(QMouseEvent* event)
{
  if (event->pos().x() < _control_panel->width()) {
    return;
  }
  _last_mouse_pos = event->pos();
}

void LmGraphWidget::mouseMoveEvent(QMouseEvent* event)
{
  if (event->pos().x() < _control_panel->width()) {
    return;
  }
  int dx = event->x() - _last_mouse_pos.x();
  int dy = event->y() - _last_mouse_pos.y();
  if (_rotation_mode) {
    if (event->modifiers() & Qt::ShiftModifier) {
      _rotation_z += dx;
    } else {
      _rotation_x += dy;
      _rotation_y += dx;
    }
  } else {
    _pan_x += dx * 0.05f;
    _pan_y -= dy * 0.05f;
  }
  _last_mouse_pos = event->pos();
  update();
}

void LmGraphWidget::wheelEvent(QWheelEvent* event)
{
#if (QT_VERSION < QT_VERSION_CHECK(5, 15, 0))
  QPoint angle_delta = event->angleDelta();
#else
  QPoint angle_delta = event->angleDelta();
#endif
  if (event->pos().x() < _control_panel->width()) {
    return;
  }
  if (angle_delta.y() > 0) {
    _zoom_factor *= 0.9f;
  } else {
    _zoom_factor *= 1.1f;
  }
  update();
}

bool LmGraphWidget::event(QEvent* e)
{
  if (e->type() == QEvent::ToolTip) {
    QHelpEvent* help_event = static_cast<QHelpEvent*>(e);
    if (help_event->pos().x() < _control_panel->width()) {
      QToolTip::hideText();
      e->ignore();
      return true;
    }
    QMatrix4x4 model_view;
    model_view.translate(_pan_x, _pan_y, -75.0f * _zoom_factor);
    model_view.rotate(_rotation_x, 1, 0, 0);
    model_view.rotate(_rotation_y, 0, 1, 0);
    model_view.rotate(_rotation_z, 0, 0, 1);
    const int threshold = 10;
    for (auto it = _shapes_by_class.begin(); it != _shapes_by_class.end(); ++it) {
      const QString& class_name = it.key();
      if (!_class_visibility.value(class_name, true)) {
        continue;
      }
      const QVector<Shape>& shape_list = it.value();
      for (const Shape& shape : shape_list) {
        float mid_x = 0.5f * (shape.x1 + shape.x2);
        float mid_y = 0.5f * (shape.y1 + shape.y2);
        float mid_z = 0.5f * (shape.z1 + shape.z2);
        QVector3D world_pos(mid_x, mid_y, mid_z);
        QVector3D screen_pos = _project(world_pos, model_view, _projection);
        QPointF shape_point(screen_pos.x(), screen_pos.y());
        if ((help_event->pos() - shape_point).manhattanLength() < threshold) {
          QToolTip::showText(help_event->globalPos(), QString::fromStdString(shape.comment), this);
          return true;
        }
      }
    }
    QToolTip::hideText();
    e->ignore();
    return true;
  }
  return QOpenGLWidget::event(e);
}

void LmGraphWidget::resizeEvent(QResizeEvent* event)
{
  QOpenGLWidget::resizeEvent(event);
  int new_width = event->size().width();
  int new_height = event->size().height();
  _control_panel->setGeometry(0, 0, int(new_width * 0.2), new_height);
}

void LmGraphWidget::toggleRotationMode(bool checked)
{
  _rotation_mode = checked;
  _rotation_button->setText(checked ? "Rotation Mode (On)" : "Rotation Mode (Off)");
}

void LmGraphWidget::resetView()
{
  initView();
}

void LmGraphWidget::onHideAllChecked(int state)
{
  if (state == Qt::Checked) {
    _show_all_check_box->setChecked(false);
    int row_count_proxy = _proxy_model->rowCount();
    for (int i = 0; i < row_count_proxy; ++i) {
      QModelIndex proxy_index = _proxy_model->index(i, 1);
      if (!proxy_index.isValid()) {
        continue;
      }
      QModelIndex source_index = _proxy_model->mapToSource(proxy_index);
      if (!source_index.isValid()) {
        continue;
      }
      QStandardItem* item = _class_model->itemFromIndex(source_index);
      if (!item) {
        continue;
      }
      QStandardItem* visible_item = _class_model->item(source_index.row(), 1);
      if (visible_item) {
        visible_item->setCheckState(Qt::Unchecked);
      }
    }
    update();
  }
}

void LmGraphWidget::onShowAllChecked(int state)
{
  if (state == Qt::Checked) {
    _hide_all_check_box->setChecked(false);
    int row_count_proxy = _proxy_model->rowCount();
    for (int i = 0; i < row_count_proxy; ++i) {
      QModelIndex proxy_index = _proxy_model->index(i, 1);
      if (!proxy_index.isValid()) {
        continue;
      }
      QModelIndex source_index = _proxy_model->mapToSource(proxy_index);
      if (!source_index.isValid()) {
        continue;
      }
      QStandardItem* visible_item = _class_model->item(source_index.row(), 1);
      if (visible_item) {
        visible_item->setCheckState(Qt::Checked);
      }
    }
    update();
  }
}

void LmGraphWidget::onUnifiedColorClicked()
{
  QColor current_color("#888");
  QColorDialog dialog(current_color, this);
  dialog.setWindowTitle("Select Unified Color");
  dialog.setOptions(QColorDialog::DontUseNativeDialog);
  QPoint global_mouse_pos = QCursor::pos();
  QRect screen_geometry = QApplication::desktop()->screenGeometry(global_mouse_pos);
  QPoint centered_pos = global_mouse_pos - QPoint(dialog.sizeHint().width() / 2, dialog.sizeHint().height() / 2);
  if (!screen_geometry.contains(QRect(centered_pos, dialog.sizeHint()))) {
    centered_pos = screen_geometry.center() - QPoint(dialog.sizeHint().width() / 2, dialog.sizeHint().height() / 2);
  }
  dialog.move(centered_pos);
  if (dialog.exec() != QDialog::Accepted) {
    return;
  }
  QColor new_color = dialog.selectedColor();
  if (!new_color.isValid()) {
    return;
  }
  int row_count_proxy = _proxy_model->rowCount();
  for (int i = 0; i < row_count_proxy; ++i) {
    QModelIndex proxy_index_visible = _proxy_model->index(i, 1);
    QModelIndex proxy_index_color = _proxy_model->index(i, 2);
    if ((!proxy_index_visible.isValid()) || (!proxy_index_color.isValid())) {
      continue;
    }
    QModelIndex source_index_visible = _proxy_model->mapToSource(proxy_index_visible);
    QModelIndex source_index_color = _proxy_model->mapToSource(proxy_index_color);
    if ((!source_index_visible.isValid()) || (!source_index_color.isValid())) {
      continue;
    }
    QStandardItem* visible_item = _class_model->item(source_index_visible.row(), 1);
    if (!visible_item) {
      continue;
    }
    if (visible_item->checkState() == Qt::Checked) {
      _class_model->setData(source_index_color, new_color.name(QColor::HexRgb), Qt::EditRole);
    }
  }
  update();
}

void LmGraphWidget::onSearchTimeout()
{
  QString text = _search_line_edit->text();
  if (text == _previous_filter_text) {
    return;
  }
  _previous_filter_text = text;
  _proxy_model->setFilterRegExp(text);
}

void LmGraphWidget::onClassModelDataChanged(const QModelIndex& top_left, const QModelIndex& bottom_right, const QVector<int>& roles)
{
  Q_UNUSED(bottom_right);
  Q_UNUSED(roles);
  if (!top_left.isValid()) {
    return;
  }
  int row = top_left.row();
  int col = top_left.column();
  QStandardItem* class_item = _class_model->item(row, 0);
  QStandardItem* visible_item = _class_model->item(row, 1);
  QStandardItem* color_item = _class_model->item(row, 2);
  if ((!class_item) || (!visible_item) || (!color_item)) {
    return;
  }
  QString class_name = class_item->text();
  if (col == 1) {
    bool visible = (visible_item->checkState() == Qt::Checked);
    _class_visibility[class_name] = visible;
  } else if (col == 2) {
    QString color_string = color_item->text();
    QColor c(color_string);
    if (c.isValid()) {
      _class_colors[class_name] = QVector3D(c.redF(), c.greenF(), c.blueF());
    }
  }
  update();
}
void LmGraphWidget::_ensureClassInModel(const QString& class_name, const QVector3D& default_color)
{
  if (_class_indices.contains(class_name)) {
    return;
  }
  if (!_class_colors.contains(class_name)) {
    _class_colors[class_name] = default_color;
  }
  if (!_class_visibility.contains(class_name)) {
    _class_visibility[class_name] = true;
  }
  QList<QStandardItem*> row_items;
  QStandardItem* name_item = new QStandardItem(class_name);
  name_item->setFlags(name_item->flags() & ~Qt::ItemIsEditable);
  QStandardItem* visible_item = new QStandardItem();
  visible_item->setCheckable(true);
  visible_item->setCheckState(Qt::Checked);
  QColor c = QColor::fromRgbF(default_color.x(), default_color.y(), default_color.z());
  QStandardItem* color_item = new QStandardItem(c.name(QColor::HexRgb));
  row_items << name_item << visible_item << color_item;
  int new_row = _class_model->rowCount();
  _class_model->appendRow(row_items);
  _class_indices[class_name] = new_row;
}

void LmGraphWidget::_drawAxes()
{
  glLineWidth(3.0f);
  glColor3f(0.31f, 0.70f, 0.45f);
  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  glVertex3f(105, 0, 0);
  glEnd();
  glColor3f(0.85f, 0.58f, 0.56f);
  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  glVertex3f(0, 105, 0);
  glEnd();
  glColor3f(0.19f, 0.52f, 0.61f);
  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  glVertex3f(0, 0, 105);
  glEnd();
  for (int i = 0; i <= 10; ++i) {
    float tick = i * 10.0f;
    glColor3f(0.31f, 0.70f, 0.45f);
    glBegin(GL_LINES);
    glVertex3f(tick, -1.0f, 0);
    glVertex3f(tick, 1.0f, 0);
    glEnd();
    glColor3f(0.85f, 0.58f, 0.56f);
    glBegin(GL_LINES);
    glVertex3f(-1.0f, tick, 0);
    glVertex3f(1.0f, tick, 0);
    glEnd();
    glColor3f(0.19f, 0.52f, 0.61f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, tick);
    glVertex3f(0, 1.0f, tick);
    glEnd();
  }
}

QVector3D LmGraphWidget::_project(const QVector3D& obj, const QMatrix4x4& model_view, const QMatrix4x4& proj)
{
  QVector4D tmp(obj, 1.0f);
  QVector4D eye = model_view * tmp;
  QVector4D clip = proj * eye;
  if (std::abs(clip.w()) < 1e-6f) {
    return QVector3D();
  }
  QVector3D ndc(clip.x() / clip.w(), clip.y() / clip.w(), clip.z() / clip.w());
  float x = (ndc.x() + 1.0f) * width() / 2.0f;
  float y = (1.0f - ndc.y()) * height() / 2.0f;
  return QVector3D(x, y, ndc.z());
}
}  // namespace ilm
