#include "Image.hh"

#include <QColor>
#include <QImage>
#include <QPainter>
#include <QString>
#include <cmath>

Qt::GlobalColor obtainQtColor(ipl::IMAGE_COLOR color)
{
  auto color_style = Qt::black;
  switch (color) {
    case ipl::IMAGE_COLOR::kRed:
      color_style = Qt::red;
      break;
    case ipl::IMAGE_COLOR::kGreen:
      color_style = Qt::green;
      break;
    case ipl::IMAGE_COLOR::kBule:
      color_style = Qt::blue;
      break;
    case ipl::IMAGE_COLOR::klightGray:
      color_style = Qt::lightGray;
      break;
    case ipl::IMAGE_COLOR::kdarkYellow:
      color_style = Qt::darkYellow;
      break;
    default:
      break;
  }

  return color_style;
}

ipl::Image::Image(int real_width, int real_height, int num_obj, double bound)
{
  int size_w = 0;
  int size_h = 0;
  double img_pixle = std::max(kimg_size, std::min(num_obj, 4096));
  double dsize_w = std::sqrt(img_pixle * img_pixle * real_width / real_height);
  double dsize_h = img_pixle * img_pixle / dsize_w;
  size_w = std::round(dsize_w);
  size_h = std::round(dsize_h);
  int bound_width = std::round(std::min(size_w, size_h) * bound);
  _scale = sqrt(img_pixle * img_pixle / real_width / real_height);
  _img = new QImage(size_w + bound_width, size_h + bound_width, QImage::Format_RGB32);
  _painter = new QPainter(_img);
  _img->fill(Qt::white);
  _painter->translate(bound_width / 2, bound_width / 2);
  _painter->setPen(QPen(Qt::black, 0));
  _painter->scale(_scale, _scale);
  // _painter->setBrush(QBrush(Qt::BrushStyle::CrossPattern));
}

ipl::Image::~Image()
{
  delete _painter;
  delete _img;
}

void ipl::Image::drawRect(int x, int y, int w, int h, double angle, IMAGE_COLOR color)
{
  int x_off = std::round(-w / 2);
  int y_off = std::round(-h / 2);
  int lx = std::round(x_off * cos(angle) - y_off * sin(angle)) + x;
  int ly = std::round(x_off * sin(angle) + y_off * cos(angle)) + y;

  _painter->save();
  // _painter->setPen(QPen(Qt::black, 5, style));
  _painter->translate(lx, ly);
  auto qt_color = obtainQtColor(color);
  _painter->setPen(QPen(qt_color, 0));
  _painter->rotate(angle * 180 / M_PI);
  _painter->drawRect(0, 0, w, h);
  _painter->drawLine(std::round(0.75 * w), h, w, std::round(0.75 * h));
  _painter->restore();
}

void ipl::Image::drawBaseRect(int lx, int ly, int ux, int uy, IMAGE_COLOR color)
{
  _painter->save();
  auto qt_color = obtainQtColor(color);
  _painter->setPen(QPen(qt_color, 0));
  _painter->drawRect(lx, ly, ux - lx, uy - ly);
  _painter->restore();
}

void ipl::Image::drawArc(int x1, int y1, int x2, int y2)
{
  _painter->save();
  QPoint startPoint(x1, y1);
  QPoint endPoint(x2, y2);
  _painter->drawLine(startPoint, endPoint);

  // 计算箭头的两个端点
  double angle = atan2(endPoint.y() - startPoint.y(), endPoint.x() - startPoint.x());
  double arrowSize = 10;  // 箭头的大小
  QPoint arrowPoint1(endPoint.x() - arrowSize * cos(angle - M_PI / 6), endPoint.y() - arrowSize * sin(angle - M_PI / 6));
  QPoint arrowPoint2(endPoint.x() - arrowSize * cos(angle + M_PI / 6), endPoint.y() - arrowSize * sin(angle + M_PI / 6));

  // 绘制箭头
  QPolygon arrowHead;
  arrowHead << endPoint << arrowPoint1 << arrowPoint2;
  _painter->setBrush(Qt::black);  // 设置箭头为黑色
  _painter->setPen(Qt::NoPen);    // 设置没有线条
  _painter->drawPolygon(arrowHead);
  _painter->restore();
}

void ipl::Image::drawLine(int x1, int y1, int x2, int y2, IMAGE_COLOR color)
{
  _painter->save();
  auto qt_color = obtainQtColor(color);
  _painter->setPen(QPen(qt_color, 0));
  _painter->drawLine(x1, y1, x2, y2);
  _painter->restore();
}

void ipl::Image::drawText(const std::string& text)
{
  QString qtext = QString::fromStdString(text);
  _painter->drawText(0, 0, qtext);
}

void ipl::Image::save(const std::string& filename)
{
  QString qfilename = QString::fromStdString(filename);
  _img->mirrored().save(qfilename);
}
