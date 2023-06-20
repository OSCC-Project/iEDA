#include "Image.hh"

#include <QColor>
#include <QImage>
#include <QPainter>
#include <QString>
#include <cmath>

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

void ipl::Image::drawRect(int x, int y, int w, int h, double angle)
{
  int x_off = std::round(-w / 2);
  int y_off = std::round(-h / 2);
  int lx = std::round(x_off * cos(angle) - y_off * sin(angle)) + x;
  int ly = std::round(x_off * sin(angle) + y_off * cos(angle)) + y;
  _painter->save();
  // _painter->setPen(QPen(Qt::black, 5, style));
  _painter->translate(lx, ly);
  _painter->rotate(angle * 180 / M_PI);
  _painter->drawRect(0, 0, w, h);
  _painter->drawLine(std::round(0.75 * w), h, w, std::round(0.75 * h));
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
