/**
 * @file Image.hh
 * @author Fuxing Huang (fxxhuang@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-06-14
 *
 * @copyright Copyright (c) 2023
 *
 */
class QColor;
class QImage;
class QPainter;
class QString;
#include <memory>
#include <string>
namespace ipl {

enum class IMAGE_COLOR
{
  kNone = 0,
  kRed,
  kGreen,
  kBule,
  klightGray,
  kdarkYellow
};

class Image
{
 public:
  Image(int real_width, int real_height, int num_obj, double bound = 0.1);
  ~Image();
  void drawRect(int x, int y, int w, int h, double angle = 0, IMAGE_COLOR color = IMAGE_COLOR::kNone);
  void drawBaseRect(int lx, int ly, int ux, int uy, IMAGE_COLOR color = IMAGE_COLOR::kNone);
  void drawArc(int x1, int y1, int x2, int y2);
  void drawLine(int x1, int y1, int x2, int y2, IMAGE_COLOR color = IMAGE_COLOR::kNone);
  void drawText(const std::string& text);
  void save(const std::string& filename);

  // Qt::GlobalColor obtainQtColor(IMAGE_COLOR color);

 private:
  // std::unique_ptr<QImage> _img;
  // std::unique_ptr<QPainter> _painter;
  QImage* _img;
  QPainter* _painter;
  const int kimg_size = 800;
  double _scale{1};
  // QColor _back_ground{Qt::white};
  // QColor _line_color{1, 132, 127};
  // QColor _rect_color{250, 190, 210};
  // QColor _point_color{1, 153, 201};
};

}  // namespace ipl
