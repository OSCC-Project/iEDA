#pragma once
#include <string>
#include <vector>
typedef struct _cairo cairo_t;
namespace imp {

struct CairoColor
{
  double r{0};
  double g{0};
  double b{0};
  double alpha{1};
};

constexpr CairoColor white{1, 1, 1};
constexpr CairoColor black{0, 0, 0};
constexpr CairoColor red{1, 0, 0};
constexpr CairoColor yellow{1, 1, 0};
constexpr CairoColor blue{0, 0, 1};
constexpr CairoColor green{0, 1, 0};

struct CairoRectangle
{
  double lx;
  double ly;
  double width;
  double height;
  CairoColor fill_color;
  CairoColor edge_color;
};
class CairoPlot
{
 public:
  CairoPlot(double canvas_width, double canvas_height) : _canvas_width(canvas_width), _canvas_height(canvas_height) {}
  ~CairoPlot() {}
  void add_rectangle(double lx, double ly, double width, double height, CairoColor face_color, CairoColor edge_color = black);
  void save_as_pdf(const std::string& filename);
  void save_as_png(const std::string& filename);
  void save_as_svg();

 private:
  void save_rectangles(cairo_t*);
  double _canvas_width;
  double _canvas_height;
  std::vector<CairoRectangle> _rect_list;
};
}  // namespace imp