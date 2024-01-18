#include <cassert>
#include <string>
#include <vector>

#include "../source/data_manager/basic/ShapeCurve.hh"
void p(int64_t num1, int64_t num2)
{
  std::cout << num1 << " " << num2 << std::endl;
}

void print_shapes(const imp::ShapeCurve<int64_t>& curve)
{
  const auto& width_list = curve.get_width_list();
  const auto& height_list = curve.get_height_list();
  assert(width_list.size() == height_list.size());
  std::cout << "==============possible_shapes:=================" << std::endl;
  for (size_t i = 0; i < width_list.size(); ++i) {
    std::cout << "[ " << width_list[i].first << ", " << width_list[i].second << " ] ";
  }
  std::cout << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;
  for (size_t i = 0; i < height_list.size(); ++i) {
    std::cout << "[ " << height_list[i].first << ", " << height_list[i].second << " ] ";
  }
  std::cout << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << "width: " << curve.get_width() << std::endl;
  std::cout << "height: " << curve.get_height() << std::endl;
  std::cout << "area: " << curve.get_area() << std::endl;
  std::cout << "min_ar: " << curve.get_min_ar() << std::endl;
  std::cout << "max_ar: " << curve.get_max_ar() << std::endl;
  std::cout << "curr_ar: " << curve.get_ar() << std::endl;
}

int main(int argc, char* argv[])
{
  // macro
  typedef int64_t T;
  imp::ShapeCurve<T> s1(0.7, 1.5);
  std::vector<std::pair<T, T>> discrete_shapes;
  discrete_shapes.emplace_back(100, 200);
  s1.setShapesDiscrete(discrete_shapes, true);
  print_shapes(s1);

  // macro cluster
  discrete_shapes.clear();
  discrete_shapes.emplace_back(100, 200);
  discrete_shapes.emplace_back(100, 300);
  discrete_shapes.emplace_back(300, 100);
  s1.setShapesDiscrete(discrete_shapes, true);
  print_shapes(s1);

  s1.setShapesDiscrete(discrete_shapes, false);
  print_shapes(s1);

  // pure soft cluster
  discrete_shapes.clear();
  s1.setShapes(10000, discrete_shapes);
  print_shapes(s1);

  // mixed cluster
  discrete_shapes.clear();
  discrete_shapes.clear();
  discrete_shapes.emplace_back(60, 160);
  discrete_shapes.emplace_back(80, 125);
  discrete_shapes.emplace_back(90, 111);
  discrete_shapes.emplace_back(100, 100);
  discrete_shapes.emplace_back(111, 90);
  discrete_shapes.emplace_back(125, 80);
  discrete_shapes.emplace_back(160, 60);
  s1.setShapes(10000, discrete_shapes);
  print_shapes(s1);

  s1.setShapes(1000, discrete_shapes);
  print_shapes(s1);

  return 0;
}