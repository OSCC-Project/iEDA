#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../source/data_manager/basic/ShapeCurve.hh"

/*  default params
    set max_num_macro 0
    set min_num_macro 0
    set max_num_inst  0
    set min_num_inst  0
    set tolerance     0.1
    set max_num_level 2
    set coarsening_ratio  10.0
    set num_bundled_ios   3
    set large_net_threshold 50
    set signature_net_threshold 50
    set halo_width   0.0
    set halo_height  0.0
    set fence_lx     0.0
    set fence_ly     0.0
    set fence_ux     100000000.0
    set fence_uy     100000000.0

    set area_weight  0.1
    set outline_weight 100.0
    set wirelength_weight 100.0
    set guidance_weight 10.0
    set fence_weight   10.0
    set boundary_weight 50.0
    set notch_weight    10.0
    set macro_blockage_weight 10.0
    set pin_access_th   0.00
    set target_util 0.25
    set target_dead_space 0.05
    set min_ar  0.33
    set snap_layer -1
    set bus_planning_flag false
*/

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
  std::string type;
  if (curve.isDiscrete())
    type = "discrete";
  else if (curve.isContinous())
    type = "continous";
  else
    type = "mixed";
  std::cout << "type: " << type << std::endl;
  std::cout << "width: " << curve.get_width() << std::endl;
  std::cout << "height: " << curve.get_height() << std::endl;
  std::cout << "area: " << curve.get_area() << std::endl;
  std::cout << "min_ar: " << curve.get_min_ar() << std::endl;
  std::cout << "max_ar: " << curve.get_max_ar() << std::endl;
  std::cout << "curr_ar: " << curve.get_ar() << std::endl;
}

int main(int argc, char* argv[])
{
  // random
  std::uniform_real_distribution<float> distribution = std::uniform_real_distribution<float>(0, 1);
  std::random_device rd;
  std::mt19937 gen(rd());

  // macro
  typedef int64_t T;
  imp::ShapeCurve<T> s1(0.3, 1.5);
  std::vector<std::pair<T, T>> discrete_shapes;
  discrete_shapes.emplace_back(100, 200);
  s1.setShapes(discrete_shapes, 0, true);
  print_shapes(s1);
  std::cout << "resize:----------------------------" << std::endl;
  s1.resizeRandomly(distribution, gen);
  print_shapes(s1);

  // macro cluster, not clipping
  discrete_shapes.clear();
  discrete_shapes.emplace_back(100, 200);
  discrete_shapes.emplace_back(100, 300);
  discrete_shapes.emplace_back(300, 100);
  s1.setShapes(discrete_shapes, 0, true);
  print_shapes(s1);
  std::cout << "resize:----------------------------  " << std::endl;
  s1.resizeRandomly(distribution, gen);
  print_shapes(s1);

  s1.clip(220, 220);
  std::cout << "clip 220, 220: ---------------------" << std::endl;
  print_shapes(s1);

  // s1.clip(100, 150);  // error, no valid shapes left
  // std::cout << "clip 100, 150: ---------------------" << std::endl;
  // print_shapes(s1);

  s1.setShapes(discrete_shapes, 0, false);
  print_shapes(s1);
  std::cout << "resize:----------------------------  " << std::endl;
  s1.resizeRandomly(distribution, gen);
  print_shapes(s1);

  // pure soft cluster
  discrete_shapes.clear();
  s1.setShapes(discrete_shapes, 10000);
  print_shapes(s1);
  std::cout << "resize:----------------------------  " << std::endl;
  s1.resizeRandomly(distribution, gen);
  print_shapes(s1);

  // mixed cluster, use clip
  discrete_shapes.clear();
  discrete_shapes.clear();
  discrete_shapes.emplace_back(60, 160);
  discrete_shapes.emplace_back(80, 125);
  discrete_shapes.emplace_back(90, 111);
  discrete_shapes.emplace_back(100, 100);
  discrete_shapes.emplace_back(111, 90);
  discrete_shapes.emplace_back(125, 80);
  discrete_shapes.emplace_back(160, 60);
  s1.setShapes(discrete_shapes, 10000, true);
  print_shapes(s1);
  std::cout << "resize:----------------------------  " << std::endl;
  s1.resizeRandomly(distribution, gen);
  print_shapes(s1);

  s1.setShapes(discrete_shapes, 1000, true);
  print_shapes(s1);
  std::cout << "resize:----------------------------  " << std::endl;
  s1.resizeRandomly(distribution, gen);
  print_shapes(s1);
  return 0;
}