#include <string>

#include "DRC.h"
#include "DrcAPI.hpp"
#include "DrcRect.h"
#include "idm.h"
#include "ids.hpp"

using namespace idrc;

// void setRectValue(ids::DRCRect& rect, int lb_x, int lb_y, int rt_x, int rt_y, std::string name, int so_id)
// {
//   rect.lb_x = lb_x;
//   rect.lb_y = lb_y;
//   rect.rt_x = rt_x;
//   rect.rt_y = rt_y;
//   rect.layer_name = name;
//   rect.so_id = so_id;
//   rect.type = ids::RectType::kRouting;
// }

int main(int argc, char* argv[])
{
  std::string path = "";
  dmInst->get_config().set_tech_lef_path(path);
  vector<string> path_list;
  path_list.push_back(path);
  dmInst->readLef(path_list);
  std::cout << "read finish" << std::endl;

  DrcAPIInst.initDRC();

  DrcRect* rect1 = DrcAPIInst.getDrcRect(1, "M5", 1000, 1800, 2000, 1910);
  DrcRect* rect2 = DrcAPIInst.getDrcRect(1, "M5", 1800, 1000, 1910, 2000);
  std::vector<DrcRect*> origin_rect_list{rect1, rect2};
  std::vector<DrcRect*> scope_list = DrcAPIInst.getMinScope(origin_rect_list);
  for (auto& rect : scope_list) {
    std::cout << "min_corner:" << rect->get_left() << "," << rect->get_bottom() << " max_corner:" << rect->get_right() << ","
              << rect->get_top() << std::endl;
  }
  //   if (DrcAPIInst.getMaxScope(origin_rect_list)) {
  //     std::cout << "true" << std::endl;
  //   } else {
  //     std::cout << "false" << std::endl;
  //   }

  return 0;
}