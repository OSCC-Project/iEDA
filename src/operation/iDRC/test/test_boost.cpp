#include <string>

#include "DRC.h"
#include "DRCUtil.h"
#include "DrcAPI.hpp"
#include "DrcEdge.h"
#include "DrcRect.h"
#include "idm.h"

using namespace idrc;

int main(int argc, char* argv[])
{
  // bgi::rtree<std::pair<RTreeSegment, DrcEdge*>, bgi::quadratic<16>> rtree;
  // DrcEdge* edge = new DrcEdge();
  // RTreeSegment rTreeSegment = DRCUtil::getRTreeSegment(edge);
  // rtree.insert(std::make_pair(rTreeSegment, edge));
  // std::cout << rtree.size() << std::endl;
  // rtree.remove(std::make_pair(rTreeSegment, edge));
  // std::cout << rtree.size() << std::endl;
  std::vector<BoostPolygon> poly_set;
  std::vector<BoostRect> rects;

  BoostRect rect1(1, 1, 3, 3);
  BoostRect rect2(2, 2, 4, 4);
  BoostRect rect3(3, 3, 5, 5);
  BoostRect rect4(2, 3, 5, 4);

  poly_set += rect1;
  poly_set += rect2;
  poly_set += rect3;
  poly_set -= rect4;
  BoostPolygon polygon = poly_set[0];
  bp::get_max_rectangles(rects, polygon);
  for (auto& rect : rects) {
    std::cout << bp::xl(rect) << std::endl;
    std::cout << bp::yl(rect) << std::endl;
    std::cout << bp::xh(rect) << std::endl;
    std::cout << bp::yh(rect) << std::endl;
  }
  return 0;
}