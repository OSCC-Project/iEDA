#include <boost/geometry.hpp>
#include <boost/polygon/polygon.hpp>
#include <cassert>
#include <fstream>
#include <sstream>

namespace bg = boost::geometry;
using bg_Point = boost::geometry::model::d2::point_xy<int>;
using bg_Segment = boost::geometry::model::linestring<bg_Point>;
using bg_Polygon = boost::geometry::model::polygon<bg_Point>;

void printGDS(std::string gds_name, std::vector<BoostBox>& boost_box_list)
{
  std::ostringstream oss;
  oss << gds_name << ".gds";
  std::string gds_file_path = oss.str();
  oss.clear();

  std::ofstream gds_file(gds_file_path);
  if (gds_file.is_open()) {
    gds_file << "HEADER 600" << std::endl;
    gds_file << "BGNLIB" << std::endl;
    gds_file << "LIBNAME GDSLib" << std::endl;
    gds_file << "UNITS 0.001 1e-9" << std::endl;
    // pin_point
    for (size_t i = 0; i < boost_box_list.size(); i++) {
      BoostBox& boost_box = boost_box_list[i];

      gds_file << "BGNSTR" << std::endl;
      gds_file << "STRNAME boost_box_" << i << std::endl;
      int lx = boost_box.min_corner().x();
      int ly = boost_box.min_corner().y();
      int hx = boost_box.max_corner().x();
      int hy = boost_box.max_corner().y();

      gds_file << "BOUNDARY" << std::endl;
      gds_file << "LAYER " << i << std::endl;
      gds_file << "DATATYPE 0" << std::endl;
      gds_file << "XY" << std::endl;
      gds_file << lx << " : " << ly << std::endl;
      gds_file << hx << " : " << ly << std::endl;
      gds_file << hx << " : " << hy << std::endl;
      gds_file << lx << " : " << hy << std::endl;
      gds_file << lx << " : " << ly << std::endl;
      gds_file << "ENDEL" << std::endl;

      gds_file << "ENDSTR" << std::endl;
    }
    // top
    gds_file << "BGNSTR" << std::endl;
    gds_file << "STRNAME top" << std::endl;
    // pin_point
    for (size_t i = 0; i < boost_box_list.size(); i++) {
      gds_file << "SREF" << std::endl;
      gds_file << "SNAME boost_box_" << i << std::endl;
      gds_file << "XY 0:0" << std::endl;
      gds_file << "ENDEL" << std::endl;
    }
    gds_file << "ENDSTR" << std::endl;
    gds_file << "ENDLIB" << std::endl;
    gds_file.close();
    std::cout << "[Info] Result has been written to '" << gds_file_path << "'!" << std::endl;
  } else {
    std::cout << "[Error] Failed to open gds file '" << gds_file_path << "'!" << std::endl;
    assert(false);
  }
}

BoostBox warp(gtl::rectangle_data<int>& gtl_rect)
{
  int xl = gtl::xl(gtl_rect);
  int yl = gtl::yl(gtl_rect);
  int xh = gtl::xh(gtl_rect);
  int yh = gtl::yh(gtl_rect);
  return BoostBox(BoostPoint(xl, yl), BoostPoint(xh, yh));
}

void printGDS(std::string gds_name, std::vector<gtl::rectangle_data<int>>& gtl_rect_list)
{
  std::vector<BoostBox> boost_box_list;
  boost_box_list.reserve(gtl_rect_list.size());
  for (size_t i = 0; i < gtl_rect_list.size(); i++) {
    boost_box_list.push_back(warp(gtl_rect_list[i]));
  }
  printGDS(gds_name, boost_box_list);
}

std::vector<gtl::rectangle_data<int>> getOpenOverlapByBoost(const std::vector<gtl::rectangle_data<int>>& master_list,
                                                            const std::vector<gtl::rectangle_data<int>>& rect_list)
{
  gtl::polygon_90_set_data<int> master_poly;
  for (const gtl::rectangle_data<int>& master : master_list) {
    master_poly += master;
  }
  gtl::polygon_90_set_data<int> rect_poly;
  for (const gtl::rectangle_data<int>& rect : rect_list) {
    rect_poly += rect;
  }

  master_poly *= rect_poly;

  std::vector<gtl::rectangle_data<int>> gtl_rect_list;
  gtl::get_rectangles(gtl_rect_list, master_poly);

  std::vector<gtl::rectangle_data<int>> overlap_rect_list;
  for (gtl::rectangle_data<int>& overlap_rect : gtl_rect_list) {
    overlap_rect_list.emplace_back(overlap_rect);
  }
  return overlap_rect_list;
}

std::vector<gtl::rectangle_data<int>> getOpenReducedRectByBoost(const std::vector<gtl::rectangle_data<int>>& rect_list, int lb_x_add_offset,
                                                                int lb_y_add_offset, int rt_x_minus_offset, int rt_y_minus_offset)
{
  gtl::polygon_90_set_data<int> rect_poly;
  for (const gtl::rectangle_data<int>& rect : rect_list) {
    rect_poly += rect;
  }
  rect_poly.shrink(lb_x_add_offset, rt_x_minus_offset, lb_y_add_offset, rt_y_minus_offset);

  std::vector<gtl::rectangle_data<int>> gtl_rect_list;
  gtl::get_rectangles(gtl_rect_list, rect_poly);

  std::vector<gtl::rectangle_data<int>> reduced_rect_list;
  for (gtl::rectangle_data<int>& gtl_rect : gtl_rect_list) {
    reduced_rect_list.emplace_back(gtl_rect);
  }
  return reduced_rect_list;
}

std::vector<gtl::rectangle_data<int>> getOpenCuttingRectListByBoost(const std::vector<gtl::rectangle_data<int>>& master_list,
                                                                    const std::vector<gtl::rectangle_data<int>>& rect_list)
{
  gtl::polygon_90_set_data<int> master_poly;
  for (const gtl::rectangle_data<int>& master : master_list) {
    master_poly += master;
  }
  gtl::polygon_90_set_data<int> rect_poly;
  for (const gtl::rectangle_data<int>& rect : rect_list) {
    rect_poly += rect;
  }

  master_poly -= rect_poly;

  std::vector<gtl::rectangle_data<int>> gtl_rect_list;
  gtl::get_rectangles(gtl_rect_list, master_poly);

  std::vector<gtl::rectangle_data<int>> cutting_rect_list;
  for (gtl::rectangle_data<int>& gtl_rect : gtl_rect_list) {
    cutting_rect_list.emplace_back(gtl_rect);
  }
  return cutting_rect_list;
}

std::vector<gtl::rectangle_data<int>> getClosedOverlapByBoost(const std::vector<gtl::rectangle_data<int>>& master_list,
                                                              const std::vector<gtl::rectangle_data<int>>& rect_list)
{
  std::vector<gtl::rectangle_data<int>> overlap_rect_list;
  return overlap_rect_list;
}

std::vector<gtl::rectangle_data<int>> getClosedReducedRectByBoost(const std::vector<gtl::rectangle_data<int>>& rect_list,
                                                                  int lb_x_add_offset, int lb_y_add_offset, int rt_x_minus_offset,
                                                                  int rt_y_minus_offset)
{
  std::vector<gtl::rectangle_data<int>> reduced_rect_list;
  return reduced_rect_list;
}

std::vector<gtl::rectangle_data<int>> getClosedCuttingRectListByBoost(const std::vector<gtl::rectangle_data<int>>& master_list,
                                                                      const std::vector<gtl::rectangle_data<int>>& rect_list)
{
  gtl::polygon_90_set_data<int> master_poly;
  for (const gtl::rectangle_data<int>& master : master_list) {
    master_poly += master;
  }
  gtl::polygon_90_set_data<int> rect_poly;
  for (const gtl::rectangle_data<int>& rect : rect_list) {
    rect_poly += rect;
  }

  gtl::polygon_90_set_data<int> intersection_poly = master_poly.interact(rect_poly);

  std::vector<gtl::polygon_90_set_data<int>> intersection_poly_list;
  bg::intersection(master_poly, rect_poly, intersection_poly_list);

  std::vector<gtl::segment_data<int>> intersection_segment_list;
  bg::intersection(master_poly, rect_poly, intersection_segment_list);

  std::vector<gtl::point_data<int>> intersection_point_list;
  bg::intersection(master_poly, rect_poly, intersection_point_list);

  /////////////////////////////////////////////

  // gtl::polygon_90_set_data<int> result_poly = master_poly - rect_poly;

  // std::vector<gtl::rectangle_data<int>> gtl_rect_list;
  // gtl::get_rectangles(gtl_rect_list, result_poly);

  // if (!gtl_rect_list.empty()) {
  //   std::vector<gtl::rectangle_data<int>> cutting_rect_list;
  //   for (gtl::rectangle_data<int>& gtl_rect : gtl_rect_list) {
  //     cutting_rect_list.emplace_back(gtl_rect);
  //   }
  //   return cutting_rect_list;
  // }

  // //////////////////////////////

  // // 将master对每个rect都进行切割
  // // 对于切割后的结果，如没有矩形则表明有一个切割覆盖了所有的多边形
  // std::vector<gtl::polygon_90_set_data<int>> cutting_poly_list;
  // for (const gtl::rectangle_data<int>& rect : rect_list) {
  //   gtl::polygon_90_set_data<int> cutting_poly = master_poly - rect;
  //   std::vector<gtl::rectangle_data<int>> gtl_rect_list;
  //   gtl::get_rectangles(gtl_rect_list, cutting_poly);
  //   if (gtl_rect_list.empty()) {
  //     return {};
  //   }
  //   cutting_poly_list.push_back(master_poly - rect);
  // }
  // gtl::polygon_90_set_data<int> result_poly = cutting_poly_list.front();
  // for (const gtl::polygon_90_set_data<int>& cutting_poly : cutting_poly_list) {
  // }
  // // 将所有候选进行overlap，overlap将允许线段
  // for (size_t i = 0; i < count; i++) {
  //   /* code */
  // }
}

int main()
{
  // {
  //   std::vector<gtl::rectangle_data<int>> master_list;
  //   master_list.emplace_back(3, 1, 5, 4);

  //   std::vector<gtl::rectangle_data<int>> rect_list;
  //   rect_list.emplace_back(1, 0, 4, 5);
  //   rect_list.emplace_back(4, 0, 7, 5);

  //   std::vector<gtl::rectangle_data<int>> result_list = getOpenCuttingRectListByBoost(master_list, rect_list);

  //   std::string function_name = "getOpenCuttingRectListByBoost";
  //   std::string master_list_name = function_name + "_master_list";
  //   std::string rect_list_name = function_name + "_rect_list";
  //   std::string result_list_name = function_name + "_result_list";
  //   printGDS(master_list_name, master_list);
  //   printGDS(rect_list_name, rect_list);
  //   printGDS(result_list_name, result_list);
  // }

  // {
  //   std::vector<gtl::rectangle_data<int>> master_list;
  //   master_list.emplace_back(3, 1, 5, 4);

  //   std::vector<gtl::rectangle_data<int>> rect_list;
  //   rect_list.emplace_back(1, 0, 4, 5);
  //   rect_list.emplace_back(4, 0, 7, 5);

  //   std::vector<gtl::rectangle_data<int>> result_list = getClosedCuttingRectListByBoost(master_list, rect_list);

  //   std::string function_name = "getClosedCuttingRectListByBoost";
  //   std::string master_list_name = function_name + "_master_list";
  //   std::string rect_list_name = function_name + "_rect_list";
  //   std::string result_list_name = function_name + "_result_list";
  //   printGDS(master_list_name, master_list);
  //   printGDS(rect_list_name, rect_list);
  //   printGDS(result_list_name, result_list);
  // }

  {
    std::vector<gtl::rectangle_data<int>> master_list;
    master_list.emplace_back(1, 0, 4, 5);

    std::vector<gtl::rectangle_data<int>> rect_list;
    rect_list.emplace_back(4, 0, 7, 5);

    gtl::polygon_90_set_data<int> master_poly;
    for (const gtl::rectangle_data<int>& master : master_list) {
      master_poly += master;
    }
    gtl::polygon_90_set_data<int> rect_poly;
    for (const gtl::rectangle_data<int>& rect : rect_list) {
      rect_poly += rect;
    }

    std::vector<gtl::polygon_90_data<int>> masters;
    master_poly.get_polygons(masters);

    std::vector<gtl::polygon_90_data<int>> rects;
    rect_poly.get_polygons(rects);

    for (gtl::polygon_90_data<int>& master : masters) {
      for (gtl::polygon_90_data<int>& rect : rects) {


      }
    }

    gtl::polygon_90_set_data<int> master_poly;
    for (const gtl::rectangle_data<int>& master : master_list) {
      master_poly += master;
    }
    gtl::polygon_90_set_data<int> rect_poly;
    for (const gtl::rectangle_data<int>& rect : rect_list) {
      rect_poly += rect;
    }

    gtl::polygon_90_set_data<int> intersection_poly = master_poly.interact(rect_poly);

    std::vector<gtl::rectangle_data<int>> result_list;
    gtl::get_rectangles(result_list, intersection_poly);

    printGDS("master_list", master_list);
    printGDS("rect_list", rect_list);
    printGDS("result_list", result_list);
  }

  return 0;
}