#include <boost/geometry.hpp>
#include <boost/polygon/polygon.hpp>
#include <cassert>
#include <fstream>
#include <sstream>

namespace gtl = boost::polygon;
using namespace boost::polygon::operators;

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bg::model::d2::point_xy<int, bg::cs::cartesian> BoostPoint;
typedef bg::model::box<BoostPoint> BoostBox;

void printGDS(int index, std::vector<BoostBox>& boost_box_list)
{
  std::ostringstream oss;
  oss << "stage_" << index << ".gds";
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

void printGDS(int index, std::vector<gtl::rectangle_data<int>>& gtl_rect_list)
{
  std::vector<BoostBox> boost_box_list;
  boost_box_list.reserve(gtl_rect_list.size());
  for (size_t i = 0; i < gtl_rect_list.size(); i++) {
    boost_box_list.push_back(warp(gtl_rect_list[i]));
  }
  printGDS(index, boost_box_list);
}

int main()
{
  srand((unsigned) time(NULL));

  int rect_num = 100;
  // origin rect
  std::vector<gtl::rectangle_data<int>> origin_rect_list;
  origin_rect_list.reserve(rect_num);
  for (int i = 0; i < rect_num; i++) {
    int xl = rand() % 50 + 1;
    int yl = rand() % 100 + 1;
    int xh = rand() % 50 + 101;
    int yh = rand() % 100 + 101;
    origin_rect_list.push_back(gtl::rectangle_data<int>(xl, yl, xh, yh));
  }

  /**********printGDS***********/
  printGDS(1, origin_rect_list);
  /**********printGDS***********/

  // enlarge by spacing
  std::vector<gtl::rectangle_data<int>> enlarge_rect_list;
  int spacing = 1;
  enlarge_rect_list.reserve(origin_rect_list.size());
  for (size_t i = 0; i < origin_rect_list.size(); i++) {
    gtl::rectangle_data<int>& origin_rect = origin_rect_list[i];
    int xl = gtl::xl(origin_rect) - spacing;
    int yl = gtl::yl(origin_rect) - spacing;
    int xh = gtl::xh(origin_rect) + spacing;
    int yh = gtl::yh(origin_rect) + spacing;
    enlarge_rect_list.push_back(gtl::rectangle_data<int>(xl, yl, xh, yh));
  }
  /**********printGDS***********/
  printGDS(2, enlarge_rect_list);
  /**********printGDS***********/

  // add to polygon
  gtl::polygon_90_set_data<int> poly_set;
  for (size_t i = 0; i < enlarge_rect_list.size(); i++) {
    poly_set += enlarge_rect_list[i];
  }

  // slicing polygon set
  std::vector<gtl::rectangle_data<int>> h_slicing_rect_list;
  gtl::get_rectangles(h_slicing_rect_list, poly_set, gtl::orientation_2d_enum::HORIZONTAL);
  std::vector<gtl::rectangle_data<int>> v_slicing_rect_list;
  gtl::get_rectangles(v_slicing_rect_list, poly_set, gtl::orientation_2d_enum::VERTICAL);

  std::vector<gtl::rectangle_data<int>> slicing_rect_list;
  if (h_slicing_rect_list.size() < v_slicing_rect_list.size()) {
    slicing_rect_list = h_slicing_rect_list;
  } else {
    slicing_rect_list = v_slicing_rect_list;
  }
  /**********printGDS***********/
  printGDS(3, slicing_rect_list);
  /**********printGDS***********/

  // construct rtree
  bgi::rtree<BoostBox, bgi::quadratic<16>> rtree;
  for (size_t i = 0; i < slicing_rect_list.size(); i++) {
    rtree.insert(warp(slicing_rect_list[i]));
  }
  // query
  std::vector<BoostBox> result_list;
  rtree.query(bgi::nearest(BoostPoint(3, 4), 5), std::back_inserter(result_list));
  /**********printGDS***********/
  printGDS(4, result_list);
  /**********printGDS***********/

  return 0;
}