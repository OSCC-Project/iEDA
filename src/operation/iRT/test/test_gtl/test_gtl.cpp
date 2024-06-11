#include <boost/geometry.hpp>
#include <boost/polygon/polygon.hpp>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>

using int32_t = int32_t;

namespace gtl = boost::polygon;
using namespace boost::polygon::operators;

using GTLPointInt = gtl::point_data<int32_t>;
using GTLRectInt = gtl::rectangle_data<int32_t>;
using GTLPolyInt = gtl::polygon_90_data<int32_t>;
using GTLPolySetInt = gtl::polygon_90_set_data<int32_t>;

#if 1  // exhibit

void plotGDS(std::string gds_name, std::vector<GTLRectInt>& rect_list)
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
    for (size_t i = 0; i < rect_list.size(); i++) {
      GTLRectInt& rect = rect_list[i];

      gds_file << "BGNSTR" << std::endl;
      gds_file << "STRNAME rect_" << i << std::endl;
      int32_t ll_x = gtl::xl(rect);
      int32_t ll_y = gtl::yl(rect);
      int32_t ur_x = gtl::xh(rect);
      int32_t ur_y = gtl::yh(rect);

      gds_file << "BOUNDARY" << std::endl;
      gds_file << "LAYER " << 0 << std::endl;
      gds_file << "DATATYPE 0" << std::endl;
      gds_file << "XY" << std::endl;
      gds_file << ll_x << " : " << ll_y << std::endl;
      gds_file << ur_x << " : " << ll_y << std::endl;
      gds_file << ur_x << " : " << ur_y << std::endl;
      gds_file << ll_x << " : " << ur_y << std::endl;
      gds_file << ll_x << " : " << ll_y << std::endl;
      gds_file << "ENDEL" << std::endl;

      gds_file << "ENDSTR" << std::endl;
    }
    // top
    gds_file << "BGNSTR" << std::endl;
    gds_file << "STRNAME top" << std::endl;
    // pin_point
    for (size_t i = 0; i < rect_list.size(); i++) {
      gds_file << "SREF" << std::endl;
      gds_file << "SNAME rect_" << i << std::endl;
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

#endif

#if 1  // cutting

std::vector<GTLRectInt> getCuttingRectListByBoost(std::vector<GTLRectInt>& master_list, std::vector<GTLRectInt>& rect_list)
{
  GTLPolySetInt master_poly;
  for (GTLRectInt& master : master_list) {
    master_poly += master;
  }

  GTLPolySetInt rect_poly;
  for (GTLRectInt& rect : rect_list) {
    rect_poly += rect;
  }

  master_poly -= rect_poly;

  std::vector<GTLRectInt> result_list;
  gtl::get_rectangles(result_list, master_poly);
  return result_list;
}

#endif

#if 1  // overlap

std::vector<GTLRectInt> getOverlapRectListByBoost(std::vector<GTLRectInt>& master_list, std::vector<GTLRectInt>& rect_list)
{
  GTLPolySetInt master_poly;
  for (GTLRectInt& master : master_list) {
    master_poly += master;
  }

  GTLPolySetInt rect_poly;
  for (GTLRectInt& rect : rect_list) {
    rect_poly += rect;
  }

  master_poly *= rect_poly;

  std::vector<GTLRectInt> result_list;
  gtl::get_rectangles(result_list, master_poly);
  return result_list;
}

#endif

#if 1  // reduce

#endif

int32_t main()
{
  int32_t factor = 100000000;
  std::vector<std::vector<GTLRectInt>> master_list_list;
  std::vector<std::vector<GTLRectInt>> rect_list_list;

  {
    std::vector<GTLRectInt> master_list;
    master_list.emplace_back(2 * factor, 1 * factor, 5 * factor, 6 * factor);
    master_list_list.push_back(master_list);

    std::vector<GTLRectInt> rect_list;
    rect_list_list.push_back(rect_list);
  }

  {
    std::vector<GTLRectInt> master_list;
    master_list_list.push_back(master_list);

    std::vector<GTLRectInt> rect_list;
    rect_list.emplace_back(0 * factor, 0 * factor, 3 * factor, 4 * factor);
    rect_list.emplace_back(0 * factor, 4 * factor, 3 * factor, 7 * factor);
    rect_list.emplace_back(3 * factor, 0 * factor, 6 * factor, 7 * factor);
    rect_list_list.push_back(rect_list);
  }

  {
    std::vector<GTLRectInt> master_list;
    master_list.emplace_back(2 * factor, 1 * factor, 5 * factor, 6 * factor);
    master_list_list.push_back(master_list);

    std::vector<GTLRectInt> rect_list;
    rect_list.emplace_back(0 * factor, 0 * factor, 3 * factor, 4 * factor);
    rect_list.emplace_back(0 * factor, 4 * factor, 3 * factor, 7 * factor);
    rect_list.emplace_back(3 * factor, 0 * factor, 6 * factor, 7 * factor);
    rect_list_list.push_back(rect_list);
  }

  {
    std::vector<GTLRectInt> master_list;
    master_list.emplace_back(2 * factor, 1 * factor, 5 * factor, 6 * factor);
    master_list_list.push_back(master_list);

    std::vector<GTLRectInt> rect_list;
    rect_list.emplace_back(1 * factor, 0 * factor, 3 * factor, 4 * factor);
    rect_list.emplace_back(3 * factor, 0 * factor, 4 * factor, 3 * factor);
    rect_list.emplace_back(4 * factor, 2 * factor, 7 * factor, 3 * factor);
    rect_list.emplace_back(1 * factor, 4 * factor, 2 * factor, 7 * factor);
    rect_list.emplace_back(2 * factor, 4 * factor, 5 * factor, 5 * factor);
    rect_list.emplace_back(2 * factor, 5 * factor, 6 * factor, 7 * factor);
    rect_list_list.push_back(rect_list);
  }

  {
    std::vector<GTLRectInt> master_list;
    master_list.emplace_back(1 * factor, 4 * factor, 5 * factor, 5 * factor);
    master_list.emplace_back(2 * factor, 0 * factor, 3 * factor, 7 * factor);
    master_list.emplace_back(3 * factor, 0 * factor, 4 * factor, 5 * factor);
    master_list.emplace_back(6 * factor, 0 * factor, 7 * factor, 7 * factor);
    master_list.emplace_back(5 * factor, 1 * factor, 7 * factor, 2 * factor);
    master_list.emplace_back(6 * factor, 4 * factor, 8 * factor, 5 * factor);
    master_list_list.push_back(master_list);

    std::vector<GTLRectInt> rect_list;
    rect_list.emplace_back(1 * factor, 3 * factor, 3 * factor, 6 * factor);
    rect_list.emplace_back(3 * factor, 4 * factor, 4 * factor, 6 * factor);
    rect_list.emplace_back(0 * factor, 1 * factor, 5 * factor, 2 * factor);
    rect_list.emplace_back(4 * factor, 1 * factor, 5 * factor, 3 * factor);
    rect_list.emplace_back(4 * factor, 6 * factor, 7 * factor, 7 * factor);
    rect_list.emplace_back(6 * factor, 1 * factor, 8 * factor, 6 * factor);
    rect_list_list.push_back(rect_list);
  }

  for (size_t i = 0; i < master_list_list.size(); i++) {
    std::vector<GTLRectInt>& master_list = master_list_list[i];
    std::vector<GTLRectInt>& rect_list = rect_list_list[i];

    plotGDS("1master_list", master_list);
    plotGDS("2rect_list", rect_list);

    std::vector<GTLRectInt> CuttingRectListByBoost = getCuttingRectListByBoost(master_list, rect_list);
    plotGDS("3CuttingRectListByBoost", CuttingRectListByBoost);

    std::vector<GTLRectInt> OverlapRectListByBoost = getOverlapRectListByBoost(master_list, rect_list);
    plotGDS("4OverlapRectListByBoost", OverlapRectListByBoost);

    int32_t a = 0;
  }

  return 0;
}