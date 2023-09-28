#include <boost/geometry.hpp>
#include <boost/polygon/polygon.hpp>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>

using irt_int = int32_t;
#define DBL_ERROR 1E-5

namespace gtl = boost::polygon;
using namespace boost::polygon::operators;
// using GTLPoint = gtl::point_data<irt_int>;
// using GTLRect = gtl::rectangle_data<irt_int>;
// using GTLPoly = gtl::polygon_90_data<irt_int>;
// using GTLPolySet = gtl::polygon_90_set_data<irt_int>;

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
// using BGPoint = bg::model::d2::point_xy<irt_int>;
// using BGMultiPoint = bg::model::multipoint<BGPoint>;
// using BGRect = bg::model::box<BGPoint>;
// using BGPoly = bg::model::polygon<BGPoint>;
// using BGMultiPoly = bg::model::multipolygon<BGPoly>;

#if 1

class PlanarCoord
{
 public:
  PlanarCoord() = default;
  PlanarCoord(const irt_int x, const irt_int y)
  {
    _x = x;
    _y = y;
  }
  ~PlanarCoord() = default;
  bool operator==(const PlanarCoord& other) const { return (_x == other._x && _y == other._y); }
  bool operator!=(const PlanarCoord& other) const { return !((*this) == other); }
  // getter
  irt_int get_x() const { return _x; }
  irt_int get_y() const { return _y; }
  // setter
  void set_x(const irt_int x) { _x = x; }
  void set_y(const irt_int y) { _y = y; }
  void set_coord(const irt_int x, const irt_int y)
  {
    _x = x;
    _y = y;
  }
  void set_coord(const PlanarCoord& coord) { set_coord(coord.get_x(), coord.get_y()); }
  // function
 private:
  irt_int _x = -1;
  irt_int _y = -1;
};

struct CmpPlanarCoordByXASC
{
  bool operator()(const PlanarCoord& a, const PlanarCoord& b) const
  {
    return a.get_x() != b.get_x() ? a.get_x() < b.get_x() : a.get_y() < b.get_y();
  }
};

struct CmpPlanarCoordByYASC
{
  bool operator()(const PlanarCoord& a, const PlanarCoord& b) const
  {
    return a.get_y() != b.get_y() ? a.get_y() < b.get_y() : a.get_x() < b.get_x();
  }
};

class PlanarRect
{
 public:
  PlanarRect() = default;
  PlanarRect(const PlanarCoord& lb, const PlanarCoord& rt)
  {
    _lb = lb;
    _rt = rt;
  }
  PlanarRect(const irt_int lb_x, const irt_int lb_y, const irt_int rt_x, const irt_int rt_y)
  {
    set_lb(lb_x, lb_y);
    set_rt(rt_x, rt_y);
  }
  ~PlanarRect() = default;
  bool operator==(const PlanarRect& other) const { return (_lb == other._lb && _rt == other._rt); }
  bool operator!=(const PlanarRect& other) const { return !((*this) == other); }
  // getter
  PlanarCoord& get_lb() { return _lb; }
  PlanarCoord& get_rt() { return _rt; }
  irt_int get_lb_x() const { return _lb.get_x(); }
  irt_int get_lb_y() const { return _lb.get_y(); }
  irt_int get_rt_x() const { return _rt.get_x(); }
  irt_int get_rt_y() const { return _rt.get_y(); }
  // const getter
  const PlanarCoord& get_lb() const { return _lb; }
  const PlanarCoord& get_rt() const { return _rt; }
  // setter
  void set_lb(const PlanarCoord& lb) { _lb = lb; }
  void set_rt(const PlanarCoord& rt) { _rt = rt; }
  void set_lb(const irt_int x, const irt_int y) { _lb.set_coord(x, y); }
  void set_rt(const irt_int x, const irt_int y) { _rt.set_coord(x, y); }
  void set_lb_x(const irt_int lb_x) { _lb.set_x(lb_x); }
  void set_lb_y(const irt_int lb_y) { _lb.set_y(lb_y); }
  void set_rt_x(const irt_int rt_x) { _rt.set_x(rt_x); }
  void set_rt_y(const irt_int rt_y) { _rt.set_y(rt_y); }
  // function

 private:
  PlanarCoord _lb;
  PlanarCoord _rt;
};

struct CmpPlanarRectByXASC
{
  bool operator()(const PlanarRect& a, const PlanarRect& b) const
  {
    if (a.get_lb() == b.get_lb()) {
      return CmpPlanarCoordByXASC()(a.get_rt(), b.get_rt());
    } else {
      return CmpPlanarCoordByXASC()(a.get_lb(), b.get_lb());
    }
  }
};

#endif

class RTUtil
{
  using GTLPoint = gtl::point_data<irt_int>;
  using GTLRect = gtl::rectangle_data<irt_int>;
  using GTLPoly = gtl::polygon_90_data<irt_int>;
  using GTLPolySet = gtl::polygon_90_set_data<irt_int>;

  using BGPoint = bg::model::d2::point_xy<double>;
  using BGMultiPoint = bg::model::multi_point<BGPoint>;
  using BGSegment = bg::model::segment<BGPoint>;
  using BGLine = bg::model::linestring<BGPoint>;
  using BGMultiLine = bg::model::multi_linestring<BGLine>;
  using BGRect = bg::model::box<BGPoint>;
  using BGPoly = bg::model::polygon<BGPoint>;
  using BGMultiPoly = bg::model::multi_polygon<BGPoly>;

 public:
  static void plotGDS(std::string gds_name, std::vector<PlanarRect>& rect_list)
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
        PlanarRect& rect = rect_list[i];

        gds_file << "BGNSTR" << std::endl;
        gds_file << "STRNAME rect_" << i << std::endl;
        irt_int lb_x = rect.get_lb_x();
        irt_int lb_y = rect.get_lb_y();
        irt_int rt_x = rect.get_rt_x();
        irt_int rt_y = rect.get_rt_y();

        gds_file << "BOUNDARY" << std::endl;
        gds_file << "LAYER " << 0 << std::endl;
        gds_file << "DATATYPE 0" << std::endl;
        gds_file << "XY" << std::endl;
        gds_file << lb_x << " : " << lb_y << std::endl;
        gds_file << rt_x << " : " << lb_y << std::endl;
        gds_file << rt_x << " : " << rt_y << std::endl;
        gds_file << lb_x << " : " << rt_y << std::endl;
        gds_file << lb_x << " : " << lb_y << std::endl;
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

  static void plotGDS(std::string gds_name, BGMultiPoly bg_multipoly)
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
      for (size_t i = 0; i < bg_multipoly.size(); i++) {
        BGPoly& bg_poly = bg_multipoly[i];

        gds_file << "BGNSTR" << std::endl;
        gds_file << "STRNAME bg_poly_" << i << std::endl;
        gds_file << "BOUNDARY" << std::endl;
        gds_file << "LAYER " << 0 << std::endl;
        gds_file << "DATATYPE 0" << std::endl;
        gds_file << "XY" << std::endl;

        for (size_t i = 0; i < bg::num_points(bg_poly); i++) {
          BGPoint bg_point = bg_poly.outer()[i];
          gds_file << int64_t(bg_point.x()) << " : " << int64_t(bg_point.y()) << std::endl;
        }
        gds_file << "ENDEL" << std::endl;

        gds_file << "ENDSTR" << std::endl;
      }
      // top
      gds_file << "BGNSTR" << std::endl;
      gds_file << "STRNAME top" << std::endl;
      // pin_point
      for (size_t i = 0; i < bg_multipoly.size(); i++) {
        gds_file << "SREF" << std::endl;
        gds_file << "SNAME bg_poly_" << i << std::endl;
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

  static void plotGDS(std::string gds_name, std::vector<BGPoly>& bg_poly_list)
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
      for (size_t i = 0; i < bg_poly_list.size(); i++) {
        BGPoly& bg_poly = bg_poly_list[i];

        gds_file << "BGNSTR" << std::endl;
        gds_file << "STRNAME bg_poly_" << i << std::endl;
        gds_file << "BOUNDARY" << std::endl;
        gds_file << "LAYER " << 0 << std::endl;
        gds_file << "DATATYPE 0" << std::endl;
        gds_file << "XY" << std::endl;

        for (size_t i = 0; i < bg::num_points(bg_poly); i++) {
          BGPoint bg_point = bg_poly.outer()[i];
          gds_file << int64_t(bg_point.x()) << " : " << int64_t(bg_point.y()) << std::endl;
        }
        gds_file << "ENDEL" << std::endl;

        gds_file << "ENDSTR" << std::endl;
      }
      // top
      gds_file << "BGNSTR" << std::endl;
      gds_file << "STRNAME top" << std::endl;
      // pin_point
      for (size_t i = 0; i < bg_poly_list.size(); i++) {
        gds_file << "SREF" << std::endl;
        gds_file << "SNAME bg_poly_" << i << std::endl;
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

  static void plotGDS(std::string gds_name, BGMultiPoint bg_multipoint)
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
      for (size_t i = 1; i < bg_multipoint.size(); i++) {
        BGPoint& pre_point = bg_multipoint[i - 1];
        BGPoint& curr_point = bg_multipoint[i];

        gds_file << "BGNSTR" << std::endl;
        gds_file << "STRNAME bg_seg_" << i << std::endl;

        gds_file << "PATH" << std::endl;
        gds_file << "LAYER " << i << std::endl;
        gds_file << "DATATYPE 0" << std::endl;
        gds_file << "WIDTH 1" << std::endl;
        gds_file << "XY" << std::endl;
        gds_file << pre_point.x() << " : " << pre_point.y() << std::endl;
        gds_file << curr_point.x() << " : " << curr_point.y() << std::endl;
        gds_file << "ENDEL" << std::endl;

        gds_file << "ENDSTR" << std::endl;
      }
      // top
      gds_file << "BGNSTR" << std::endl;
      gds_file << "STRNAME top" << std::endl;
      // pin_point
      for (size_t i = 1; i < bg_multipoint.size(); i++) {
        gds_file << "SREF" << std::endl;
        gds_file << "SNAME bg_seg_" << i << std::endl;
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

  static void addOffset(PlanarCoord& coord, irt_int x_offset, irt_int y_offset)
  {
    coord.set_x(coord.get_x() + x_offset);
    coord.set_y(coord.get_y() + y_offset);
  }

  static void addOffset(PlanarCoord& coord, PlanarCoord& offset_coord) { addOffset(coord, offset_coord.get_x(), offset_coord.get_y()); }

  static void minusOffset(PlanarCoord& coord, irt_int x_offset, irt_int y_offset)
  {
    coord.set_x((coord.get_x() - x_offset) < 0 ? 0 : (coord.get_x() - x_offset));
    coord.set_y((coord.get_y() - y_offset) < 0 ? 0 : (coord.get_y() - y_offset));
  }

  static void minusOffset(PlanarCoord& coord, PlanarCoord& offset_coord) { minusOffset(coord, offset_coord.get_x(), offset_coord.get_y()); }

  static std::vector<PlanarRect> getOpenReducedRectListByBoost(const std::vector<PlanarRect>& master_list, int lb_x_add_offset,
                                                               int lb_y_add_offset, int rt_x_minus_offset, int rt_y_minus_offset)
  {
    return getReducedRectListByBoost(master_list, lb_x_add_offset, lb_y_add_offset, rt_x_minus_offset, rt_y_minus_offset, true);
  }

  static std::vector<PlanarRect> getClosedReducedRectListByBoost(const std::vector<PlanarRect>& master_list, int lb_x_add_offset,
                                                                 int lb_y_add_offset, int rt_x_minus_offset, int rt_y_minus_offset)
  {
    return getReducedRectListByBoost(master_list, lb_x_add_offset, lb_y_add_offset, rt_x_minus_offset, rt_y_minus_offset, false);
  }

  static std::vector<PlanarRect> getReducedRectListByBoost(const std::vector<PlanarRect>& master_list, int lb_x_add_offset,
                                                           int lb_y_add_offset, int rt_x_minus_offset, int rt_y_minus_offset, bool is_open)
  {
    std::vector<PlanarRect> result_list;

    gtl::polygon_90_set_data<int> master_poly;
    for (const PlanarRect& master : master_list) {
      master_poly += gtl::rectangle_data<int>(master.get_lb_x(), master.get_lb_y(), master.get_rt_x(), master.get_rt_y());
    }
    if (!is_open) {
      // 提取点矩形，线段矩形
      std::vector<gtl::rectangle_data<int>> gtl_rect_list;
      gtl::get_rectangles(gtl_rect_list, master_poly, gtl::HORIZONTAL);
      gtl::get_rectangles(gtl_rect_list, master_poly, gtl::VERTICAL);

      std::vector<PlanarRect> candidate_rect_list;
      for (gtl::rectangle_data<int>& gtl_rect : gtl_rect_list) {
        candidate_rect_list.emplace_back(gtl::xl(gtl_rect), gtl::yl(gtl_rect), gtl::xh(gtl_rect), gtl::yh(gtl_rect));
      }
      for (PlanarRect candidate_rect : candidate_rect_list) {
        PlanarCoord& lb = candidate_rect.get_lb();
        PlanarCoord& rt = candidate_rect.get_rt();
        addOffset(lb, lb_x_add_offset, lb_y_add_offset);
        minusOffset(rt, rt_x_minus_offset, rt_y_minus_offset);
        if (lb.get_x() == rt.get_x() || lb.get_y() == rt.get_y()) {
          result_list.push_back(candidate_rect);
        }
      }
    }
    // 获得常规收缩的矩形
    {
      master_poly.shrink(lb_x_add_offset, rt_x_minus_offset, lb_y_add_offset, rt_y_minus_offset);

      std::vector<gtl::rectangle_data<int>> gtl_rect_list;
      gtl::get_rectangles(gtl_rect_list, master_poly, gtl::HORIZONTAL);
      gtl::get_rectangles(gtl_rect_list, master_poly, gtl::VERTICAL);

      for (gtl::rectangle_data<int>& gtl_rect : gtl_rect_list) {
        result_list.emplace_back(gtl::xl(gtl_rect), gtl::yl(gtl_rect), gtl::xh(gtl_rect), gtl::yh(gtl_rect));
      }
    }
    // rect去重
    std::sort(result_list.begin(), result_list.end(), CmpPlanarRectByXASC());
    result_list.erase(std::unique(result_list.begin(), result_list.end()), result_list.end());

    return result_list;
  }

 private:
};

int main()
{
  irt_int factor = 10;

  {
    std::vector<PlanarRect> master_list;
    master_list.emplace_back(1 * factor, 2 * factor, 6 * factor, 5 * factor);
    master_list.emplace_back(2 * factor, 4 * factor, 5 * factor, 7 * factor);

    RTUtil::plotGDS("master_list", master_list);

    std::vector<PlanarRect> reduced_result_list = RTUtil::getOpenReducedRectListByBoost(master_list, 5, 5, 5, 5);
    RTUtil::plotGDS("reduced_result_list", reduced_result_list);
  }

  {
    std::vector<PlanarRect> master_list;
    master_list.emplace_back(0 * factor, 0 * factor, 3 * factor, 4 * factor);
    master_list.emplace_back(0 * factor, 4 * factor, 3 * factor, 7 * factor);
    master_list.emplace_back(3 * factor, 0 * factor, 6 * factor, 7 * factor);

    RTUtil::plotGDS("master_list", master_list);

    std::vector<PlanarRect> reduced_result_list = RTUtil::getOpenReducedRectListByBoost(master_list, 5, 5, 5, 5);
    RTUtil::plotGDS("reduced_result_list", reduced_result_list);
  }

  {
    std::vector<PlanarRect> master_list;
    master_list.emplace_back(1 * factor, 0 * factor, 3 * factor, 4 * factor);
    master_list.emplace_back(3 * factor, 0 * factor, 4 * factor, 3 * factor);
    master_list.emplace_back(4 * factor, 2 * factor, 7 * factor, 3 * factor);
    master_list.emplace_back(1 * factor, 4 * factor, 2 * factor, 7 * factor);
    master_list.emplace_back(2 * factor, 4 * factor, 5 * factor, 5 * factor);
    master_list.emplace_back(2 * factor, 5 * factor, 6 * factor, 7 * factor);
    RTUtil::plotGDS("master_list", master_list);

    std::vector<PlanarRect> reduced_result_list = RTUtil::getOpenReducedRectListByBoost(master_list, 5, 5, 5, 5);
    RTUtil::plotGDS("reduced_result_list", reduced_result_list);
  }

  {
    std::vector<PlanarRect> master_list;
    master_list.emplace_back(1 * factor, 3 * factor, 3 * factor, 6 * factor);
    master_list.emplace_back(3 * factor, 4 * factor, 4 * factor, 6 * factor);
    master_list.emplace_back(0 * factor, 1 * factor, 5 * factor, 2 * factor);
    master_list.emplace_back(4 * factor, 1 * factor, 5 * factor, 3 * factor);
    master_list.emplace_back(4 * factor, 6 * factor, 7 * factor, 7 * factor);
    master_list.emplace_back(6 * factor, 1 * factor, 8 * factor, 6 * factor);
    RTUtil::plotGDS("master_list", master_list);

    std::vector<PlanarRect> reduced_result_list = RTUtil::getOpenReducedRectListByBoost(master_list, 5, 5, 5, 5);
    RTUtil::plotGDS("reduced_result_list", reduced_result_list);
  }

  {
    std::vector<PlanarRect> master_list;
    master_list.emplace_back(1 * factor, 3 * factor, 3 * factor, 6 * factor);

    RTUtil::plotGDS("master_list", master_list);

    std::vector<PlanarRect> reduced_result_list = RTUtil::getOpenReducedRectListByBoost(master_list, 5, 5, 5, 5);
    RTUtil::plotGDS("reduced_result_list", reduced_result_list);
  }

  return 0;
}