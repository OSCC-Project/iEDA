#include <boost/geometry.hpp>
#include <boost/polygon/polygon.hpp>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>

using int32_t = int32_t;
#define DBL_ERROR 1E-5

namespace gtl = boost::polygon;
using namespace boost::polygon::operators;

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

#if 1

class PlanarCoord
{
 public:
  PlanarCoord() = default;
  PlanarCoord(const int32_t x, const int32_t y)
  {
    _x = x;
    _y = y;
  }
  ~PlanarCoord() = default;
  bool operator==(const PlanarCoord& other) const { return (_x == other._x && _y == other._y); }
  bool operator!=(const PlanarCoord& other) const { return !((*this) == other); }
  // getter
  int32_t get_x() const { return _x; }
  int32_t get_y() const { return _y; }
  // setter
  void set_x(const int32_t x) { _x = x; }
  void set_y(const int32_t y) { _y = y; }
  void set_coord(const int32_t x, const int32_t y)
  {
    _x = x;
    _y = y;
  }
  void set_coord(const PlanarCoord& coord) { set_coord(coord.get_x(), coord.get_y()); }
  // function
 private:
  int32_t _x = -1;
  int32_t _y = -1;
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
  PlanarRect(const int32_t lb_x, const int32_t lb_y, const int32_t rt_x, const int32_t rt_y)
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
  int32_t get_lb_x() const { return _lb.get_x(); }
  int32_t get_lb_y() const { return _lb.get_y(); }
  int32_t get_rt_x() const { return _rt.get_x(); }
  int32_t get_rt_y() const { return _rt.get_y(); }
  // const getter
  const PlanarCoord& get_lb() const { return _lb; }
  const PlanarCoord& get_rt() const { return _rt; }
  // setter
  void set_lb(const PlanarCoord& lb) { _lb = lb; }
  void set_rt(const PlanarCoord& rt) { _rt = rt; }
  void set_lb(const int32_t x, const int32_t y) { _lb.set_coord(x, y); }
  void set_rt(const int32_t x, const int32_t y) { _rt.set_coord(x, y); }
  void set_lb_x(const int32_t lb_x) { _lb.set_x(lb_x); }
  void set_lb_y(const int32_t lb_y) { _lb.set_y(lb_y); }
  void set_rt_x(const int32_t rt_x) { _rt.set_x(rt_x); }
  void set_rt_y(const int32_t rt_y) { _rt.set_y(rt_y); }
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
  using GTLPointInt = gtl::point_data<int32_t>;
  using GTLRectInt = gtl::rectangle_data<int32_t>;
  using GTLPolyInt = gtl::polygon_90_data<int32_t>;
  using GTLPolySetInt = gtl::polygon_90_set_data<int32_t>;

  using BGPointDBL = bg::model::d2::point_xy<double>;
  using BGMultiPointDBL = bg::model::multi_point<BGPointDBL>;
  using BGSegmentDBL = bg::model::segment<BGPointDBL>;
  using BGLineDBL = bg::model::linestring<BGPointDBL>;
  using BGMultiLineDBL = bg::model::multi_linestring<BGLineDBL>;
  using BGRectDBL = bg::model::box<BGPointDBL>;
  using BGPolyDBL = bg::model::polygon<BGPointDBL>;
  using BGMultiPolyDBL = bg::model::multi_polygon<BGPolyDBL>;

 public:
#if 1  // exhibit

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
        int32_t lb_x = rect.get_lb_x();
        int32_t lb_y = rect.get_lb_y();
        int32_t rt_x = rect.get_rt_x();
        int32_t rt_y = rect.get_rt_y();

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

  static void plotGDS(std::string gds_name, BGMultiPolyDBL bg_multipoly)
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
        BGPolyDBL& bg_poly = bg_multipoly[i];

        gds_file << "BGNSTR" << std::endl;
        gds_file << "STRNAME bg_poly_" << i << std::endl;
        gds_file << "BOUNDARY" << std::endl;
        gds_file << "LAYER " << 0 << std::endl;
        gds_file << "DATATYPE 0" << std::endl;
        gds_file << "XY" << std::endl;

        for (size_t i = 0; i < bg::num_points(bg_poly); i++) {
          BGPointDBL bg_point = bg_poly.outer()[i];
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

  static void plotGDS(std::string gds_name, std::vector<BGPolyDBL>& bg_poly_list)
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
        BGPolyDBL& bg_poly = bg_poly_list[i];

        gds_file << "BGNSTR" << std::endl;
        gds_file << "STRNAME bg_poly_" << i << std::endl;
        gds_file << "BOUNDARY" << std::endl;
        gds_file << "LAYER " << 0 << std::endl;
        gds_file << "DATATYPE 0" << std::endl;
        gds_file << "XY" << std::endl;

        for (size_t i = 0; i < bg::num_points(bg_poly); i++) {
          BGPointDBL bg_point = bg_poly.outer()[i];
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

  static void plotGDS(std::string gds_name, BGMultiPointDBL bg_multipoint)
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
        BGPointDBL& pre_point = bg_multipoint[i - 1];
        BGPointDBL& curr_point = bg_multipoint[i];

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

#endif

#if 1  // cutting

  static std::vector<PlanarRect> getOpenCuttingRectListByBoost(const PlanarRect& master, const std::vector<PlanarRect>& rect_list)
  {
    std::vector<PlanarRect> master_list{master};
    return getCuttingRectListByBoost(master_list, rect_list, true);
  }

  static std::vector<PlanarRect> getOpenCuttingRectListByBoost(const std::vector<PlanarRect>& master_list,
                                                               const std::vector<PlanarRect>& rect_list)
  {
    return getCuttingRectListByBoost(master_list, rect_list, true);
  }

  static std::vector<PlanarRect> getClosedCuttingRectListByBoost(const std::vector<PlanarRect>& master_list,
                                                                 const std::vector<PlanarRect>& rect_list)
  {
    return getCuttingRectListByBoost(master_list, rect_list, false);
  }

  static std::vector<PlanarRect> getCuttingRectListByBoost(const std::vector<PlanarRect>& master_list,
                                                           const std::vector<PlanarRect>& rect_list, bool is_open)
  {
    std::vector<PlanarRect> result_list;

    if (!is_open) {
      // 先保存master_list中的特殊矩形
      for (const PlanarRect& master : master_list) {
        if (master.get_lb_x() == master.get_rt_x() || master.get_lb_y() == master.get_rt_y()) {
          // 特殊矩形
          result_list.push_back(master);
        }
      }
    }
    /**
     * 下面每个字母表示一个独立的直角多边形
     * 求解(A ∪ B) - (D ∪ E ∪ F)
     * 转((A - D) ∩ (A - E) ∩ (A - F)) ∪ ((B - D) ∩ (B - E) ∩ (B - F))
     * 其中利用(A - D)、(A - E)等式中结果不可能出现线，实现boost结果传递
     */
    // 将输入解析
    // 其中master_poly_list为(A ∪ B)
    std::vector<BGPolyDBL> master_poly_list = getBGPolyDBLList(master_list);
    // 其中rect_poly_list为(D ∪ E ∪ F)
    std::vector<BGPolyDBL> rect_poly_list = getBGPolyDBLList(rect_list);

    // 计算((A - D) ∩ (A - E) ∩ (A - F)) ∪ ((B - D) ∩ (B - E) ∩ (B - F))
    BGMultiPolyDBL top_multipoly;
    BGMultiLineDBL top_multiline;
    BGMultiPointDBL top_multipoint;
    for (BGPolyDBL& master_poly : master_poly_list) {
      // 计算(A - D)和(A - E)和(A - F)
      std::vector<BGMultiPolyDBL> diff_mutilpoly_list;
      {
        if (rect_poly_list.empty()) {
          BGMultiPolyDBL diff_mutilpoly;
          diff_mutilpoly.push_back(master_poly);
          diff_mutilpoly_list.push_back(diff_mutilpoly);
        } else {
          for (BGPolyDBL& rect_poly : rect_poly_list) {
            // 计算(A - D)
            BGMultiPolyDBL diff_mutilpoly;
            bg::difference(master_poly, rect_poly, diff_mutilpoly);
            if (diff_mutilpoly.empty()) {
              // 当(A - D)为空，后续(A - D) ∩ (A - E) ∩ (A - F)结果为空，直接跳过
              diff_mutilpoly_list.clear();
              break;
            } else {
              diff_mutilpoly_list.push_back(diff_mutilpoly);
            }
          }
        }
      }
      if (diff_mutilpoly_list.empty()) {
        continue;
      }
      // 计算(A - D) ∩ (A - E) ∩ (A - F)
      BGMultiPolyDBL mid_multipoly;
      BGMultiLineDBL mid_multiline;
      BGMultiPointDBL mid_multipoint;
      {
        // 用(A - D)初始化
        mid_multipoly = diff_mutilpoly_list.front();
        for (size_t i = 1; i < diff_mutilpoly_list.size(); i++) {
          BGMultiPolyDBL& curr_mutilpoly = diff_mutilpoly_list[i];
          // (A - D) ∩ (A - E)
          BGMultiPolyDBL mid_multipoly_temp;
          // 与顶层poly相交
          bg::intersection(mid_multipoly, curr_mutilpoly, mid_multipoly_temp);
          if (!is_open) {
            bg::intersection(mid_multipoly, curr_mutilpoly, mid_multiline);
            bg::intersection(mid_multipoly, curr_mutilpoly, mid_multipoint);
          }
          mid_multipoly = mid_multipoly_temp;
        }
      }
      // 计算((A - D) ∩ (A - E) ∩ (A - F)) ∪ ((B - D) ∩ (B - E) ∩ (B - F))
      {
        top_multipoly.insert(top_multipoly.end(), mid_multipoly.begin(), mid_multipoly.end());
        top_multiline.insert(top_multiline.end(), mid_multiline.begin(), mid_multiline.end());
        top_multipoint.insert(top_multipoint.end(), mid_multipoint.begin(), mid_multipoint.end());
      }
    }
    // 生成对应的矩形结果
    for (PlanarRect& rect : getRTRectListByBGMultiPolyDBL(top_multipoly)) {
      result_list.push_back(rect);
    }
    for (PlanarRect& rect : getRTRectListByBGMultiLineDBL(top_multiline)) {
      result_list.push_back(rect);
    }
    for (PlanarRect& rect : getRTRectListByBGMultiPointDBL(top_multipoint)) {
      result_list.push_back(rect);
    }
    // rect去重
    std::sort(result_list.begin(), result_list.end(), CmpPlanarRectByXASC());
    result_list.erase(std::unique(result_list.begin(), result_list.end()), result_list.end());
    return result_list;
  }

#endif

#if 1  // overlap

  static std::vector<PlanarRect> getOpenOverlapRectListByBoost(const std::vector<PlanarRect>& master_list,
                                                               const std::vector<PlanarRect>& rect_list)
  {
    return getOverlapRectListByBoost(master_list, rect_list, true);
  }

  static std::vector<PlanarRect> getClosedOverlapRectListByBoost(const std::vector<PlanarRect>& master_list,
                                                                 const std::vector<PlanarRect>& rect_list)
  {
    return getOverlapRectListByBoost(master_list, rect_list, false);
  }

  static std::vector<PlanarRect> getOverlapRectListByBoost(const std::vector<PlanarRect>& master_list,
                                                           const std::vector<PlanarRect>& rect_list, bool is_open)
  {
    std::vector<PlanarRect> result_list;

    if (!is_open) {
      // 先保存master_list中的特殊矩形
      for (const PlanarRect& master : master_list) {
        if (master.get_lb_x() == master.get_rt_x() || master.get_lb_y() == master.get_rt_y()) {
          // 特殊矩形
          result_list.push_back(master);
        }
      }
    }
    /**
     * 下面每个字母表示一个独立的直角多边形
     * 求解(A ∪ B) ∩ (D ∪ E ∪ F)
     * 转(A ∩ D) ∪ (A ∩ E) ∪ (A ∩ F) ∪ (B ∩ D) ∪ (B ∩ E) ∪ (B ∩ F)
     */
    // 其中master_poly_list为(A ∪ B)
    std::vector<BGPolyDBL> master_poly_list = getBGPolyDBLList(master_list);
    // 其中rect_poly_list为(D ∪ E ∪ F)
    std::vector<BGPolyDBL> rect_poly_list = getBGPolyDBLList(rect_list);

    BGMultiPolyDBL result_multipoly;
    BGMultiLineDBL result_multiline;
    BGMultiPointDBL result_multipoint;
    for (BGPolyDBL& master_poly : master_poly_list) {
      for (BGPolyDBL& rect_poly : rect_poly_list) {
        bg::intersection(master_poly, rect_poly, result_multipoly);
        if (!is_open) {
          bg::intersection(master_poly, rect_poly, result_multiline);
          bg::intersection(master_poly, rect_poly, result_multipoint);
        }
      }
    }
    // 生成对应的矩形结果
    for (PlanarRect& rect : getRTRectListByBGMultiPolyDBL(result_multipoly)) {
      result_list.push_back(rect);
    }
    for (PlanarRect& rect : getRTRectListByBGMultiLineDBL(result_multiline)) {
      result_list.push_back(rect);
    }
    for (PlanarRect& rect : getRTRectListByBGMultiPointDBL(result_multipoint)) {
      result_list.push_back(rect);
    }
    // rect去重
    std::sort(result_list.begin(), result_list.end(), CmpPlanarRectByXASC());
    result_list.erase(std::unique(result_list.begin(), result_list.end()), result_list.end());
    return result_list;
  }

#endif

#if 1  // reduce

  static std::vector<PlanarRect> getOpenReducedRectListByBoost(const std::vector<PlanarRect>& master_list, int lb_x_add_offset,
                                                               int lb_y_add_offset, int rt_x_minus_offset, int rt_y_minus_offset)
  {
    return getReducedRectListByBoost(master_list, lb_x_add_offset, lb_y_add_offset, rt_x_minus_offset, rt_y_minus_offset, true);
  }

  static std::vector<PlanarRect> getClosedReducedRectListByBoost(const std::vector<PlanarRect>& master_list, int reduced_offset)
  {
    return getReducedRectListByBoost(master_list, reduced_offset, reduced_offset, reduced_offset, reduced_offset, false);
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
        // 去除不是矩形的
        if (candidate_rect.isIncorrected()) {
          continue;
        }
        // 去除面积不为0的
        if (candidate_rect.getArea() > 0) {
          continue;
        }
        result_list.push_back(candidate_rect);
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

#endif

 private:
  static std::vector<BGPolyDBL> getBGPolyDBLList(const std::vector<PlanarRect>& rect_list)
  {
    std::vector<BGPolyDBL> bg_poly_list;
    for (const PlanarRect& rect : rect_list) {
      BGPolyDBL bg_poly;
      for (BGPointDBL& bg_point : getPointList(rect)) {
        bg::append(bg_poly.outer(), bg_point);
      }
      bg_poly_list.push_back(bg_poly);
    }
    return bg_poly_list;
  }

  static std::vector<BGPointDBL> getPointList(PlanarRect rect)
  {
    // 要求顺时针
    std::vector<BGPointDBL> point_list;
    point_list.emplace_back(rect.get_lb_x(), rect.get_lb_y());
    point_list.emplace_back(rect.get_lb_x(), rect.get_rt_y());
    point_list.emplace_back(rect.get_rt_x(), rect.get_rt_y());
    point_list.emplace_back(rect.get_rt_x(), rect.get_lb_y());
    point_list.emplace_back(rect.get_lb_x(), rect.get_lb_y());
    return point_list;
  }

  static int32_t getIntScale(double double_scale)
  {
    int32_t integer_scale = std::round(double_scale);
    if (std::abs(double_scale - integer_scale) > DBL_ERROR) {
      std::cout << "Exceeding the error range of a double!" << std::endl;
    }
    return integer_scale;
  }

  static std::vector<PlanarRect> getRTRectListByBGMultiPolyDBL(const BGMultiPolyDBL& bg_multipoly)
  {
    std::vector<PlanarRect> rect_list;

    GTLPolySetInt gtl_poly_set;
    for (const BGPolyDBL& bg_poly : bg_multipoly) {
      // 将double类型转int32_t
      std::vector<GTLPointInt> gtl_point_list;
      for (size_t i = 0; i < bg::num_points(bg_poly); i++) {
        BGPointDBL bg_point = bg_poly.outer()[i];
        gtl_point_list.emplace_back(getIntScale(bg_point.x()), getIntScale(bg_point.y()));
      }
      // 检查是否有斜线
      for (size_t i = 1; i < gtl_point_list.size(); i++) {
        GTLPointInt& pre_coord = gtl_point_list[i - 1];
        GTLPointInt& curr_coord = gtl_point_list[i];
        if (gtl::x(pre_coord) != gtl::x(curr_coord) && gtl::y(pre_coord) != gtl::y(curr_coord)) {
          std::cout << "The segment is oblique!" << std::endl;
        }
      }
      // 生成poly_90
      GTLPolyInt gtl_poly;
      gtl_poly.set(gtl_point_list.begin(), gtl_point_list.end());
      gtl_poly_set.insert(gtl_poly);
    }
    // 横竖切割
    std::vector<GTLRectInt> gtl_rect_list;
    gtl::get_rectangles(gtl_rect_list, gtl_poly_set, gtl::HORIZONTAL);
    gtl::get_rectangles(gtl_rect_list, gtl_poly_set, gtl::VERTICAL);
    for (GTLRectInt& gtl_rect : gtl_rect_list) {
      rect_list.emplace_back(gtl::xl(gtl_rect), gtl::yl(gtl_rect), gtl::xh(gtl_rect), gtl::yh(gtl_rect));
    }
    return rect_list;
  }

  static std::vector<PlanarRect> getRTRectListByBGMultiLineDBL(const BGMultiLineDBL& bg_multiline)
  {
    std::vector<PlanarRect> rect_list;

    for (const BGLineDBL& bg_line : bg_multiline) {
      PlanarCoord first_coord(getIntScale(bg_line[0].x()), getIntScale(bg_line[0].y()));
      PlanarCoord second_coord(getIntScale(bg_line[1].x()), getIntScale(bg_line[1].y()));
      rect_list.emplace_back(first_coord, second_coord);
    }

    return rect_list;
  }

  static std::vector<PlanarRect> getRTRectListByBGMultiPointDBL(const BGMultiPointDBL& bg_multipoint)
  {
    std::vector<PlanarRect> rect_list;

    for (const BGPointDBL& bg_point : bg_multipoint) {
      PlanarCoord coord(getIntScale(bg_point.x()), getIntScale(bg_point.y()));
      rect_list.emplace_back(coord, coord);
    }

    return rect_list;
  }
};

int main()
{
  int32_t factor = 10;
  std::vector<std::vector<PlanarRect>> master_list_list;
  std::vector<std::vector<PlanarRect>> rect_list_list;

  {
    std::vector<PlanarRect> master_list;
    master_list.emplace_back(2 * factor, 1 * factor, 5 * factor, 6 * factor);
    master_list_list.push_back(master_list);

    std::vector<PlanarRect> rect_list;
    rect_list_list.push_back(rect_list);
  }

  {
    std::vector<PlanarRect> master_list;
    master_list_list.push_back(master_list);

    std::vector<PlanarRect> rect_list;
    rect_list.emplace_back(0 * factor, 0 * factor, 3 * factor, 4 * factor);
    rect_list.emplace_back(0 * factor, 4 * factor, 3 * factor, 7 * factor);
    rect_list.emplace_back(3 * factor, 0 * factor, 6 * factor, 7 * factor);
    rect_list_list.push_back(rect_list);
  }

  // {
  //   std::vector<PlanarRect> master_list;
  //   master_list.emplace_back(2 * factor, 1 * factor, 5 * factor, 6 * factor);
  //   master_list_list.push_back(master_list);

  //   std::vector<PlanarRect> rect_list;
  //   rect_list.emplace_back(0 * factor, 0 * factor, 3 * factor, 4 * factor);
  //   rect_list.emplace_back(0 * factor, 4 * factor, 3 * factor, 7 * factor);
  //   rect_list.emplace_back(3 * factor, 0 * factor, 6 * factor, 7 * factor);
  //   rect_list_list.push_back(rect_list);
  // }

  // {
  //   std::vector<PlanarRect> master_list;
  //   master_list.emplace_back(2 * factor, 1 * factor, 5 * factor, 6 * factor);
  //   master_list_list.push_back(master_list);

  //   std::vector<PlanarRect> rect_list;
  //   rect_list.emplace_back(1 * factor, 0 * factor, 3 * factor, 4 * factor);
  //   rect_list.emplace_back(3 * factor, 0 * factor, 4 * factor, 3 * factor);
  //   rect_list.emplace_back(4 * factor, 2 * factor, 7 * factor, 3 * factor);
  //   rect_list.emplace_back(1 * factor, 4 * factor, 2 * factor, 7 * factor);
  //   rect_list.emplace_back(2 * factor, 4 * factor, 5 * factor, 5 * factor);
  //   rect_list.emplace_back(2 * factor, 5 * factor, 6 * factor, 7 * factor);
  //   rect_list_list.push_back(rect_list);
  // }

  // {
  //   std::vector<PlanarRect> master_list;
  //   master_list.emplace_back(1 * factor, 4 * factor, 5 * factor, 5 * factor);
  //   master_list.emplace_back(2 * factor, 0 * factor, 3 * factor, 7 * factor);
  //   master_list.emplace_back(3 * factor, 0 * factor, 4 * factor, 5 * factor);
  //   master_list.emplace_back(6 * factor, 0 * factor, 7 * factor, 7 * factor);
  //   master_list.emplace_back(5 * factor, 1 * factor, 7 * factor, 2 * factor);
  //   master_list.emplace_back(6 * factor, 4 * factor, 8 * factor, 5 * factor);
  //   master_list_list.push_back(master_list);

  //   std::vector<PlanarRect> rect_list;
  //   rect_list.emplace_back(1 * factor, 3 * factor, 3 * factor, 6 * factor);
  //   rect_list.emplace_back(3 * factor, 4 * factor, 4 * factor, 6 * factor);
  //   rect_list.emplace_back(0 * factor, 1 * factor, 5 * factor, 2 * factor);
  //   rect_list.emplace_back(4 * factor, 1 * factor, 5 * factor, 3 * factor);
  //   rect_list.emplace_back(4 * factor, 6 * factor, 7 * factor, 7 * factor);
  //   rect_list.emplace_back(6 * factor, 1 * factor, 8 * factor, 6 * factor);
  //   rect_list_list.push_back(rect_list);
  // }

  // {
  //   std::vector<PlanarRect> master_list;
  //   master_list.emplace_back(1 * factor, 5 * factor, 5 * factor, 5 * factor);
  //   master_list.emplace_back(5 * factor, 5 * factor, 5 * factor, 5 * factor);
  //   master_list.emplace_back(50 * factor, 50 * factor, 50 * factor, 50 * factor);
  //   master_list_list.push_back(master_list);

  //   std::vector<PlanarRect> rect_list;
  //   rect_list.emplace_back(1 * factor, 3 * factor, 3 * factor, 6 * factor);
  //   rect_list_list.push_back(rect_list);
  // }

  for (size_t i = 0; i < master_list_list.size(); i++) {
    std::vector<PlanarRect>& master_list = master_list_list[i];
    std::vector<PlanarRect>& rect_list = rect_list_list[i];

    RTUtil::plotGDS("1master_list", master_list);
    RTUtil::plotGDS("2rect_list", rect_list);

    std::vector<PlanarRect> OpenCuttingRectListByBoost = RTUtil::getOpenCuttingRectListByBoost(master_list, rect_list);
    RTUtil::plotGDS("3OpenCuttingRectListByBoost", OpenCuttingRectListByBoost);
    std::vector<PlanarRect> ClosedCuttingRectListByBoost = RTUtil::getClosedCuttingRectListByBoost(master_list, rect_list);
    RTUtil::plotGDS("4ClosedCuttingRectListByBoost", ClosedCuttingRectListByBoost);

    // std::vector<PlanarRect> OpenOverlapRectListByBoost = RTUtil::getOpenOverlapRectListByBoost(master_list, rect_list);
    // RTUtil::plotGDS("5OpenOverlapRectListByBoost", OpenOverlapRectListByBoost);
    // std::vector<PlanarRect> ClosedOverlapRectListByBoost = RTUtil::getClosedOverlapRectListByBoost(master_list, rect_list);
    // RTUtil::plotGDS("6ClosedOverlapRectListByBoost", ClosedOverlapRectListByBoost);

    // std::vector<PlanarRect> master_OpenReducedRectListByBoost = RTUtil::getOpenReducedRectListByBoost(master_list, 5, 5, 5, 5);
    // RTUtil::plotGDS("7master_OpenReducedRectListByBoost", master_OpenReducedRectListByBoost);
    // std::vector<PlanarRect> master_ClosedReducedRectListByBoost = RTUtil::getClosedReducedRectListByBoost(master_list, 5, 5, 5, 5);
    // RTUtil::plotGDS("8master_ClosedReducedRectListByBoost", master_ClosedReducedRectListByBoost);

    // std::vector<PlanarRect> rect_OpenReducedRectListByBoost = RTUtil::getOpenReducedRectListByBoost(rect_list, 5, 5, 5, 5);
    // RTUtil::plotGDS("9rect_OpenReducedRectListByBoost", rect_OpenReducedRectListByBoost);
    // std::vector<PlanarRect> rect_ClosedReducedRectListByBoost = RTUtil::getClosedReducedRectListByBoost(rect_list, 5, 5, 5, 5);
    // RTUtil::plotGDS("91rect_ClosedReducedRectListByBoost", rect_ClosedReducedRectListByBoost);
    int a = 0;
  }

  return 0;
}