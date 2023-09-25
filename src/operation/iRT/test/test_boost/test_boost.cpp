#if 0

#include <boost/geometry.hpp>
#include <boost/polygon/polygon.hpp>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>

namespace gtl = boost::polygon;
using namespace boost::polygon::operators;
using GTLPoint = gtl::point_data<int>;
using GTLRectangle = gtl::rectangle_data<int>;
using GTLPolygon = gtl::polygon_90_data<int>;
using GTLPolygonSet = gtl::polygon_90_set_data<int>;

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using BGPoint = bg::model::d2::point_xy<int>;
using BGMultiPoint = bg::model::multi_point<BGPoint>;

using BGBox = bg::model::box<BGPoint>;

using BGPolygon = bg::model::polygon<BGPoint>;
using BGMultiPolygon = bg::model::multi_polygon<BGPolygon>;

////////////////////////////////////////////////////////////////////////////////
class PlanarCoord
{
 public:
  PlanarCoord() = default;
  PlanarCoord(const int x, const int y)
  {
    _x = x;
    _y = y;
  }
  ~PlanarCoord() = default;
  bool operator==(const PlanarCoord& other) const { return (_x == other._x && _y == other._y); }
  bool operator!=(const PlanarCoord& other) const { return !((*this) == other); }
  // getter
  int get_x() const { return _x; }
  int get_y() const { return _y; }
  // setter
  void set_x(const int x) { _x = x; }
  void set_y(const int y) { _y = y; }
  void set_coord(const int x, const int y)
  {
    _x = x;
    _y = y;
  }
  void set_coord(const PlanarCoord& coord) { set_coord(coord.get_x(), coord.get_y()); }
  // function
 private:
  int _x = -1;
  int _y = -1;
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
  PlanarRect(const int lb_x, const int lb_y, const int rt_x, const int rt_y)
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
  int get_lb_x() const { return _lb.get_x(); }
  int get_lb_y() const { return _lb.get_y(); }
  int get_rt_x() const { return _rt.get_x(); }
  int get_rt_y() const { return _rt.get_y(); }
  // const getter
  const PlanarCoord& get_lb() const { return _lb; }
  const PlanarCoord& get_rt() const { return _rt; }
  // setter
  void set_lb(const PlanarCoord& lb) { _lb = lb; }
  void set_rt(const PlanarCoord& rt) { _rt = rt; }
  void set_lb(const int x, const int y) { _lb.set_coord(x, y); }
  void set_rt(const int x, const int y) { _rt.set_coord(x, y); }
  void set_lb_x(const int lb_x) { _lb.set_x(lb_x); }
  void set_lb_y(const int lb_y) { _lb.set_y(lb_y); }
  void set_rt_x(const int rt_x) { _rt.set_x(rt_x); }
  void set_rt_y(const int rt_y) { _rt.set_y(rt_y); }
  // function
  inline int getXSpan() const;
  inline int getYSpan() const;
  inline int getLength() const;
  inline int getWidth() const;
  inline int getHalfPerimeter() const;
  inline int getPerimeter() const;
  inline double getArea() const;
  inline PlanarCoord getMidPoint() const;

 private:
  PlanarCoord _lb;
  PlanarCoord _rt;
};

inline int PlanarRect::getXSpan() const
{
  return get_rt_x() - get_lb_x();
}

inline int PlanarRect::getYSpan() const
{
  return get_rt_y() - get_lb_y();
}

inline int PlanarRect::getLength() const
{
  return std::max(getXSpan(), getYSpan());
}

inline int PlanarRect::getWidth() const
{
  return std::min(getXSpan(), getYSpan());
}

inline int PlanarRect::getHalfPerimeter() const
{
  return getLength() + getWidth();
}

inline int PlanarRect::getPerimeter() const
{
  return getHalfPerimeter() * 2;
}

inline double PlanarRect::getArea() const
{
  double area = getLength();
  area *= getWidth();
  return area;
}

inline PlanarCoord PlanarRect::getMidPoint() const
{
  return PlanarCoord((get_lb_x() + get_rt_x()) / 2, (get_lb_y() + get_rt_y()) / 2);
}

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

struct CmpPlanarRectByYASC
{
  bool operator()(const PlanarRect& a, const PlanarRect& b) const
  {
    if (a.get_lb() == b.get_lb()) {
      return CmpPlanarCoordByYASC()(a.get_rt(), b.get_rt());
    } else {
      return CmpPlanarCoordByYASC()(a.get_lb(), b.get_lb());
    }
  }
};
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

std::vector<PlanarRect> getClosedOverlapByBoost(const PlanarRect& master, const std::vector<PlanarRect>& rect_list)
{
  return getClosedOverlapByBoost({master}, rect_list);
}

std::vector<PlanarRect> getClosedOverlapByBoost(const std::vector<PlanarRect>& master_list, const PlanarRect& rect)
{
  return getClosedOverlapByBoost(master_list, {rect});
}

std::vector<PlanarRect> getClosedOverlapByBoost(const std::vector<PlanarRect>& master_list, const std::vector<PlanarRect>& rect_list)
{
  BGMultiPolygon master_poly = convertToBGMultiPolygon(master_list);
  BGMultiPolygon rect_poly = convertToBGMultiPolygon(rect_list);

  BGMultiPolygon result_multi_poly;
  bg::intersection(master_poly, rect_poly, result_multi_poly);
  std::vector<PlanarRect> result1 = convertToRTRect(result_multi_poly);

  BGMultiPoint bg_multi_point;
  bg::intersection(master_poly, rect_poly, bg_multi_point);
  std::vector<PlanarRect> result2 = getRectListByPoint(convertToRTPoint(bg_multi_point));

  std::vector<PlanarRect> overlap_rect_list;
  overlap_rect_list.insert(overlap_rect_list.end(), result1.begin(), result1.end());
  overlap_rect_list.insert(overlap_rect_list.end(), result2.begin(), result2.end());

  return overlap_rect_list;
}

std::vector<PlanarRect> getClosedReducedRectByBoost(const PlanarRect& rect, int reduce_size)
{
  std::vector<PlanarRect> rect_list = {rect};
  return getClosedReducedRectByBoost(rect_list, reduce_size);
}

std::vector<PlanarRect> getClosedReducedRectByBoost(const PlanarRect& rect, int lb_x_add_offset, int lb_y_add_offset, int rt_x_minus_offset,
                                                    int rt_y_minus_offset)
{
  std::vector<PlanarRect> rect_list = {rect};
  return getClosedReducedRectByBoost(rect_list, lb_x_add_offset, lb_y_add_offset, rt_x_minus_offset, rt_y_minus_offset);
}

std::vector<PlanarRect> getClosedReducedRectByBoost(const std::vector<PlanarRect>& rect_list, int reduce_size)
{
  return getClosedReducedRectByBoost(rect_list, reduce_size, reduce_size, reduce_size, reduce_size);
}

std::vector<PlanarRect> getClosedReducedRectByBoost(const std::vector<PlanarRect>& rect_list, int lb_x_add_offset, int lb_y_add_offset,
                                                    int rt_x_minus_offset, int rt_y_minus_offset)
{
  GTLPolygonSet rect_poly;
  for (const PlanarRect& rect : rect_list) {
    rect_poly += convertToGTLRect(rect);
  }

  GTLPolygonSet reduced_poly = gtl::shrink(rect_poly, lb_x_add_offset, rt_x_minus_offset, lb_y_add_offset, rt_y_minus_offset);

  std::vector<GTLRectangle> gtl_rect_list;
  gtl::get_rectangles(gtl_rect_list, reduced_poly);

  std::vector<PlanarRect> reduced_rect_list;
  for (gtl::rectangle_data<int>& gtl_rect : gtl_rect_list) {
    reduced_rect_list.emplace_back(convertToPlanarRect(gtl_rect));
  }

  if (!reduced_rect_list.empty()) {
    return reduced_rect_list;
  }

  for (PlanarRect reduced_rect : rect_list) {
    addOffset(reduced_rect.get_lb(), lb_x_add_offset, lb_y_add_offset);
    minusOffset(reduced_rect.get_rt(), rt_x_minus_offset, rt_y_minus_offset);
    if (reduced_rect.get_lb_x() <= reduced_rect.get_rt_x() && reduced_rect.get_lb_y() <= reduced_rect.get_rt_y()) {
      reduced_rect_list.push_back(reduced_rect);
    }
  }
  return reduced_rect_list;
}

PlanarRect convertToPlanarRect(BGBox& boost_box)
{
  return PlanarRect(boost_box.min_corner().x(), boost_box.min_corner().y(), boost_box.max_corner().x(), boost_box.max_corner().y());
}

gtl::rectangle_data<int> convertToGTLRect(const PlanarRect& rect)
{
  return gtl::rectangle_data<int>(rect.get_lb_x(), rect.get_lb_y(), rect.get_rt_x(), rect.get_rt_y());
}

gtl::rectangle_data<int> convertToGTLRect(BGBox& boost_box)
{
  return gtl::rectangle_data<int>(boost_box.min_corner().x(), boost_box.min_corner().y(), boost_box.max_corner().x(),
                                  boost_box.max_corner().y());
}

void addOffset(PlanarCoord& coord, PlanarCoord& offset_coord)
{
  addOffset(coord, offset_coord.get_x(), offset_coord.get_y());
}

void addOffset(PlanarCoord& coord, int x_offset, int y_offset)
{
  coord.set_x(coord.get_x() + x_offset);
  coord.set_y(coord.get_y() + y_offset);
}

void minusOffset(PlanarCoord& coord, PlanarCoord& offset_coord)
{
  minusOffset(coord, offset_coord.get_x(), offset_coord.get_y());
}

void minusOffset(PlanarCoord& coord, int x_offset, int y_offset)
{
  coord.set_x((coord.get_x() - x_offset) < 0 ? 0 : (coord.get_x() - x_offset));
  coord.set_y((coord.get_y() - y_offset) < 0 ? 0 : (coord.get_y() - y_offset));
}

/**
 *  切割矩形，将master矩形用rect进行切割，求差集
 *       ┌────────────────────────────────────┐
 *       │ master                             │
 *       │           ┌─────────────────┐      │
 *       └───────────┼─────────────────┼──────┘
 *                   │ rect            │
 *        cut  │     └─────────────────┘  │cut
 *             ▼                          ▼
 *       ┌───────────┐┌────────────────┐┌──────┐
 *       │           ││       c        ││      │
 *       │     a     │└────────────────┘│  b   │
 *       └───────────┘                  └──────┘
 *  如上图所示，输入master和rect，切割后得到a b c三个矩形
 */
std::vector<PlanarRect> getClosedCuttingRectListByBoost(const PlanarRect& master, const PlanarRect& rect)
{
  std::vector<PlanarRect> master_list = {master};
  std::vector<PlanarRect> rect_list = {rect};
  return getClosedCuttingRectListByBoost(master_list, rect_list);
}

std::vector<PlanarRect> getClosedCuttingRectListByBoost(const PlanarRect& master, const std::vector<PlanarRect>& rect_list)
{
  std::vector<PlanarRect> master_list = {master};
  return getClosedCuttingRectListByBoost(master_list, rect_list);
}

std::vector<PlanarRect> getClosedCuttingRectListByBoost(const std::vector<PlanarRect>& master_list, const PlanarRect& rect)
{
  std::vector<PlanarRect> rect_list = {rect};
  return getClosedCuttingRectListByBoost(master_list, rect_list);
}

std::vector<PlanarRect> getClosedCuttingRectListByBoost(const std::vector<PlanarRect>& master_list,
                                                        const std::vector<PlanarRect>& rect_list)
{
  BGMultiPolygon master_poly = convertToBGMultiPolygon(master_list);
  BGMultiPolygon cutting_poly = convertToBGMultiPolygon(rect_list);
  bg::correct(master_poly);
  bg::correct(cutting_poly);

  BGMultiPolygon result_multi_poly;
  bg::difference(master_poly, cutting_poly, result_multi_poly);
  std::vector<PlanarRect> result1 = convertToRTRect(result_multi_poly);

  BGMultiPoint bg_multi_point;
  bg::difference(master_poly, cutting_poly, bg_multi_point);
  std::vector<PlanarRect> result2 = getRectListByPoint(convertToRTPoint(bg_multi_point));

  std::vector<PlanarRect> cutting_rect_list;
  cutting_rect_list.insert(cutting_rect_list.end(), result1.begin(), result1.end());
  cutting_rect_list.insert(cutting_rect_list.end(), result2.begin(), result2.end());
  return cutting_rect_list;
}

BGMultiPolygon convertToBGMultiPolygon(const std::vector<PlanarRect>& rect_list)
{
  GTLPolygonSet gtl_poly_set;
  for (const PlanarRect& rect : rect_list) {
    gtl_poly_set += convertToGTLRect(rect);
  }
  std::vector<GTLPolygon> gtl_poly_list;
  gtl_poly_set.get_polygons(gtl_poly_list);

  BGMultiPolygon bg_multi_poly;
  for (GTLPolygon& gtl_poly : gtl_poly_list) {
    bg_multi_poly.push_back(convertToBGPolygon(gtl_poly));
  }
  return bg_multi_poly;
}

BGPolygon convertToBGPolygon(const GTLPolygon& gtl_poly)
{
  std::vector<BGPoint> bg_point_list;
  for (auto iter = gtl_poly.begin(); iter != gtl_poly.end(); iter++) {
    bg_point_list.emplace_back(gtl::x(*iter), gtl::y(*iter));
  }
  if (bg_point_list.empty()) {
    LOG_INST.error(Loc::current(), "GTLPolygon is empty!");
  }
  bg_point_list.push_back(bg_point_list.front());
  BGPolygon bg_poly;
  for (BGPoint& bg_point : bg_point_list) {
    bg::append(bg_poly.outer(), bg_point);
  }
  return bg_poly;
}

std::vector<PlanarRect> convertToRTRect(BGMultiPolygon bg_multi_poly)
{
  GTLPolygonSet gtl_polygon_set;
  for (BGPolygon& bg_poly : bg_multi_poly) {
    gtl_polygon_set.insert(convertToGTLPolygon(bg_poly));
  }

  std::vector<GTLRectangle> gtl_rect_list;
  gtl::get_rectangles(gtl_rect_list, gtl_polygon_set);

  std::vector<PlanarRect> rect_list;
  for (GTLRectangle& gtl_rect : gtl_rect_list) {
    PlanarCoord lb(gtl::xl(gtl_rect), gtl::yl(gtl_rect));
    PlanarCoord rt(gtl::xh(gtl_rect), gtl::yh(gtl_rect));
    rect_list.emplace_back(lb, rt);
  }
  return rect_list;
}

GTLPolygon convertToGTLPolygon(BGPolygon& bg_poly)
{
  std::vector<GTLPoint> gtl_point_list;
  for (size_t i = 0; i < bg::num_points(bg_poly); i++) {
    BGPoint bg_point = bg_poly.outer()[i];
    gtl_point_list.emplace_back(bg_point.x(), bg_point.y());
  }

  GTLPolygon gtl_poly;
  gtl_poly.set(gtl_point_list.begin(), gtl_point_list.end());
  return gtl_poly;
}

std::vector<PlanarCoord> convertToRTPoint(BGMultiPoint& bg_multi_point)
{
  std::vector<PlanarCoord> rt_point_list;
  for (BGPoint& bg_point : bg_multi_point) {
    rt_point_list.emplace_back(bg_point.x(), bg_point.y());
  }
  return rt_point_list;
}

std::vector<PlanarRect> getRectListByPoint(const std::vector<PlanarCoord>& point_list)
{
  std::vector<PlanarRect> rect_list;

  std::map<int, std::set<int>> x_y_set_map;
  std::map<int, std::set<int>> y_x_set_map;
  for (const PlanarCoord& point : point_list) {
    x_y_set_map[point.get_x()].insert(point.get_y());
    y_x_set_map[point.get_y()].insert(point.get_x());
  }

  for (auto& [x, y_set] : x_y_set_map) {
    if (y_set.size() > 1) {
      PlanarCoord lb(x, *(y_set.begin()));
      PlanarCoord rt(x, *(y_set.rbegin()));
      rect_list.emplace_back(lb, rt);
    }
  }

  for (auto& [y, x_set] : y_x_set_map) {
    if (x_set.size() > 1) {
      PlanarCoord lb(*(x_set.begin()), y);
      PlanarCoord rt(*(x_set.rbegin()), y);
      rect_list.emplace_back(lb, rt);
    }
  }
  return rect_list;
}
#endif