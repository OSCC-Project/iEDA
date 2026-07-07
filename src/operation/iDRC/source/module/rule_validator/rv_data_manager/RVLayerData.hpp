#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <vector>

#include "PlanarRect.hpp"

namespace boost::geometry::traits {

template <>
struct tag<GTLRectInt>
{
  using type = box_tag;
};

template <>
struct point_type<GTLRectInt>
{
  using type = BGPointInt;
};

template <>
struct indexed_access<GTLRectInt, min_corner, 0>
{
  static int32_t get(GTLRectInt const& r) { return gtl::xl(r); }
  static void set(GTLRectInt& r, int32_t const& v) { r = GTLRectInt(v, gtl::yl(r), gtl::xh(r), gtl::yh(r)); }
};

template <>
struct indexed_access<GTLRectInt, min_corner, 1>
{
  static int32_t get(GTLRectInt const& r) { return gtl::yl(r); }
  static void set(GTLRectInt& r, int32_t const& v) { r = GTLRectInt(gtl::xl(r), v, gtl::xh(r), gtl::yh(r)); }
};

template <>
struct indexed_access<GTLRectInt, max_corner, 0>
{
  static int32_t get(GTLRectInt const& r) { return gtl::xh(r); }
  static void set(GTLRectInt& r, int32_t const& v) { r = GTLRectInt(gtl::xl(r), gtl::yl(r), v, gtl::yh(r)); }
};

template <>
struct indexed_access<GTLRectInt, max_corner, 1>
{
  static int32_t get(GTLRectInt const& r) { return gtl::yh(r); }
  static void set(GTLRectInt& r, int32_t const& v) { r = GTLRectInt(gtl::xl(r), gtl::yl(r), gtl::xh(r), v); }
};
}  // namespace boost::geometry::traits

namespace idrc {

struct CutData
{
  GTLRectInt rect;
  int32_t net_idx = -1;
  bool isEnv = false;

  bool operator==(const CutData& other) const = default;
};

struct CutDataIndexable
{
  using result_type = GTLRectInt const&;

  GTLRectInt const& operator()(CutData const& cut_data) const { return cut_data.rect; }
};

struct MaxRectData
{
  GTLRectInt rect;
  int32_t polygon_id = -1;
  bool isEnv = false;
};

struct BoundaryData
{
  GTLRectInt edge;
  PlanarCoord begin_coord;
  PlanarCoord end_coord;
  Orientation orient = Orientation::kNone;
  int32_t polygon_id;
  int32_t prev_boundary_id = -1;
  int32_t next_boundary_id = -1;
  int32_t edge_length = 0;
  bool isConvex = false;
  bool isHole = false;
  bool isEnv = false;
};

struct PolygonData
{
  int32_t net_id = -1;
  int32_t max_rect_begin = 0;
  int32_t max_rect_count = 0;
  int32_t boundary_begin = 0;
  int32_t boundary_count = 0;
  GTLHolePolyInt hole_poly;
  bool isEnv = false;
};

struct RVRoutingNet
{
  GTLPolySetInt polyset;  // env + result;
  int32_t polygon_begin = 0;
  int32_t polygon_count = 0;
  int32_t max_rect_begin = 0;
  int32_t max_rect_count = 0;
  int32_t boundary_begin = 0;
  int32_t boundary_count = 0;
};

struct RVLayerData
{
  std::map<int32_t, RVRoutingNet> nets;
  std::vector<CutData> cut_pool;
  std::vector<PolygonData> polygon_pool;
  std::vector<MaxRectData> max_rect_pool;
  std::vector<BoundaryData> boundary_pool;
  bgi::rtree<std::pair<GTLRectInt, int32_t>, bgi::quadratic<16>> rect_rtrees;
  bgi::rtree<std::pair<GTLRectInt, int32_t>, bgi::quadratic<16>> boundary_rtrees;
  bgi::rtree<CutData, bgi::quadratic<16>, CutDataIndexable> cut_rtrees;

  const CutData& getCut(int32_t cut_id) const { return cut_pool[cut_id]; }
  int32_t getCutId(const CutData& cut_data) const { return static_cast<int32_t>(&cut_data - cut_pool.data()); }
  const PolygonData& getPolygon(int32_t polygon_id) const { return polygon_pool[polygon_id]; }
  int32_t getPolygonId(const PolygonData& polygon_data) const { return static_cast<int32_t>(&polygon_data - polygon_pool.data()); }
  const MaxRectData& getMaxRect(int32_t max_rect_id) const { return max_rect_pool[max_rect_id]; }
  const BoundaryData& getBoundary(int32_t boundary_id) const { return boundary_pool[boundary_id]; }
  int32_t getBoundaryId(const BoundaryData& boundary_data) const { return static_cast<int32_t>(&boundary_data - boundary_pool.data()); }
  const BoundaryData& getPrevBoundary(int32_t boundary_id) const { return getBoundary(getBoundary(boundary_id).prev_boundary_id); }
  const BoundaryData& getNextBoundary(int32_t boundary_id) const { return getBoundary(getBoundary(boundary_id).next_boundary_id); }

  int32_t getNetIdxByCutId(int32_t cut_id) const { return getCut(cut_id).net_idx; }
  int32_t getNetIdxByPolygonId(int32_t polygon_id) const { return getPolygon(polygon_id).net_id; }
  int32_t getNetIdxByMaxRectId(int32_t max_rect_id) const { return getNetIdxByPolygonId(getMaxRect(max_rect_id).polygon_id); }
  int32_t getNetIdxByBoundaryId(int32_t boundary_id) const { return getNetIdxByPolygonId(getBoundary(boundary_id).polygon_id); }

  std::span<const CutData> getCuts() const { return std::span<const CutData>(cut_pool.data(), cut_pool.size()); }

  std::span<const PolygonData> getPolygons(const RVRoutingNet& routing_net) const
  {
    return makeSpan(polygon_pool, routing_net.polygon_begin, routing_net.polygon_count);
  }

  std::span<const MaxRectData> getMaxRects(const RVRoutingNet& routing_net) const
  {
    return makeSpan(max_rect_pool, routing_net.max_rect_begin, routing_net.max_rect_count);
  }

  std::span<const BoundaryData> getBoundaries(const RVRoutingNet& routing_net) const
  {
    return makeSpan(boundary_pool, routing_net.boundary_begin, routing_net.boundary_count);
  }

  std::span<const MaxRectData> getMaxRects(const PolygonData& polygon_data) const
  {
    return makeSpan(max_rect_pool, polygon_data.max_rect_begin, polygon_data.max_rect_count);
  }

  std::span<const BoundaryData> getBoundaries(const PolygonData& polygon_data) const
  {
    return makeSpan(boundary_pool, polygon_data.boundary_begin, polygon_data.boundary_count);
  }

  int32_t getOuterBoundaryCount(const PolygonData& polygon_data) const
  {
    int32_t outer_boundary_count = 0;
    for (const BoundaryData& boundary_data : getBoundaries(polygon_data)) {
      if (!boundary_data.isHole) {
        outer_boundary_count++;
      }
    }
    return outer_boundary_count;
  }

  template <typename OutputIt>
  void queryMaxRects(const GTLRectInt& query_rect, OutputIt out) const
  {
    rect_rtrees.query(bgi::intersects(query_rect), out);
  }

  template <typename OutputIt>
  void queryBoundaries(const GTLRectInt& query_rect, OutputIt out) const
  {
    boundary_rtrees.query(bgi::intersects(query_rect), out);
  }

  template <typename OutputIt>
  void queryCuts(const GTLRectInt& query_rect, OutputIt out) const
  {
    cut_rtrees.query(bgi::intersects(query_rect), out);
  }

 private:
  template <typename T>
  static std::span<const T> makeSpan(const std::vector<T>& data, int32_t begin, int32_t count)
  {
    if (count <= 0) {
      return {};
    }
    return std::span<const T>(data.data() + begin, static_cast<size_t>(count));
  }
};

}  // namespace idrc
