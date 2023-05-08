#pragma once

#include <string>

#include "CtsEnum.h"
#include "Traits.h"
#include "pgl.h"

namespace icts {
using std::string;
using std::vector;

struct Endpoint {
 public:
  Endpoint() {}
  Endpoint(const string& name, const Point& point)
      : _name(name), _point(point) {}
  Endpoint(const string& name, const Point& point, const Concept& cpt)
      : _name(name), _point(point), _cpt(cpt) {}
  string _name;
  Point _point;
  Concept _cpt;
  int _sub_wirelength = 0;
  bool _detour = false;
  vector<Point> _detour_points;
  int _fanout = 0;
};

template <>
struct DataTraits<Endpoint> {
  typedef typename Point::coord_t coordinate_type;
  typedef std::string id_type;

  static inline id_type getId(const Endpoint& t) { return t._name; }
  static inline coordinate_type getX(const Endpoint& t) { return t._point.x(); }
  static inline coordinate_type getY(const Endpoint& t) { return t._point.y(); }
  static inline Point getPoint(const Endpoint& t) { return t._point; }
  static inline bool needDetour(const Endpoint& t) { return t._detour; }
  static inline vector<Point> getDetourPoints(const Endpoint& t) {
    return t._detour_points;
  }
  static inline int getSubWirelength(const Endpoint& t) {
    return t._sub_wirelength;
  }
  static inline int getFanout(const Endpoint& t) { return t._fanout; }

  static inline void setId(Endpoint& t, const id_type& id) { t._name = id; }
  static inline void setX(Endpoint& t, const coordinate_type& x) {
    t._point.x(x);
  }
  static inline void setY(Endpoint& t, const coordinate_type& y) {
    t._point.y(y);
  }
  static inline void setPoint(Endpoint& t, const Point& p) { t._point = p; }

  static inline void setDetour(Endpoint& t) { t._detour = true; }
  static inline void setDetourPoints(Endpoint& t, const vector<Point>& p) {
    t._detour_points = p;
  }
  static inline void setSubWirelength(Endpoint& t, const int& sub_wirelength) {
    t._sub_wirelength = sub_wirelength;
  }
  static inline void setFanout(Endpoint& t, const int& fanout) {
    t._fanout = fanout;
  }
};

class CtsSignalWire {
 public:
  CtsSignalWire() = default;
  CtsSignalWire(const Endpoint& first, const Endpoint& second) {
    _wire.first = first;
    _wire.second = second;
  }
  CtsSignalWire(const CtsSignalWire&) = default;
  ~CtsSignalWire() = default;

  Endpoint get_first() const { return _wire.first; }
  Endpoint get_second() const { return _wire.second; }

  void set_first(const Endpoint& end_point) { _wire.first = end_point; }
  void set_second(const Endpoint& end_point) { _wire.second = end_point; }

  int getWireLength() const {
    return static_cast<int>(
        gtl::manhattan_distance(_wire.first._point, _wire.second._point));
  }

 private:
  std::pair<Endpoint, Endpoint> _wire;
};

}  // namespace icts