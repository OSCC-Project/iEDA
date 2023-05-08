#ifndef SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRITER_DATABASE_GDSPIN_HPP_
#define SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRITER_DATABASE_GDSPIN_HPP_

#include <string>

#include "EvalPoint.hpp"
#include "GDSPort.hpp"

namespace eval {

class GDSPin
{
 public:
  GDSPin() = default;
  ~GDSPin() = default;

  int32_t get_idx() const { return _idx; }
  std::string get_name() const { return _name; }
  std::vector<GDSPort*> get_port_list() const { return _port_list; }
  Point<int32_t> get_coord() const { return _coord; }

  void set_idx(const int32_t& idx) { _idx = idx; }
  void set_name(const std::string& name) { _name = name; }
  void set_coord(const Point<int32_t>& coord) { _coord = coord; }

  void add_port(GDSPort* gds_port) { _port_list.push_back(gds_port); }

 private:
  int32_t _idx = -1;
  std::string _name;
  std::vector<GDSPort*> _port_list;
  Point<int32_t> _coord;
};

}  // namespace eval

#endif  // SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRITER_DATABASE_GDSPIN_HPP_
