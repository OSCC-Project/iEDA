#pragma once

#include <map>
#include <string>
#include <vector>

namespace idb {
class IdbSpecialWire;
class IdbSpecialWireSegment;
}  // namespace idb

namespace ipdn {

class RouteInfo
{
 public:
  RouteInfo();
  ~RouteInfo(){};

  int32_t get_route_width() const { return _route_width; }
  int32_t get_route_pitch() const { return _route_pitch; }
  int32_t get_route_offset() const { return _route_offset; }

  std::map<std::string, idb::IdbSpecialWire*> get_power_route() { return _power_route; }

  void set_route_width(int32_t width) { _route_width = width; }
  void set_route_pitch(int32_t pitch) { _route_pitch = pitch; }
  void set_route_offset(int32_t offset) { _route_offset = offset; }
  void set_width_pitch_offset(int32_t width, int32_t pitch, int32_t offset)
  {
    set_route_width(width);
    set_route_pitch(pitch);
    set_route_offset(offset);
  }

  void add_special_wire_segment(std::string special_net_name, idb::IdbSpecialWireSegment* special_wire_segment);

 private:
  int32_t _route_width;
  int32_t _route_pitch;
  int32_t _route_offset;
  std::map<std::string, idb::IdbSpecialWire*> _power_route;
};

}  // namespace ipdn
