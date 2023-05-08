#include "ipdn_basic.h"

#include "IdbSpecialNet.h"
#include "IdbSpecialWire.h"

namespace ipdn {

RouteInfo::RouteInfo()
{
  idb::IdbSpecialWire* power = new idb::IdbSpecialWire();
  idb::IdbSpecialWire* ground = new idb::IdbSpecialWire();
  _power_route.insert(std::make_pair("VDD", power));
  _power_route.insert(std::make_pair("VSS", ground));
};

void RouteInfo::add_special_wire_segment(std::string special_net_name, idb::IdbSpecialWireSegment* special_wire_segment)
{
  _power_route[special_net_name]->add_segment(special_wire_segment);
}

}  // namespace ipdn
