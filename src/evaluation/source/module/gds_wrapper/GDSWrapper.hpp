#ifndef SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRAPPER_GDSWRAPPER_HPP_
#define SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRAPPER_GDSWRAPPER_HPP_

#include <vector>

#include "GDSNet.hpp"

namespace eval {

class GDSWrapper
{
 public:
  GDSWrapper() = default;
  ~GDSWrapper() = default;

  void set_net_list(const std::vector<GDSNet*>& net_list) { _gds_net_list = net_list; }
  std::vector<GDSNet*>& get_net_list() { return _gds_net_list; }

 private:
  std::vector<GDSNet*> _gds_net_list;
};
}  // namespace eval

#endif  // SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRAPPER_GDSWRAPPER_HPP_
