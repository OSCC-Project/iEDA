#ifndef SRC_EVALUATION_SOURCE_MODULE_WIRELENGTH_WL_HPP_
#define SRC_EVALUATION_SOURCE_MODULE_WIRELENGTH_WL_HPP_

#include "WLNet.hpp"

namespace eval {
class WL
{
 public:
  virtual ~WL() {}
  virtual int64_t getTotalWL(const std::vector<WLNet*>& net_list) = 0;
};

class WLMWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class HPWLWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class HTreeWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class VTreeWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class CliqueWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class StarWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class B2BWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class FluteWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class PlaneRouteWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class SpaceRouteWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class DRWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};
}  // namespace eval

#endif // SRC_EVALUATION_SOURCE_MODULE_WIRELENGTH_WL_HPP_
