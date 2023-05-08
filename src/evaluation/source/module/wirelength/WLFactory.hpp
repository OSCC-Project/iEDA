#ifndef SRC_EVALUATOR_SOURCE_WIRELENGTH_DATABASE_WLFACTORY_HPP_
#define SRC_EVALUATOR_SOURCE_WIRELENGTH_DATABASE_WLFACTORY_HPP_

#include "EvalMagicEnum.hpp"
#include "WL.hpp"

namespace eval {

enum WL_TYPE
{
  kWLM,
  kHPWL,
  kHTree,
  kVTree,
  kClique,
  kStar,
  kB2B,
  kFlute,
  kPlaneRoute,
  kSpaceRoute,
  kDR
};

class WLFactory
{
 public:
  WL* createWL(const std::string& wl_type)
  {
    auto enum_type = magic_enum::enum_cast<WL_TYPE>(wl_type);
    switch (enum_type.value()) {
      case kWLM:
        return new WLMWL();
        break;
      case kHPWL:
        return new HPWLWL();
        break;
      case kHTree:
        return new HTreeWL();
        break;
      case kVTree:
        return new VTreeWL();
        break;
      case kClique:
        return new CliqueWL();
        break;
      case kStar:
        return new StarWL();
        break;
      case kB2B:
        return new B2BWL();
        break;
      case kFlute:
        return new FluteWL();
        break;
      case kPlaneRoute:
        return new PlaneRouteWL();
        break;
      case kSpaceRoute:
        return new SpaceRouteWL();
        break;
      case kDR:
        return new DRWL();
        break;
      default:
        return nullptr;
        break;
    }
  }
};
}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_WIRELENGTH_DATABASE_WLFACTORY_HPP_
