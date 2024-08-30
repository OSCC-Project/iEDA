/**
 * @file timing_eval.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-28
 * @brief evaluation with timing & power
 */
#include <memory>
#include <vector>

#include "init_sta.hh"
#include "timing_db.hh"
namespace ieval {

class TimingEval
{
 public:
  TimingEval(const std::string& routing_type) : _routing_type(routing_type)
  {
    RoutingType type = _routing_type == "WLM"     ? RoutingType::kWLM
                       : _routing_type == "HPWL"  ? RoutingType::kHPWL
                       : _routing_type == "FLUTE" ? RoutingType::kFLUTE
                       : _routing_type == "EGR"   ? RoutingType::kEGR
                       : _routing_type == "DR"    ? RoutingType::kDR
                                                  : RoutingType::kNone;
    _init_sta = std::make_unique<InitSTA>(type);
    _init_sta->runSTA();
  }
  ~TimingEval() = default;

  TimingSummary evalDesign();

  double evalNetPower(const std::string& net_name) const;
  std::map<std::string, double> evalAllNetPower() const;

 private:
  std::string _routing_type = "WLM";  // RoutingType::kWLM, kHPWL, kFLUTE, kEGR, kDR
  std::unique_ptr<InitSTA> _init_sta;
};

}  // namespace ieval