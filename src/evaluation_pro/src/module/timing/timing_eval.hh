/**
 * @file timing_eval.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-28
 * @brief evaluation with timing & power
 */
#include <memory>
#include <vector>

#include "timing_db.hh"
namespace ieval {

class TimingEval
{
 public:
  TimingEval();
  ~TimingEval() = default;
  static void initRoutingType(const std::string& routing_type);
  static TimingEval* getInst();

  static void destroyInst();

  TimingSummary evalDesign();

  double evalNetPower(const std::string& net_name) const;
  std::map<std::string, double> evalAllNetPower() const;

 private:
  static TimingEval* _timing_eval;
  std::string _routing_type = "WLM";  // RoutingType::kWLM, kHPWL, kFLUTE, kEGR, kDR
};

}  // namespace ieval