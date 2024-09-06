/**
 * @file timing_eval.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-08-28
 * @brief evaluation with timing & power
 */
#include <memory>
#include <unordered_map>
#include <vector>

#include "timing_db.hh"
namespace ieval {

class TimingEval
{
 public:
  TimingEval();
  ~TimingEval() = default;
  static TimingEval* getInst();

  static void destroyInst();

  std::map<std::string, TimingSummary> evalDesign();

  std::map<std::string, std::unordered_map<std::string, double>> evalNetPower() const;

 private:
  static TimingEval* _timing_eval;
};

}  // namespace ieval