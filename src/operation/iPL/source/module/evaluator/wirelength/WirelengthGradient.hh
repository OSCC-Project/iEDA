/*
 * @Author: S.J Chen
 * @Date: 2022-03-08 22:36:24
 * @LastEditTime: 2022-11-23 11:57:32
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/evaluator/wirelength/WirelengthGradient.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_EVALUATOR_WIRELENGTH_GRADIENT_H
#define IPL_EVALUATOR_WIRELENGTH_GRADIENT_H

#include "TopologyManager.hh"
#include "data/Point.hh"

namespace ipl {

class WirelengthGradient
{
 public:
  WirelengthGradient() = delete;
  explicit WirelengthGradient(TopologyManager* topology_manager);
  WirelengthGradient(const WirelengthGradient&) = delete;
  WirelengthGradient(WirelengthGradient&&) = delete;
  virtual ~WirelengthGradient() = default;

  WirelengthGradient& operator=(const WirelengthGradient&) = delete;
  WirelengthGradient& operator=(WirelengthGradient&&) = delete;

  virtual void updateWirelengthForce(float coeff_x, float coeff_y, float min_force_bar, int32_t thread_num) = 0;
  virtual Point<float> obtainWirelengthGradient(std::string inst_name, float coeff_x, float coeff_y) = 0;

  // Debug
  virtual void waWLAnalyzeForDebug(float coeff_x, float coeff_y) = 0;

 protected:
  TopologyManager* _topology_manager;
};
inline WirelengthGradient::WirelengthGradient(TopologyManager* topology_manager) : _topology_manager(topology_manager)
{
}

}  // namespace ipl

#endif