/*
 * @Author: S.J Chen
 * @Date: 2022-04-19 14:15:41
 * @LastEditTime: 2022-04-23 15:53:31
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/evaluator/timing/TimingEvaluation.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_EVALUATOR_TIMING_H
#define IPL_EVALUATOR_TIMING_H

#include "TimingEval.hpp"
#include "config/Config.hh"
#include "config/TimingConfig.hh"
#include "evaluator/wirelength/SteinerWirelength.hh"
#include "util/topology_manager/TopologyManager.hh"

namespace ipl {

class TimingEvaluation
{
 public:
  TimingEvaluation() = delete;
  TimingEvaluation(eval::TimingEval* timing_evaluator, TopologyManager* topo_manager);
  TimingEvaluation(const TimingEvaluation&) = delete;
  TimingEvaluation(TimingEvaluation&&) = delete;
  ~TimingEvaluation();

  TimingEvaluation& operator=(const TimingEvaluation&) = delete;
  TimingEvaluation& operator=(TimingEvaluation&&) = delete;

  void updateEvalTiming();

 private:
  eval::TimingEval* _timing_evaluator;
  TopologyManager* _topo_manager;
  SteinerWirelength* _steiner_wirelength;

  eval::TimingPin* wrapTimingTruePin(Node* node);
  eval::TimingPin* wrapTimingFakePin(int id, Point<int32_t> coordi);
};
inline TimingEvaluation::TimingEvaluation(eval::TimingEval* timing_evaluator, TopologyManager* topo_manager)
{
  _timing_evaluator = timing_evaluator;
  _topo_manager = topo_manager;
  _steiner_wirelength = new SteinerWirelength(_topo_manager);
}
inline TimingEvaluation::~TimingEvaluation()
{
  delete _steiner_wirelength;
}

}  // namespace ipl

#endif
