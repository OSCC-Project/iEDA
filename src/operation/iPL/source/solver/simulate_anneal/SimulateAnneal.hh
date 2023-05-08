
#pragma once

#include "Evaluation.hh"
#include "SAParam.hh"
#include "Solution.hh"

namespace ipl {

class SimulateAnneal
{
 public:
  SimulateAnneal(SAParam* param, Evaluation* evaluation)
  {
    _param = param;
    _evaluation = evaluation;
    _solution = _evaluation->get_solution();
  }
  ~SimulateAnneal(){};
  void runAnneal();

 private:
  SAParam* _param;
  Evaluation* _evaluation;
  Solution* _solution;
};

}  // namespace ipl