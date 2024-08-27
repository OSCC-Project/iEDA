#include "congestion_eval.h"

#include "init_egr.h"

namespace ieval {

CongestionEval::CongestionEval()
{
}

CongestionEval::~CongestionEval()
{
}

void CongestionEval::runEGR()
{
  InitEGR init_EGR;
  init_EGR.runEGR();
}

void CongestionEval::runRUDY()
{
}

void CongestionEval::runNCTUgr()
{
}

void CongestionEval::computeOverflow()
{
}

string CongestionEval::plotEGR()
{
  return "EGR path";
}

string CongestionEval::plotRUDY()
{
  return "RUDY path";
}

}  // namespace ieval