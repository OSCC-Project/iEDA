#include "congestion_api.h"

#include "congestion_eval.h"

namespace ieval {

CongestionAPI::CongestionAPI()
{
}

CongestionAPI::~CongestionAPI()
{
}

OverflowSummary CongestionAPI::getOverflowSummary()
{
  OverflowSummary overflow_summary;

  CongestionEval congestion_eval;
  congestion_eval.runEGR();
  congestion_eval.computeOverflow();

  overflow_summary.total_overflow = congestion_eval.get_total_overflow();
  overflow_summary.max_overflow = congestion_eval.get_max_overflow();
  overflow_summary.average_overflow = congestion_eval.get_average_overflow();

  return overflow_summary;
}

MapPathSummary CongestionAPI::getMapPathSummary()
{
  MapPathSummary map_path_summary;

  CongestionEval congestion_eval;
  congestion_eval.runEGR();
  congestion_eval.runRUDY();

  map_path_summary.egr_path = congestion_eval.plotEGR();
  map_path_summary.rudy_path = congestion_eval.plotRUDY();

  return map_path_summary;
}

}  // namespace ieval
