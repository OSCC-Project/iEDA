/**
 * @file StaArc.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of timing arc.
 * @version 0.1
 * @date 2021-02-10
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "StaArc.hh"

#include "StaDump.hh"
#include "StaFunc.hh"
#include "StaVertex.hh"

namespace ista {

StaArc::StaArc(StaVertex* src, StaVertex* snk) : _src(src), _snk(snk) {}

/**
 * @brief Add arc delay data.
 *
 * @param arc_delay_data
 */
void StaArc::addData(StaArcDelayData* arc_delay_data) {
  if (arc_delay_data) {
    _arc_delay_bucket.addData(arc_delay_data, 0);
  }
}

/**
 * @brief Get arc delay.
 *
 * @param analysis_mode
 * @param trans_type
 * @return int
 */
int StaArc::get_arc_delay(AnalysisMode analysis_mode, TransType trans_type) {
  StaData* data;
  FOREACH_ARC_DELAY_DATA(this, data) {
    if (data->get_delay_type() == analysis_mode &&
        data->get_trans_type() == trans_type) {
      auto* arc_delay = dynamic_cast<StaArcDelayData*>(data);
      return arc_delay->get_arc_delay();
    }
  }

  return 0;
}

/**
 * @brief Get arc delay data.
 *
 * @param analysis_mode
 * @param trans_type
 * @return StaArcDelayData*
 */
StaArcDelayData* StaArc::getArcDelayData(AnalysisMode analysis_mode,
                                         TransType trans_type) {
  StaData* data;
  FOREACH_ARC_DELAY_DATA(this, data) {
    if (data->get_delay_type() == analysis_mode &&
        data->get_trans_type() == trans_type) {
      auto* arc_delay = dynamic_cast<StaArcDelayData*>(data);
      return arc_delay;
    }
  }

  return nullptr;
}

unsigned StaArc::exec(StaFunc& func) { return func(this); }

/**
 * @brief dump arc info for debug.
 *
 */
void StaArc::dump() {
  StaDumpYaml dump_data;
  dump_data(this);
  dump_data.printText("arc.txt");
}

StaNetArc::StaNetArc(StaVertex* driver, StaVertex* load, Net* net)
    : StaArc(driver, load), _net(net) {}

StaInstArc::StaInstArc(StaVertex* src, StaVertex* snk, LibertyArc* lib_arc,
                       Instance* inst)
    : StaArc(src, snk), _lib_arc(lib_arc), _inst(inst) {}

}  // namespace ista
