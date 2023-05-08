/**
 * @file StaApplySdc.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief Apply the sdc constrain to the graph.
 * @version 0.1
 * @date 2021-03-01
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

#include <memory>
#include <vector>

#include "StaFunc.hh"

namespace ista {

/**
 * @brief The functor of apply sdc.
 *
 */
class StaApplySdc : public StaFunc {
 public:
  enum class PropType {
    kApplySdcPreProp,
    kApplySdcPostClockProp,
    kApplySdcPostProp
  };

  explicit StaApplySdc(PropType prop_type) : _prop_type(prop_type) {}
  ~StaApplySdc() override = default;

  unsigned operator()(StaGraph* the_graph) override;

 private:
  // apply sdc pre-propagation.
  unsigned setupClocks(StrMap<std::unique_ptr<SdcClock>>& sdc_clocks,
                       StaGraph* the_graph);
  unsigned setupInputTransition(
      const std::unique_ptr<SdcIOConstrain>& io_constraint,
      StaGraph* the_graph);

  unsigned setupOutputLoad(const std::unique_ptr<SdcIOConstrain>& io_constraint,
                           StaGraph* the_graph);

  unsigned setupIODelay(const std::unique_ptr<SdcIOConstrain>& io_constraint,
                        StaGraph* the_graph);

  unsigned setupIOConstrain(
      std::vector<std::unique_ptr<SdcIOConstrain>>& sdc_io_constraints,
      StaGraph* the_graph);

  unsigned setupTimingDrc(
      std::vector<std::unique_ptr<SdcTimingDRC>>& sdc_timing_drcs,
      StaGraph* the_graph);

  unsigned setupOcvDerate(
      std::vector<std::unique_ptr<SdcTimingDerate>>& sdc_timing_derates,
      StaGraph* the_graph);

  std::vector<std::string> getExceptionObjs(std::vector<std::string>& obj_vec,
                                            StaGraph* the_graph,
                                            AnalysisMode analysis_mode,
                                            TransType trans_type, bool is_from);
  unsigned setupException(
      std::vector<std::unique_ptr<SdcException>>& sdc_exceptions,
      StaGraph* the_graph);

  // apply sdc post-propagation after propagation.
  unsigned processClockUncertainty(
      std::vector<std::unique_ptr<SdcSetClockUncertainty>>&
          sdc_clock_uncertaintys,
      StaGraph* the_graph);

  PropType _prop_type;
};

}  // namespace ista
