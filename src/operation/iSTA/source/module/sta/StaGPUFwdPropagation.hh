/**
 * @file StaGPUFwdPropagation.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The host api for gpu propagation.
 * @version 0.1
 * @date 2025-02-02
 *
 */
#pragma once
#include "StaFunc.hh"

namespace ista {

/**
 * @brief The class for gpu fwd propagation wrapper.
 *
 */
class StaGPUFwdPropagation : public StaFunc {
 public:
  StaGPUFwdPropagation(std::map<unsigned, std::vector<StaArc*>>&& level_to_arcs)
      : _level_to_arcs(std::move(level_to_arcs)) {}
  ~StaGPUFwdPropagation() override = default;

  unsigned operator()(StaGraph* the_graph) override;

 private:
  std::map<unsigned, std::vector<StaArc*>> _level_to_arcs;
};

}  // namespace ista