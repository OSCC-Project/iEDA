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

#if CUDA_PROPAGATION
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

  unsigned prepareGPUData(StaGraph* the_graph);

 private:
  std::map<unsigned, std::vector<StaArc*>> _level_to_arcs;
  std::map<unsigned, std::vector<unsigned>> _level_to_arc_index;
  std::map<unsigned, StaArc*> _index_to_arc; //!< gpu index to sta arc.
};

}  // namespace ista

#endif