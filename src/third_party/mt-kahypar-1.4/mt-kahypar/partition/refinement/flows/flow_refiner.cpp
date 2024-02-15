/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2021 Tobias Heuer <tobias.heuer@kit.edu>
 * Copyright (C) 2021 Lars Gottesbüren <lars.gottesbueren@kit.edu>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#include "mt-kahypar/partition/refinement/flows/flow_refiner.h"
#include "mt-kahypar/partition/refinement/gains/gain_definitions.h"
#include "mt-kahypar/utils/utilities.h"

#include "tbb/concurrent_queue.h"

#include "mt-kahypar/definitions.h"

namespace mt_kahypar {

template<typename GraphAndGainTypes>
MoveSequence FlowRefiner<GraphAndGainTypes>::refineImpl(mt_kahypar_partitioned_hypergraph_const_t& hypergraph,
                                                     const Subhypergraph& sub_hg,
                                                     const HighResClockTimepoint& start) {
  const PartitionedHypergraph& phg = utils::cast_const<PartitionedHypergraph>(hypergraph);
  MoveSequence sequence { { }, 0 };
  utils::Timer& timer = utils::Utilities::instance().getTimer(_context.utility_id);
  // Construct flow network that contains all vertices given in refinement nodes
  timer.start_timer("construct_flow_network", "Construct Flow Network", true);
  FlowProblem flow_problem = constructFlowHypergraph(phg, sub_hg);
  timer.stop_timer("construct_flow_network");
  if ( flow_problem.total_cut - flow_problem.non_removable_cut > 0 ) {

    // Solve max-flow min-cut problem
    bool time_limit_reached = false;
    timer.start_timer("hyper_flow_cutter", "HyperFlowCutter", true);
    bool flowcutter_succeeded = runFlowCutter(flow_problem, start, time_limit_reached);
    timer.stop_timer("hyper_flow_cutter");
    if ( flowcutter_succeeded ) {
      // We apply the solution if it either improves the cut or the balance of
      // the bipartition induced by the two blocks

      HyperedgeWeight new_cut = flow_problem.non_removable_cut;
      HypernodeWeight max_part_weight;
      const bool sequential = _context.shared_memory.num_threads == _context.refinement.flows.num_parallel_searches;
      if (sequential) {
        new_cut += _sequential_hfc.cs.flow_algo.flow_value;
        max_part_weight = std::max(_sequential_hfc.cs.source_weight, _sequential_hfc.cs.target_weight);
      } else {
        new_cut += _parallel_hfc.cs.flow_algo.flow_value;
        max_part_weight = std::max(_parallel_hfc.cs.source_weight, _parallel_hfc.cs.target_weight);
      }

      const bool improved_solution = new_cut < flow_problem.total_cut ||
        (new_cut == flow_problem.total_cut && max_part_weight < std::max(flow_problem.weight_of_block_0, flow_problem.weight_of_block_1));

      // Extract move sequence
      if ( improved_solution ) {
        sequence.expected_improvement = flow_problem.total_cut - new_cut;
        for ( const whfc::Node& u : _flow_hg.nodeIDs() ) {
          const HypernodeID hn = _whfc_to_node[u];
          if ( hn != kInvalidHypernode ) {
            const PartitionID from = phg.partID(hn);
            PartitionID to;
            if (sequential) {
              to = _sequential_hfc.cs.flow_algo.isSource(u) ? _block_0 : _block_1;
            } else {
              to = _parallel_hfc.cs.flow_algo.isSource(u) ? _block_0 : _block_1;
            }

            if ( from != to ) {
              sequence.moves.push_back(Move { from, to, hn, kInvalidGain });
            }
          }
        }
      }
    } else if ( time_limit_reached ) {
      sequence.state = MoveSequenceState::TIME_LIMIT;
    }
  }
  return sequence;
}

#define NOW std::chrono::high_resolution_clock::now()
#define RUNNING_TIME(X) std::chrono::duration<double>(NOW - X).count();

template<typename GraphAndGainTypes>
bool FlowRefiner<GraphAndGainTypes>::runFlowCutter(const FlowProblem& flow_problem,
                                                const HighResClockTimepoint& start,
                                                bool& time_limit_reached) {
  whfc::Node s = flow_problem.source;
  whfc::Node t = flow_problem.sink;
  bool result = false;

  size_t iteration = 0;
  auto on_cut = [&] {
    if (++iteration == 25) {
      iteration = 0;
      double elapsed = RUNNING_TIME(start);
      if (elapsed > _time_limit) {
        time_limit_reached = true;
        return false;
      }
    }
    return true;
  };


  const bool sequential = _context.shared_memory.num_threads == _context.refinement.flows.num_parallel_searches;
  if (sequential) {
    _sequential_hfc.cs.setMaxBlockWeight(0, std::max(
            flow_problem.weight_of_block_0, _context.partition.max_part_weights[_block_0]));
    _sequential_hfc.cs.setMaxBlockWeight(1, std::max(
            flow_problem.weight_of_block_1, _context.partition.max_part_weights[_block_1]));

    _sequential_hfc.reset();
    _sequential_hfc.setFlowBound(flow_problem.total_cut - flow_problem.non_removable_cut);
    result = _sequential_hfc.enumerateCutsUntilBalancedOrFlowBoundExceeded(s, t, on_cut);
  } else {
    _parallel_hfc.cs.setMaxBlockWeight(0, std::max(
            flow_problem.weight_of_block_0, _context.partition.max_part_weights[_block_0]));
    _parallel_hfc.cs.setMaxBlockWeight(1, std::max(
            flow_problem.weight_of_block_1, _context.partition.max_part_weights[_block_1]));

    _parallel_hfc.reset();
    _parallel_hfc.setFlowBound(flow_problem.total_cut - flow_problem.non_removable_cut);
    result = _parallel_hfc.enumerateCutsUntilBalancedOrFlowBoundExceeded(s, t, on_cut);
  }
  return result;
}

template<typename GraphAndGainTypes>
FlowProblem FlowRefiner<GraphAndGainTypes>::constructFlowHypergraph(const PartitionedHypergraph& phg,
                                                                 const Subhypergraph& sub_hg) {
  _block_0 = sub_hg.block_0;
  _block_1 = sub_hg.block_1;
  ASSERT(_block_0 != kInvalidPartition && _block_1 != kInvalidPartition);
  FlowProblem flow_problem;


  const bool sequential = _context.shared_memory.num_threads == _context.refinement.flows.num_parallel_searches;
  if ( sequential ) {
    flow_problem = _sequential_construction.constructFlowHypergraph(
      phg, sub_hg, _block_0, _block_1, _whfc_to_node);
  } else {
    flow_problem = _parallel_construction.constructFlowHypergraph(
      phg, sub_hg, _block_0, _block_1, _whfc_to_node);
  }

  DBG << "Flow Hypergraph [ Nodes =" << _flow_hg.numNodes()
      << ", Edges =" << _flow_hg.numHyperedges()
      << ", Pins =" << _flow_hg.numPins()
      << ", Blocks = (" << _block_0 << "," << _block_1 << ") ]";

  return flow_problem;
}

namespace {
#define FLOW_REFINER(X) FlowRefiner<X>
}

INSTANTIATE_CLASS_WITH_VALID_TRAITS(FLOW_REFINER)

} // namespace mt_kahypar
