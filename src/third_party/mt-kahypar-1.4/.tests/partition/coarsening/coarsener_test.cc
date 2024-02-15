/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2019 Tobias Heuer <tobias.heuer@kit.edu>
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

#include "gmock/gmock.h"

#include "tests/definitions.h"
#include "tests/partition/coarsening/coarsener_fixtures.h"
#include "mt-kahypar/partition/coarsening/multilevel_coarsener.h"
#include "mt-kahypar/partition/coarsening/multilevel_uncoarsener.h"
#ifdef KAHYPAR_ENABLE_HIGHEST_QUALITY_FEATURES
#include "mt-kahypar/partition/coarsening/nlevel_coarsener.h"
#include "mt-kahypar/partition/coarsening/nlevel_uncoarsener.h"
#endif


using ::testing::Test;

namespace mt_kahypar {

using AMultilevelCoarsener = ACoarsener<StaticHypergraphTypeTraits,
                                        MultilevelCoarsener,
                                        MultilevelUncoarsener,
                                        PresetType::default_preset>;

TEST_F(AMultilevelCoarsener, DecreasesNumberOfPins) {
  context.coarsening.contraction_limit = 4;
  decreasesNumberOfPins(6 /* expected number of pins */ );
}

TEST_F(AMultilevelCoarsener, DecreasesNumberOfHyperedges) {
  context.coarsening.contraction_limit = 4;
  decreasesNumberOfHyperedges(3 /* expected number of hyperedges */ );
}

TEST_F(AMultilevelCoarsener, RemovesHyperedgesOfSizeOneDuringCoarsening) {
  using Hypergraph = typename StaticHypergraphTypeTraits::Hypergraph;
  context.coarsening.contraction_limit = 4;
  doCoarsening();
  auto& hypergraph = utils::cast<Hypergraph>(coarsener->coarsestHypergraph());
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_GE(hypergraph.edgeSize(he), 2);
  }
}

TEST_F(AMultilevelCoarsener, RemovesParallelHyperedgesDuringCoarsening) {
  using Hypergraph = typename StaticHypergraphTypeTraits::Hypergraph;
  context.coarsening.contraction_limit = 4;
  doCoarsening();
  auto& hypergraph = utils::cast<Hypergraph>(coarsener->coarsestHypergraph());
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_EQ(hypergraph.edgeWeight(he), 2);
  }
}

TEST_F(AMultilevelCoarsener, ProjectsPartitionBackToOriginalHypergraph) {
  using PartitionedHypergraph = typename StaticHypergraphTypeTraits::PartitionedHypergraph;
  context.coarsening.contraction_limit = 4;
  context.refinement.label_propagation.algorithm = LabelPropagationAlgorithm::do_nothing;
  context.refinement.fm.algorithm = FMAlgorithm::do_nothing;
  context.refinement.flows.algorithm = FlowAlgorithm::do_nothing;
  context.type = ContextType::initial_partitioning;
  doCoarsening();
  PartitionedHypergraph& coarsest_partitioned_hypergraph =
    utils::cast<PartitionedHypergraph>(coarsener->coarsestPartitionedHypergraph());
  assignPartitionIDs(coarsest_partitioned_hypergraph);
  PartitionedHypergraph partitioned_hypergraph = uncoarsener->uncoarsen();
  for ( const HypernodeID& hn : partitioned_hypergraph.nodes() ) {
    PartitionID part_id = 0;
    ASSERT_EQ(part_id, partitioned_hypergraph.partID(hn));
  }
}

#ifdef KAHYPAR_ENABLE_HIGHEST_QUALITY_FEATURES
using ANLevelCoarsener = ACoarsener<DynamicHypergraphTypeTraits,
                                    NLevelCoarsener,
                                    NLevelUncoarsener,
                                    PresetType::highest_quality>;

TEST_F(ANLevelCoarsener, DecreasesNumberOfPins) {
  context.coarsening.contraction_limit = 4;
  decreasesNumberOfPins(6 /* expected number of pins */ );
}

TEST_F(ANLevelCoarsener, DecreasesNumberOfHyperedges) {
  context.coarsening.contraction_limit = 4;
  decreasesNumberOfHyperedges(3 /* expected number of hyperedges */ );
}

TEST_F(ANLevelCoarsener, RemovesHyperedgesOfSizeOneDuringCoarsening) {
  using Hypergraph = typename DynamicHypergraphTypeTraits::Hypergraph;
  context.coarsening.contraction_limit = 4;
  doCoarsening();
  auto& hypergraph = utils::cast<Hypergraph>(coarsener->coarsestHypergraph());
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_GE(hypergraph.edgeSize(he), 2);
  }
}

TEST_F(ANLevelCoarsener, RemovesParallelHyperedgesDuringCoarsening) {
  using Hypergraph = typename DynamicHypergraphTypeTraits::Hypergraph;
  context.coarsening.contraction_limit = 4;
  doCoarsening();
  auto& hypergraph = utils::cast<Hypergraph>(coarsener->coarsestHypergraph());
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_EQ(hypergraph.edgeWeight(he), 2);
  }
}

TEST_F(ANLevelCoarsener, ProjectsPartitionBackToOriginalHypergraph) {
  using PartitionedHypergraph = typename DynamicHypergraphTypeTraits::PartitionedHypergraph;
  context.coarsening.contraction_limit = 4;
  context.refinement.label_propagation.algorithm = LabelPropagationAlgorithm::do_nothing;
  context.refinement.fm.algorithm = FMAlgorithm::do_nothing;
  context.refinement.flows.algorithm = FlowAlgorithm::do_nothing;
  context.type = ContextType::initial_partitioning;
  doCoarsening();
  PartitionedHypergraph& coarsest_partitioned_hypergraph =
    utils::cast<PartitionedHypergraph>(coarsener->coarsestPartitionedHypergraph());
  assignPartitionIDs(coarsest_partitioned_hypergraph);
  PartitionedHypergraph partitioned_hypergraph = uncoarsener->uncoarsen();
  for ( const HypernodeID& hn : partitioned_hypergraph.nodes() ) {
    PartitionID part_id = 0;
    ASSERT_EQ(part_id, partitioned_hypergraph.partID(hn));
  }
}
#endif

}  // namespace mt_kahypar
