/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2021 Tobias Heuer <tobias.heuer@kit.edu>
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

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/context_enum_classes.h"
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/partition/refinement/gains/bipartitioning_policy.h"
#include "mt-kahypar/partition/initial_partitioning/bfs_initial_partitioner.h"

using ::testing::Test;

namespace mt_kahypar {

namespace {
  using Hypergraph = ds::StaticHypergraph;
  using PartitionedHypergraph = ds::PartitionedHypergraph<Hypergraph, ds::ConnectivityInfo>;
}

template<Objective obj>
struct ObjectiveF {
  static constexpr Objective objective = obj;
};

template<typename ObjectiveFunc>
class ABipartitioningPolicy : public Test {

 public:
  ABipartitioningPolicy() :
    hypergraph(),
    partitioned_hg(),
    context() {
    hypergraph = io::readInputFile<Hypergraph>(
      "../tests/instances/contracted_ibm01.hgr", FileFormat::hMetis, true);
    partitioned_hg = PartitionedHypergraph(4, hypergraph, parallel_tag_t { });

    context.partition.mode = Mode::direct;
    context.partition.objective = ObjectiveFunc::objective;
    context.partition.epsilon = 0.1;
    context.partition.k = 2;
    context.partition.preset_type = PresetType::default_preset;
    context.partition.instance_type = InstanceType::hypergraph;
    context.partition.partition_type = PartitionedHypergraph::TYPE;
    context.partition.verbose_output = false;
    context.initial_partitioning.mode = Mode::deep_multilevel;
    context.initial_partitioning.runs = 1;

    context.load_default_preset();
    context.sanityCheck(nullptr);
    context.setupPartWeights(hypergraph.totalWeight());
  }

  void partition(PartitionedHypergraph& phg) {
    Context ip_context(context);
    ip_context.refinement.label_propagation.algorithm = LabelPropagationAlgorithm::do_nothing;
    ip_context.initial_partitioning.fm_refinment_rounds = 0;
    InitialPartitioningDataContainer<StaticHypergraphTypeTraits> ip_data(phg, ip_context);
    ip_data_container_t* ip_data_ptr = ip::to_pointer(ip_data);
    BFSInitialPartitioner<StaticHypergraphTypeTraits> initial_partitioner(
      InitialPartitioningAlgorithm::bfs, ip_data_ptr, ip_context, 420, 0);
    initial_partitioner.partition();
    ip_data.apply();
  }

  Hypergraph hypergraph;
  PartitionedHypergraph partitioned_hg;
  Context context;
};

typedef ::testing::Types<ObjectiveF<Objective::cut>,
                         ObjectiveF<Objective::km1>
                         ENABLE_SOED(COMMA ObjectiveF<Objective::soed>)> TestConfigs;

TYPED_TEST_CASE(ABipartitioningPolicy, TestConfigs);

TYPED_TEST(ABipartitioningPolicy, ModelsObjectiveFunctionCorrectlyWhenPerformingRecursiveBipartitioning) {
  const HyperedgeWeight non_cut_edge_multiplier =
    BipartitioningPolicy::nonCutEdgeMultiplier(this->context.partition.gain_policy);
  auto adapt_edge_weights = [&](PartitionedHypergraph& phg,
                                const vec<uint8_t>& already_cut,
                                const double multiplier) {
    phg.doParallelForAllEdges([&](const HyperedgeID& he) {
      if ( !already_cut[he] ) {
        phg.setEdgeWeight(he, multiplier * phg.edgeWeight(he));
      }
    });
  };


  // Bipartition input
  vec<uint8_t> already_cut(this->partitioned_hg.initialNumEdges(), 0);
  adapt_edge_weights(this->partitioned_hg, already_cut, non_cut_edge_multiplier);
  this->partition(this->partitioned_hg);
  const HyperedgeWeight cut_0 = metrics::quality(this->partitioned_hg, Objective::cut);
  adapt_edge_weights(this->partitioned_hg, already_cut, 1.0 / non_cut_edge_multiplier);

  // Extract Cut Hyperedges
  this->partitioned_hg.doParallelForAllEdges([&](const HyperedgeID& he) {
    if ( this->partitioned_hg.connectivity(he) > 1 ) {
      already_cut[he] = 1;
    }
  });

  // Bipartition block 0
  auto extracted_0 = this->partitioned_hg.extract(0, &already_cut,
    BipartitioningPolicy::useCutNetSplitting(this->context.partition.gain_policy), true);
  Hypergraph hg_0 = std::move(extracted_0.hg);
  vec<uint8_t> already_cut_0 = std::move(extracted_0.already_cut);
  PartitionedHypergraph phg_0(2, hg_0, parallel_tag_t { });
  adapt_edge_weights(phg_0, already_cut_0, non_cut_edge_multiplier);
  this->partition(phg_0);
  const HyperedgeWeight cut_1 = metrics::quality(phg_0, Objective::cut);
  adapt_edge_weights(phg_0, already_cut_0, 1.0 / non_cut_edge_multiplier);

  // Bipartition block 1
  auto extracted_1 = this->partitioned_hg.extract(1, &already_cut,
    BipartitioningPolicy::useCutNetSplitting(this->context.partition.gain_policy), true);
  Hypergraph hg_1 = std::move(extracted_1.hg);
  vec<uint8_t> already_cut_1 = std::move(extracted_1.already_cut);
  PartitionedHypergraph phg_1(2, hg_1, parallel_tag_t { });
  adapt_edge_weights(phg_1, already_cut_1, non_cut_edge_multiplier);
  this->partition(phg_1);
  const HyperedgeWeight cut_2 = metrics::quality(phg_1, Objective::cut);
  adapt_edge_weights(phg_1, already_cut_1, 1.0 / non_cut_edge_multiplier);

  // Apply bipartitions to top-level hypergraph to obtain an 4-way partition
  this->partitioned_hg.doParallelForAllNodes([&](const HypernodeID hn) {
    const PartitionID from = this->partitioned_hg.partID(hn);
    PartitionID to = kInvalidPartition;
    const HypernodeID mapped_hn = from == 0 ?
      extracted_0.hn_mapping[hn] : extracted_1.hn_mapping[hn];
    if ( from == 0 ) to = phg_0.partID(mapped_hn) == 0 ? 0 : 1;
    else if (from == 1) to = phg_1.partID(mapped_hn) == 0 ? 2 : 3;
    if ( from != to ) {
      this->partitioned_hg.changeNodePart(hn, from, to);
    }
  });

  // The cut values of each bipartition should sum
  // up to the quality of the 4-way partition
  ASSERT_EQ(cut_0 + cut_1 + cut_2,
    metrics::quality(this->partitioned_hg, this->context.partition.objective));
}

TYPED_TEST(ABipartitioningPolicy, ModelsObjectiveFunctionCorrectlyWhenPerformingDeepMultilevelPartitioning) {
  const HyperedgeWeight non_cut_edge_multiplier =
    BipartitioningPolicy::nonCutEdgeMultiplier(this->context.partition.gain_policy);
  auto adapt_edge_weights = [&](PartitionedHypergraph& phg,
                                const vec<uint8_t>& already_cut,
                                const double multiplier) {
    phg.doParallelForAllEdges([&](const HyperedgeID& he) {
      if ( !already_cut[he] ) {
        phg.setEdgeWeight(he, multiplier * phg.edgeWeight(he));
      }
    });
  };


  // Bipartition input
  vec<uint8_t> already_cut(this->partitioned_hg.initialNumEdges(), 0);
  adapt_edge_weights(this->partitioned_hg, already_cut, non_cut_edge_multiplier);
  this->partition(this->partitioned_hg);
  const HyperedgeWeight cut_0 = metrics::quality(this->partitioned_hg, Objective::cut);
  adapt_edge_weights(this->partitioned_hg, already_cut, 1.0 / non_cut_edge_multiplier);

  // Extract Cut Hyperedges
  this->partitioned_hg.doParallelForAllEdges([&](const HyperedgeID& he) {
    if ( this->partitioned_hg.connectivity(he) > 1 ) {
      already_cut[he] = 1;
    }
  });

  // Extract all blocks
  auto extracted = this->partitioned_hg.extractAllBlocks(2, &already_cut,
    BipartitioningPolicy::useCutNetSplitting(this->context.partition.gain_policy), true);
  const vec<HypernodeID>& hn_mapping = extracted.second;

  // Bipartition block 0
  Hypergraph hg_0 = std::move(extracted.first[0].hg);
  vec<uint8_t> already_cut_0 = std::move(extracted.first[0].already_cut);
  PartitionedHypergraph phg_0(2, hg_0, parallel_tag_t { });
  adapt_edge_weights(phg_0, already_cut_0, non_cut_edge_multiplier);
  this->partition(phg_0);
  const HyperedgeWeight cut_1 = metrics::quality(phg_0, Objective::cut);
  adapt_edge_weights(phg_0, already_cut_0, 1.0 / non_cut_edge_multiplier);

  // Bipartition block 1
  Hypergraph hg_1 = std::move(extracted.first[1].hg);
  vec<uint8_t> already_cut_1 = std::move(extracted.first[1].already_cut);
  PartitionedHypergraph phg_1(2, hg_1, parallel_tag_t { });
  adapt_edge_weights(phg_1, already_cut_1, non_cut_edge_multiplier);
  this->partition(phg_1);
  const HyperedgeWeight cut_2 = metrics::quality(phg_1, Objective::cut);
  adapt_edge_weights(phg_1, already_cut_1, 1.0 / non_cut_edge_multiplier);

  // Apply bipartitions to top-level hypergraph to obtain an 4-way partition
  this->partitioned_hg.doParallelForAllNodes([&](const HypernodeID hn) {
    const PartitionID from = this->partitioned_hg.partID(hn);
    PartitionID to = kInvalidPartition;
    const HypernodeID mapped_hn = hn_mapping[hn];
    if ( from == 0 ) to = phg_0.partID(mapped_hn) == 0 ? 0 : 1;
    else if (from == 1) to = phg_1.partID(mapped_hn) == 0 ? 2 : 3;
    if ( from != to ) {
      this->partitioned_hg.changeNodePart(hn, from, to);
    }
  });

  // The cut values of each bipartition should sum
  // up to the quality of the 4-way partition
  ASSERT_EQ(cut_0 + cut_1 + cut_2,
    metrics::quality(this->partitioned_hg, this->context.partition.objective));
}

}