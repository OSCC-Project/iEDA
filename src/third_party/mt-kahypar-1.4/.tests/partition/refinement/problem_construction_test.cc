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
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/io/hypergraph_io.h"
#include "mt-kahypar/partition/refinement/flows/problem_construction.h"
#include "tests/partition/refinement/flow_refiner_mock.h"

using ::testing::Test;

namespace mt_kahypar {

namespace {
  using TypeTraits = StaticHypergraphTypeTraits;
  using Hypergraph = typename TypeTraits::Hypergraph;
  using HypergraphFactory = typename Hypergraph::Factory;
  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;
}


class AProblemConstruction : public Test {
 public:
  AProblemConstruction() :
    hg(),
    phg(),
    context(),
    max_part_weights(8, std::numeric_limits<HypernodeWeight>::max()) {

    context.partition.graph_filename = "../tests/instances/ibm01.hgr";
    context.partition.k = 8;
    context.partition.epsilon = 0.03;
    context.partition.mode = Mode::direct;
    context.partition.objective = Objective::km1;
    context.shared_memory.num_threads = std::thread::hardware_concurrency();
    context.refinement.flows.algorithm = FlowAlgorithm::mock;
    context.refinement.flows.max_bfs_distance = 2;

    // Read hypergraph
    hg = io::readInputFile<Hypergraph>(
      context.partition.graph_filename, FileFormat::hMetis, true);
    phg = PartitionedHypergraph(
      context.partition.k, hg, parallel_tag_t());
    context.setupPartWeights(hg.totalWeight());

    // Read Partition
    std::vector<PartitionID> partition;
    io::readPartitionFile("../tests/instances/ibm01.hgr.part8", partition);
    phg.doParallelForAllNodes([&](const HypernodeID& hn) {
      phg.setOnlyNodePart(hn, partition[hn]);
    });
    phg.initializePartition();

    FlowRefinerMockControl::instance().reset();
  }

  void verifyThatPartWeightsAreLessEqualToMaxPartWeight(const Subhypergraph& sub_hg,
                                                        const SearchID search_id,
                                                        const QuotientGraph<TypeTraits>& qg) {
    vec<HypernodeWeight> part_weights(context.partition.k, 0);
    for ( const HypernodeID& hn : sub_hg.nodes_of_block_0 ) {
      part_weights[phg.partID(hn)] += phg.nodeWeight(hn);
    }
    for ( const HypernodeID& hn : sub_hg.nodes_of_block_1 ) {
      part_weights[phg.partID(hn)] += phg.nodeWeight(hn);
    }

    vec<bool> used_blocks(context.partition.k, false);
    const BlockPair blocks = qg.getBlockPair(search_id);
    used_blocks[blocks.i] = true;
    used_blocks[blocks.j] = true;
    for ( PartitionID i = 0; i < context.partition.k; ++i ) {
      if ( used_blocks[i] ) {
        ASSERT_LE(part_weights[i], max_part_weights[i]);
      } else {
        ASSERT_EQ(0, part_weights[i]);
      }
    }
  }

  Hypergraph hg;
  PartitionedHypergraph phg;
  Context context;
  vec<HypernodeWeight> max_part_weights;
};

template <class F, class K>
void executeConcurrent(F f1, K f2) {
  std::atomic<int> cnt(0);

  tbb::parallel_invoke([&] {
    cnt++;
    while (cnt < 2) { }
    f1();
  }, [&] {
    cnt++;
    while (cnt < 2) { }
    f2();
  });
}

void verifyThatVertexSetAreDisjoint(const Subhypergraph& sub_hg_1, const Subhypergraph& sub_hg_2) {
  std::set<HypernodeID> nodes;
  for ( const HypernodeID& hn : sub_hg_1.nodes_of_block_0 ) {
    nodes.insert(hn);
  }
  for ( const HypernodeID& hn : sub_hg_1.nodes_of_block_1 ) {
    nodes.insert(hn);
  }
  for ( const HypernodeID& hn : sub_hg_2.nodes_of_block_0 ) {
    ASSERT_TRUE(nodes.find(hn) == nodes.end());
  }
  for ( const HypernodeID& hn : sub_hg_2.nodes_of_block_1 ) {
    ASSERT_TRUE(nodes.find(hn) == nodes.end());
  }
}

TEST_F(AProblemConstruction, GrowAnFlowProblemAroundTwoBlocks1) {
  ProblemConstruction<TypeTraits> constructor(
    hg.initialNumNodes(), hg.initialNumEdges(), context);
  FlowRefinerAdapter<TypeTraits> refiner(hg.initialNumEdges(), context);
  QuotientGraph<TypeTraits> qg(hg.initialNumEdges(), context);
  refiner.initialize(context.shared_memory.num_threads);
  qg.initialize(phg);

  max_part_weights.assign(context.partition.k, 400);
  max_part_weights[2] = 300;
  SearchID search_id = qg.requestNewSearch(refiner);
  Subhypergraph sub_hg = constructor.construct(search_id, qg, phg);

  verifyThatPartWeightsAreLessEqualToMaxPartWeight(sub_hg, search_id, qg);
}

TEST_F(AProblemConstruction, GrowAnFlowProblemAroundTwoBlocks2) {
  ProblemConstruction<TypeTraits> constructor(
    hg.initialNumNodes(), hg.initialNumEdges(), context);
  FlowRefinerAdapter<TypeTraits> refiner(hg.initialNumEdges(), context);
  QuotientGraph<TypeTraits> qg(hg.initialNumEdges(), context);
  refiner.initialize(context.shared_memory.num_threads);
  qg.initialize(phg);

  max_part_weights.assign(context.partition.k, 800);
  max_part_weights[2] = 500;
  SearchID search_id = qg.requestNewSearch(refiner);
  Subhypergraph sub_hg = constructor.construct(search_id, qg, phg);

  verifyThatPartWeightsAreLessEqualToMaxPartWeight(sub_hg, search_id, qg);
}

TEST_F(AProblemConstruction, GrowTwoFlowProblemAroundTwoBlocksSimultanously) {
  ProblemConstruction<TypeTraits> constructor(
    hg.initialNumNodes(), hg.initialNumEdges(), context);
  FlowRefinerAdapter<TypeTraits> refiner(hg.initialNumEdges(), context);
  QuotientGraph<TypeTraits> qg(hg.initialNumEdges(), context);
  refiner.initialize(context.shared_memory.num_threads);
  qg.initialize(phg);

  max_part_weights.assign(context.partition.k, 400);

  Subhypergraph sub_hg_1;
  Subhypergraph sub_hg_2;
  executeConcurrent([&] {
    SearchID search_id = qg.requestNewSearch(refiner);
     sub_hg_1 = constructor.construct(search_id, qg, phg);
    verifyThatPartWeightsAreLessEqualToMaxPartWeight(sub_hg_1, search_id, qg);
  }, [&] {
    SearchID search_id = qg.requestNewSearch(refiner);
    sub_hg_2 = constructor.construct(search_id, qg, phg);
    verifyThatPartWeightsAreLessEqualToMaxPartWeight(sub_hg_2, search_id, qg);
  });
  verifyThatVertexSetAreDisjoint(sub_hg_1, sub_hg_2);
}

}