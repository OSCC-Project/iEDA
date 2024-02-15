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

#include "mt-kahypar/io/hypergraph_io.h"
#include "mt-kahypar/partition/refinement/flows/quotient_graph.h"
#include "tests/partition/refinement/flow_refiner_mock.h"

using ::testing::Test;

#define MOVE(HN, FROM, TO) Move { FROM, TO, HN, 0 }

namespace mt_kahypar {

/*class AQuotientGraph : public Test {
 public:
  AQuotientGraph() :
    hg(),
    phg(),
    context() {

    context.partition.graph_filename = "../tests/instances/ibm01.hgr";
    context.partition.k = 8;
    context.partition.epsilon = 0.03;
    context.partition.mode = Mode::direct;
    context.partition.objective = Objective::km1;
    context.shared_memory.num_threads = std::thread::hardware_concurrency();
    context.refinement.flows.algorithm = FlowAlgorithm::mock;
    context.refinement.flows.max_bfs_distance = 2;

    // Read hypergraph
    hg = io::readHypergraphFile(context.partition.graph_filename);
    phg = PartitionedHypergraph(context.partition.k, hg, parallel_tag_t());
    context.setupPartWeights(hg.totalWeight());

    // Read Partition
    std::vector<PartitionID> partition;
    io::readPartitionFile("../tests/instances/ibm01.hgr.part8", partition);
    phg.doParallelForAllNodes([&](const HypernodeID& hn) {
      phg.setOnlyNodePart(hn, partition[hn]);
    });
    phg.initializePartition();
  }

  Hypergraph hg;
  PartitionedHypergraph phg;
  Context context;
};

TEST_F(AQuotientGraph, SimulatesBlockScheduling) {
  FlowRefinerMockControl::instance().max_num_blocks = 2;
  FlowRefinerAdapter refiner(hg, context);
  QuotientGraph qg(hg, context);
  qg.initialize(phg);
  const bool debug = false;

  vec<vec<CAtomic<HyperedgeWeight>>> cut_he_weights(
    context.partition.k, vec<CAtomic<HyperedgeWeight>>(
      context.partition.k, CAtomic<HyperedgeWeight>(0)));
  vec<vec<HyperedgeWeight>> initial_cut_he_weights(
    context.partition.k, vec<HyperedgeWeight>(context.partition.k, 0));
  for ( PartitionID i = 0; i < context.partition.k; ++i ) {
    for ( PartitionID j = i + 1; j < context.partition.k; ++j ) {
      cut_he_weights[i][j] += qg.getCutHyperedgeWeightOfBlockPair(i, j);
      initial_cut_he_weights[i][j] = qg.getCutHyperedgeWeightOfBlockPair(i, j);
    }
  }

  tbb::parallel_for(0U, std::thread::hardware_concurrency(), [&](const unsigned int) {
    while ( true ) {
      SearchID search_id = qg.requestNewSearch(refiner);
      if ( search_id != QuotientGraph::INVALID_SEARCH_ID ) {
        BlockPairCutHyperedges block_pair_cut_hes =
          qg.requestCutHyperedges(search_id, 10);
        const PartitionID i = qg.getBlockPair(search_id).i;
        const PartitionID j = qg.getBlockPair(search_id).j;
        size_t num_edges = 0;
        for ( const HyperedgeID& he : block_pair_cut_hes.cut_hes ) {
          cut_he_weights[i][j] -= phg.edgeWeight(he);
          ++num_edges;
        }
        ASSERT_LE(num_edges, 10);
        qg.finalizeConstruction(search_id);
        qg.finalizeSearch(search_id, 0);
        refiner.finalizeSearch(search_id);

        if ( debug ) {
          LOG << "Thread" << THREAD_ID << "executes search on block pair (" << i << "," << j << ")"
              << "with" << block_pair_cut_hes.cut_hes.size() << "cut hyperedges ( Search ID:" << search_id << ")";
        }
      } else {
        break;
      }
    }
  });

  // Each edge should be scheduled once
  for ( PartitionID i = 0; i < context.partition.k; ++i ) {
    for ( PartitionID j = i + 1; j < context.partition.k; ++j ) {
      if ( initial_cut_he_weights[i][j] > 0 ) {
        ASSERT_LT(cut_he_weights[i][j], initial_cut_he_weights[i][j])
          << "Blocks (" << i << "," << j << ") not scheduled!";
      }
    }
  }
}

TEST_F(AQuotientGraph, SimulatesBlockSchedulingWithSuccessfulSearches) {
  FlowRefinerMockControl::instance().max_num_blocks = 2;
  FlowRefinerAdapter refiner(hg, context);
  QuotientGraph qg(hg, context);
  qg.initialize(phg);
  const bool debug = false;

  vec<vec<CAtomic<HyperedgeWeight>>> cut_he_weights(
    context.partition.k, vec<CAtomic<HyperedgeWeight>>(
      context.partition.k, CAtomic<HyperedgeWeight>(0)));
  vec<vec<HyperedgeWeight>> initial_cut_he_weights(
    context.partition.k, vec<HyperedgeWeight>(context.partition.k, 0));
  for ( PartitionID i = 0; i < context.partition.k; ++i ) {
    for ( PartitionID j = i + 1; j < context.partition.k; ++j ) {
      cut_he_weights[i][j] += qg.getCutHyperedgeWeightOfBlockPair(i, j);
      initial_cut_he_weights[i][j] = qg.getCutHyperedgeWeightOfBlockPair(i, j);
    }
  }

  tbb::parallel_for(0U, std::thread::hardware_concurrency(), [&](const unsigned int cpu_id) {
    while ( true ) {
      SearchID search_id = qg.requestNewSearch(refiner);
      if ( search_id != QuotientGraph::INVALID_SEARCH_ID ) {
        BlockPairCutHyperedges block_pair_cut_hes =
          qg.requestCutHyperedges(search_id, 10);
        const PartitionID i = qg.getBlockPair(search_id).i;
        const PartitionID j = qg.getBlockPair(search_id).j;
        HyperedgeWeight cut_he_weight = 0;
        size_t num_edges = 0;
        for ( const HyperedgeID& he : block_pair_cut_hes.cut_hes ) {
          cut_he_weight += phg.edgeWeight(he);
          ++num_edges;
        }

        ASSERT_LE(num_edges, 10);
        cut_he_weights[i][j] -= cut_he_weight;
        qg.finalizeConstruction(search_id);

        bool success = utils::Randomize::instance().flipCoin(cpu_id);
        qg.finalizeSearch(search_id, success);
        refiner.finalizeSearch(search_id);

        if ( debug ) {
          LOG << "Thread" << THREAD_ID << "executes search on block pair (" << i << "," << j << ")"
              << "with" << block_pair_cut_hes.cut_hes.size() << "cut hyperedges ( Search ID:" << search_id << ", Success:"
              << std::boolalpha << success << ")";
        }
      } else {
        break;
      }
    }
  });

  // Each edge should be scheduled once
  for ( PartitionID i = 0; i < context.partition.k; ++i ) {
    for ( PartitionID j = i + 1; j < context.partition.k; ++j ) {
      if ( initial_cut_he_weights[i][j] > 0 ) {
        ASSERT_LT(cut_he_weights[i][j], initial_cut_he_weights[i][j])
          << "Blocks (" << i << "," << j << ") not scheduled!";
      }
    }
  }
}*/

}