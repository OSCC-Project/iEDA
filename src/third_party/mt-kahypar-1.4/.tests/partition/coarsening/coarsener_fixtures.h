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

#include "kahypar-resources/datastructure/fast_reset_flag_array.h"

#include "tests/datastructures/hypergraph_fixtures.h"
#include "mt-kahypar/definitions.h"
#include "mt-kahypar/utils/cast.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/refinement/i_refiner.h"
#include "mt-kahypar/partition/coarsening/coarsening_commons.h"
#include "mt-kahypar/partition/coarsening/policies/rating_heavy_node_penalty_policy.h"
#include "mt-kahypar/partition/coarsening/policies/rating_score_policy.h"

using ::testing::Test;
using ::testing::Eq;
using ::testing::Le;

namespace mt_kahypar {

namespace tmp {
class BestRatingWithoutTieBreaking final : public kahypar::meta::PolicyBase {
 public:
  MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE static bool acceptRating(const RatingType tmp,
                                                              const RatingType max_rating,
                                                              const HypernodeID u,
                                                              const HypernodeID v,
                                                              const int,
                                                              const kahypar::ds::FastResetFlagArray<> &) {
    return max_rating < tmp || ( max_rating == tmp && u < v );
  }
};
}

template<typename TypeTraits,
         template<typename, typename, typename, typename> typename CoarsenerT,
         template<typename> typename UncoarsenerT,
         PresetType PRESET >
class ACoarsener : public Test {
 private:
  using Hypergraph = typename TypeTraits::Hypergraph;
  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;
  using HypergraphFactory = typename Hypergraph::Factory;
  using Coarsener = CoarsenerT<TypeTraits,
    HeavyEdgeScore, NoWeightPenalty, tmp::BestRatingWithoutTieBreaking>;
  using Uncoarsener = UncoarsenerT<TypeTraits>;

 public:
  ACoarsener() :
    hypergraph(HypergraphFactory::construct(16, 18, { { 0, 1 }, { 0, 1, 3 }, { 1, 2, 3 }, { 2, 3, 4 }, { 2, 4 },
                { 4, 5 }, { 4, 5, 7 }, { 5, 6, 7 }, { 6, 7, 8 }, { 6, 8 },
                { 8, 9 }, { 8, 9, 11 }, { 9, 10, 11 }, { 10, 11, 12 }, { 10, 12 },
                { 12, 13 }, { 12, 13, 15 }, { 13, 14, 15 } }, nullptr, nullptr, true)),
    context(),
    uncoarseningData(nullptr),
    coarsener(nullptr),
    uncoarsener(nullptr),
    nullptr_refiner(nullptr) {
    for ( const HypernodeID& hn : hypergraph.nodes() ) {
      hypergraph.setCommunityID(hn, hn / 4);
    }

    context.partition.k = 2;
    context.partition.mode = Mode::direct;
    context.partition.preset_type = PRESET;
    context.partition.instance_type = InstanceType::hypergraph;
    context.partition.partition_type = PartitionedHypergraph::TYPE;
    context.partition.objective = Objective::km1;
    context.partition.gain_policy = GainPolicy::km1;
    context.coarsening.max_allowed_node_weight = std::numeric_limits<HypernodeWeight>::max();
    context.coarsening.contraction_limit = 8;
    context.coarsening.minimum_shrink_factor = 1.0;
    context.coarsening.maximum_shrink_factor = 4.0;
    context.refinement.max_batch_size = 5;
    context.shared_memory.original_num_threads = std::thread::hardware_concurrency();
    context.shared_memory.num_threads = std::thread::hardware_concurrency();
    context.setupPartWeights(hypergraph.totalWeight());

    uncoarseningData = std::make_unique<UncoarseningData<TypeTraits>>(
      PRESET != PresetType::default_preset, hypergraph, context);

    mt_kahypar_hypergraph_t hg = utils::hypergraph_cast(hypergraph);
    uncoarsening_data_t* data_ptr = uncoarsening::to_pointer(*uncoarseningData);
    coarsener = std::make_unique<Coarsener>(hg, context, data_ptr);
    uncoarsener = std::make_unique<Uncoarsener>(hypergraph, context, *uncoarseningData, nullptr);
  }

  void assignPartitionIDs(PartitionedHypergraph& phg) {
    for (const HypernodeID& hn : phg.nodes()) {
      PartitionID part_id = 0;
      phg.setNodePart(hn, part_id);
    }
  }

  HypernodeID currentNumNodes(mt_kahypar_hypergraph_t hg_ptr) {
    Hypergraph& hg = utils::cast<Hypergraph>(hg_ptr);
    HypernodeID num_nodes = 0;
    for (const HypernodeID& hn : hg.nodes()) {
      unused(hn);
      ++num_nodes;
    }
    return num_nodes;
  }

  HyperedgeID currentNumEdges(mt_kahypar_hypergraph_t hg_ptr) {
    Hypergraph& hg = utils::cast<Hypergraph>(hg_ptr);
    HyperedgeID num_edges = 0;
    for (const HyperedgeID& he : hg.edges()) {
      unused(he);
      ++num_edges;
    }
    return num_edges;
  }

  HypernodeID currentNumPins(mt_kahypar_hypergraph_t hg_ptr) {
    Hypergraph& hg = utils::cast<Hypergraph>(hg_ptr);
    HypernodeID num_pins = 0;
    for (const HypernodeID& he : hg.edges()) {
      num_pins += hg.edgeSize(he);
    }
    return num_pins;
  }

  void doCoarsening() {
    coarsener->disableRandomization();
    coarsener->coarsen();
  }

  void decreasesNumberOfPins(const size_t number_of_pins) {
    doCoarsening();
    ASSERT_THAT(currentNumPins(coarsener->coarsestHypergraph()), Eq(number_of_pins));
  }

  void decreasesNumberOfHyperedges(const HyperedgeID num_hyperedges) {
    doCoarsening();
    ASSERT_THAT(currentNumEdges(coarsener->coarsestHypergraph()), Eq(num_hyperedges));
  }

  Hypergraph hypergraph;
  Context context;
  std::unique_ptr<UncoarseningData<TypeTraits>> uncoarseningData;
  std::unique_ptr<Coarsener> coarsener;
  std::unique_ptr<Uncoarsener> uncoarsener;
  std::unique_ptr<IRefiner> nullptr_refiner;
};
}  // namespace mt_kahypar
