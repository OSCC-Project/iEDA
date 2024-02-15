/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2019 Lars Gottesbüren <lars.gottesbueren@kit.edu>
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

#include "kahypar-resources/meta/registrar.h"
#include "kahypar-resources/meta/static_multi_dispatch_factory.h"
#include "kahypar-resources/meta/typelist.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/factories.h"
#include "mt-kahypar/partition/refinement/do_nothing_refiner.h"
#include "mt-kahypar/partition/refinement/label_propagation/label_propagation_refiner.h"
#include "mt-kahypar/partition/refinement/deterministic/deterministic_label_propagation.h"
#include "mt-kahypar/partition/refinement/fm/multitry_kway_fm.h"
#include "mt-kahypar/partition/refinement/fm/strategies/gain_cache_strategy.h"
#include "mt-kahypar/partition/refinement/fm/strategies/unconstrained_strategy.h"
#include "mt-kahypar/partition/refinement/flows/do_nothing_refiner.h"
#include "mt-kahypar/partition/refinement/flows/scheduler.h"
#include "mt-kahypar/partition/refinement/flows/flow_refiner.h"
#include "mt-kahypar/partition/refinement/gains/gain_definitions.h"
#include "mt-kahypar/partition/refinement/rebalancing/simple_rebalancer.h"
#include "mt-kahypar/partition/refinement/rebalancing/advanced_rebalancer.h"


namespace mt_kahypar {
using LabelPropagationDispatcher = kahypar::meta::StaticMultiDispatchFactory<
                                   LabelPropagationRefiner,
                                   IRefiner,
                                   kahypar::meta::Typelist<GraphAndGainTypesList>>;

using DeterministicLabelPropagationDispatcher = kahypar::meta::StaticMultiDispatchFactory<
                                                DeterministicLabelPropagationRefiner,
                                                IRefiner,
                                                kahypar::meta::Typelist<GraphAndGainTypesList>>;

using DefaultFMDispatcher = kahypar::meta::StaticMultiDispatchFactory<
                            MultiTryKWayFM,
                            IRefiner,
                            kahypar::meta::Typelist<GraphAndGainTypesList>>;

using UnconstrainedFMDispatcher = DefaultFMDispatcher;

using GainCacheFMStrategyDispatcher = kahypar::meta::StaticMultiDispatchFactory<
                                      GainCacheStrategy,
                                      IFMStrategy,
                                      kahypar::meta::Typelist<GraphAndGainTypesList>>;

using UnconstrainedFMStrategyDispatcher = kahypar::meta::StaticMultiDispatchFactory<
                                          UnconstrainedStrategy,
                                          IFMStrategy,
                                          kahypar::meta::Typelist<GraphAndGainTypesList>>;

using FlowSchedulerDispatcher = kahypar::meta::StaticMultiDispatchFactory<
                                FlowRefinementScheduler,
                                IRefiner,
                                kahypar::meta::Typelist<GraphAndGainTypesList>>;

using SimpleRebalancerDispatcher = kahypar::meta::StaticMultiDispatchFactory<
                                   SimpleRebalancer,
                                   IRebalancer,
                                   kahypar::meta::Typelist<GraphAndGainTypesList>>;

using AdvancedRebalancerDispatcher = kahypar::meta::StaticMultiDispatchFactory<
                                     AdvancedRebalancer,
                                     IRebalancer,
                                     kahypar::meta::Typelist<GraphAndGainTypesList>>;

using FlowRefinementDispatcher = kahypar::meta::StaticMultiDispatchFactory<
                                 FlowRefiner,
                                 IFlowRefiner,
                                 kahypar::meta::Typelist<GraphAndGainTypesList>>;


#define REGISTER_DISPATCHED_LP_REFINER(id, dispatcher, ...)                                            \
  static kahypar::meta::Registrar<LabelPropagationFactory> register_ ## dispatcher(                    \
    id,                                                                                                \
    [](const HypernodeID num_hypernodes, const HyperedgeID num_hyperedges,                             \
       const Context& context, gain_cache_t gain_cache, IRebalancer& rebalancer) {                     \
    return dispatcher::create(                                                                         \
      std::forward_as_tuple(num_hypernodes, num_hyperedges, context, gain_cache, rebalancer),          \
      __VA_ARGS__                                                                                      \
      );                                                                                               \
  })

#define REGISTER_LP_REFINER(id, refiner, t)                                                      \
  static kahypar::meta::Registrar<LabelPropagationFactory> JOIN(register_ ## refiner, t)(        \
    id,                                                                                          \
    [](const HypernodeID num_hypernodes, const HyperedgeID num_hyperedges,                       \
       const Context& context, gain_cache_t gain_cache, IRebalancer& rebalancer) -> IRefiner* {  \
    return new refiner(num_hypernodes, num_hyperedges, context, gain_cache, rebalancer);         \
  })

#define REGISTER_DISPATCHED_FM_REFINER(id, dispatcher, ...)                                            \
  static kahypar::meta::Registrar<FMFactory> register_ ## dispatcher(                                  \
    id,                                                                                                \
    [](const HypernodeID num_hypernodes, const HyperedgeID num_hyperedges,                             \
       const Context& context, gain_cache_t gain_cache, IRebalancer& rebalancer) {                     \
    return dispatcher::create(                                                                         \
      std::forward_as_tuple(num_hypernodes, num_hyperedges, context, gain_cache, rebalancer),          \
      __VA_ARGS__                                                                                      \
      );                                                                                               \
  })

#define REGISTER_FM_REFINER(id, refiner, t)                                                      \
  static kahypar::meta::Registrar<FMFactory> JOIN(register_ ## refiner, t)(                      \
    id,                                                                                          \
    [](const HypernodeID num_hypernodes, const HyperedgeID num_hyperedges,                       \
       const Context& context, gain_cache_t gain_cache, IRebalancer& rebalancer) -> IRefiner* {  \
    return new refiner(num_hypernodes, num_hyperedges, context, gain_cache, rebalancer);         \
  })

#define REGISTER_DISPATCHED_FM_STRATEGY(id, dispatcher, ...)                                           \
  static kahypar::meta::Registrar<FMStrategyFactory> register_ ## dispatcher(                          \
    id,                                                                                                \
    [](const Context& context, FMSharedData& shared_data) {                                            \
    return dispatcher::create(                                                                         \
      std::forward_as_tuple(context, shared_data),                                                     \
      __VA_ARGS__                                                                                      \
      );                                                                                               \
  })

#define REGISTER_DISPATCHED_FLOW_SCHEDULER(id, dispatcher, ...)                                        \
  static kahypar::meta::Registrar<FlowSchedulerFactory> register_ ## dispatcher(                       \
    id,                                                                                                \
    [](const HypernodeID num_hypernodes, const HyperedgeID num_hyperedges,                             \
       const Context& context, gain_cache_t gain_cache) {                                              \
    return dispatcher::create(                                                                         \
      std::forward_as_tuple(num_hypernodes, num_hyperedges, context, gain_cache),                      \
      __VA_ARGS__                                                                                      \
      );                                                                                               \
  })

#define REGISTER_FLOW_SCHEDULER(id, refiner, t)                                                  \
  static kahypar::meta::Registrar<FlowSchedulerFactory> JOIN(register_ ## refiner, t)(           \
    id,                                                                                          \
    [](const HypernodeID num_hypernodes, const HyperedgeID num_hyperedges,                       \
       const Context& context, gain_cache_t gain_cache) -> IRefiner* {                           \
    return new refiner(num_hypernodes, num_hyperedges, context, gain_cache);                     \
  })

#define REGISTER_DISPATCHED_REBALANCER(id, dispatcher, ...)                                            \
  static kahypar::meta::Registrar<RebalancerFactory> register_ ## dispatcher(                          \
    id,                                                                                                \
    [](HypernodeID num_hypernodes, const Context& context, gain_cache_t gain_cache) {                  \
    return dispatcher::create(                                                                         \
      std::forward_as_tuple(num_hypernodes, context, gain_cache),                                      \
      __VA_ARGS__                                                                                      \
      );                                                                                               \
  })

#define REGISTER_REBALANCER(id, refiner, t)                                                            \
  static kahypar::meta::Registrar<RebalancerFactory> JOIN(register_ ## refiner, t)(                    \
    id,                                                                                                \
    [](HypernodeID num_hypernodes, const Context& context, gain_cache_t gain_cache) -> IRebalancer* {  \
    return new refiner(num_hypernodes, context, gain_cache);                                           \
  })

#define REGISTER_DISPATCHED_FLOW_REFINER(id, dispatcher, ...)                                          \
  static kahypar::meta::Registrar<FlowRefinementFactory> register_ ## dispatcher(                      \
    id,                                                                                                \
    [](const HyperedgeID num_hyperedges, const Context& context) {                                     \
    return dispatcher::create(                                                                         \
      std::forward_as_tuple(num_hyperedges, context),                                                  \
      __VA_ARGS__                                                                                      \
      );                                                                                               \
  })

#define REGISTER_FLOW_REFINER(id, refiner, t)                                                   \
  static kahypar::meta::Registrar<FlowRefinementFactory> JOIN(register_ ## refiner, t)(         \
    id,                                                                                         \
    [](const HyperedgeID num_Hyperedges, const Context& context) -> IFlowRefiner* {             \
    return new refiner(num_Hyperedges, context);                                                \
  })


kahypar::meta::PolicyBase& getGraphAndGainTypesPolicy(mt_kahypar_partition_type_t partition_type, GainPolicy gain_policy) {
  switch ( partition_type ) {
    case MULTILEVEL_HYPERGRAPH_PARTITIONING: SWITCH_HYPERGRAPH_GAIN_TYPES(StaticHypergraphTypeTraits, gain_policy);
    case MULTILEVEL_GRAPH_PARTITIONING: SWITCH_GRAPH_GAIN_TYPES(StaticGraphTypeTraits, gain_policy);
    case N_LEVEL_HYPERGRAPH_PARTITIONING: SWITCH_HYPERGRAPH_GAIN_TYPES(DynamicHypergraphTypeTraits, gain_policy);
    case N_LEVEL_GRAPH_PARTITIONING: SWITCH_GRAPH_GAIN_TYPES(DynamicGraphTypeTraits, gain_policy);
    case LARGE_K_PARTITIONING: SWITCH_HYPERGRAPH_GAIN_TYPES(LargeKHypergraphTypeTraits, gain_policy);
    default: {
      LOG << "Invalid partition type";
      std::exit(-1);
    }
  }
}


REGISTER_DISPATCHED_LP_REFINER(LabelPropagationAlgorithm::label_propagation,
                               LabelPropagationDispatcher,
                               getGraphAndGainTypesPolicy(context.partition.partition_type, context.partition.gain_policy));
REGISTER_DISPATCHED_LP_REFINER(LabelPropagationAlgorithm::deterministic,
                               DeterministicLabelPropagationDispatcher,
                               getGraphAndGainTypesPolicy(context.partition.partition_type, context.partition.gain_policy));
REGISTER_LP_REFINER(LabelPropagationAlgorithm::do_nothing, DoNothingRefiner, 1);

REGISTER_DISPATCHED_FM_REFINER(FMAlgorithm::kway_fm,
                               DefaultFMDispatcher,
                               getGraphAndGainTypesPolicy(context.partition.partition_type, context.partition.gain_policy));
REGISTER_DISPATCHED_FM_REFINER(FMAlgorithm::unconstrained_fm,
                               UnconstrainedFMDispatcher,
                               getGraphAndGainTypesPolicy(context.partition.partition_type, context.partition.gain_policy));
REGISTER_FM_REFINER(FMAlgorithm::do_nothing, DoNothingRefiner, 3);

REGISTER_DISPATCHED_FM_STRATEGY(FMAlgorithm::kway_fm,
                                GainCacheFMStrategyDispatcher,
                                getGraphAndGainTypesPolicy(context.partition.partition_type, context.partition.gain_policy));
REGISTER_DISPATCHED_FM_STRATEGY(FMAlgorithm::unconstrained_fm,
                                UnconstrainedFMStrategyDispatcher,
                                getGraphAndGainTypesPolicy(context.partition.partition_type, context.partition.gain_policy));

REGISTER_DISPATCHED_FLOW_SCHEDULER(FlowAlgorithm::flow_cutter,
                                   FlowSchedulerDispatcher,
                                   getGraphAndGainTypesPolicy(context.partition.partition_type, context.partition.gain_policy));
REGISTER_FLOW_SCHEDULER(FlowAlgorithm::do_nothing, DoNothingRefiner, 4);

REGISTER_DISPATCHED_REBALANCER(RebalancingAlgorithm::simple_rebalancer,
                               SimpleRebalancerDispatcher,
                               getGraphAndGainTypesPolicy(context.partition.partition_type, context.partition.gain_policy));
REGISTER_DISPATCHED_REBALANCER(RebalancingAlgorithm::advanced_rebalancer,
                               AdvancedRebalancerDispatcher,
                               getGraphAndGainTypesPolicy(context.partition.partition_type, context.partition.gain_policy));
REGISTER_REBALANCER(RebalancingAlgorithm::do_nothing, DoNothingRefiner, 5);

REGISTER_DISPATCHED_FLOW_REFINER(FlowAlgorithm::flow_cutter,
                                 FlowRefinementDispatcher,
                                 getGraphAndGainTypesPolicy(context.partition.partition_type, context.partition.gain_policy));
REGISTER_FLOW_REFINER(FlowAlgorithm::do_nothing, DoNothingFlowRefiner, 6);
}  // namespace mt_kahypar
