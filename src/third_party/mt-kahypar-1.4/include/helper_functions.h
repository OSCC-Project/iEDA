/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2023 Tobias Heuer <tobias.heuer@kit.edu>
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

#pragma once

#include <string>
#include <sstream>

#include "libmtkahypartypes.h"

#include "mt-kahypar/partition/context.h"

using namespace mt_kahypar;

namespace lib {
bool check_compatibility(mt_kahypar_hypergraph_t hypergraph,
                         mt_kahypar_preset_type_t preset) {
  switch ( preset ) {
    case DEFAULT:
    case QUALITY:
    case DETERMINISTIC:
    case LARGE_K:
      return hypergraph.type == STATIC_GRAPH || hypergraph.type == STATIC_HYPERGRAPH;
    case HIGHEST_QUALITY:
      return hypergraph.type == DYNAMIC_GRAPH || hypergraph.type == DYNAMIC_HYPERGRAPH;
  }
  return false;
}

bool check_compatibility(mt_kahypar_partitioned_hypergraph_t partitioned_hg,
                         mt_kahypar_preset_type_t preset) {
  switch ( preset ) {
    case DEFAULT:
    case QUALITY:
    case DETERMINISTIC:
      return partitioned_hg.type == MULTILEVEL_GRAPH_PARTITIONING ||
             partitioned_hg.type == MULTILEVEL_HYPERGRAPH_PARTITIONING;
    case LARGE_K:
      return partitioned_hg.type == MULTILEVEL_GRAPH_PARTITIONING ||
             partitioned_hg.type == LARGE_K_PARTITIONING;
    case HIGHEST_QUALITY:
      return partitioned_hg.type == N_LEVEL_GRAPH_PARTITIONING ||
             partitioned_hg.type == N_LEVEL_HYPERGRAPH_PARTITIONING;
  }
  return false;
}


bool check_if_all_relavant_parameters_are_set(Context& context) {
  bool success = true;
  if ( context.partition.preset_type == PresetType::UNDEFINED ) {
    WARNING("Preset type not specified. Either use mt_kahypar_load_preset(...) or specify"
      << "parameter 'preset-type' in your configuration file!");
    success = false;
  }
  if ( context.partition.k == std::numeric_limits<PartitionID>::max() ) {
    WARNING("Number of blocks not specified.");
    success = false;
  }
  if ( context.partition.epsilon == std::numeric_limits<double>::max() ) {
    WARNING("Imbalance not specified.");
    success = false;
  }
  if ( context.partition.objective == Objective::UNDEFINED ) {
    WARNING("Objective function not specified.");
    success = false;
  }
  return success;
}

void prepare_context(Context& context) {
  context.shared_memory.original_num_threads = mt_kahypar::TBBInitializer::instance().total_number_of_threads();
  context.shared_memory.num_threads = mt_kahypar::TBBInitializer::instance().total_number_of_threads();
  context.utility_id = mt_kahypar::utils::Utilities::instance().registerNewUtilityObjects();

  context.partition.perfect_balance_part_weights.clear();
  if ( !context.partition.use_individual_part_weights ) {
    context.partition.max_part_weights.clear();
  }
}

InstanceType get_instance_type(mt_kahypar_hypergraph_t hypergraph) {
  switch ( hypergraph.type ) {
    case STATIC_GRAPH:
    case DYNAMIC_GRAPH:
      return InstanceType::graph;
    case STATIC_HYPERGRAPH:
    case DYNAMIC_HYPERGRAPH:
      return InstanceType::hypergraph;
    case NULLPTR_HYPERGRAPH:
      return InstanceType::UNDEFINED;
  }
  return InstanceType::UNDEFINED;
}

InstanceType get_instance_type(mt_kahypar_partitioned_hypergraph_t partitioned_hg) {
  switch ( partitioned_hg.type ) {
    case MULTILEVEL_GRAPH_PARTITIONING:
    case N_LEVEL_GRAPH_PARTITIONING:
      return InstanceType::graph;
    case MULTILEVEL_HYPERGRAPH_PARTITIONING:
    case N_LEVEL_HYPERGRAPH_PARTITIONING:
    case LARGE_K_PARTITIONING:
      return InstanceType::hypergraph;
    case NULLPTR_PARTITION:
      return InstanceType::UNDEFINED;
  }
  return InstanceType::UNDEFINED;
}

mt_kahypar_preset_type_t get_preset_c_type(const PresetType preset) {
  switch ( preset ) {
    case PresetType::default_preset: return DEFAULT;
    case PresetType::quality: return QUALITY;
    case PresetType::highest_quality: return HIGHEST_QUALITY;
    case PresetType::deterministic: return DETERMINISTIC;
    case PresetType::large_k: return LARGE_K;
    case PresetType::UNDEFINED: return DEFAULT;
  }
  return DEFAULT;
}

std::string incompatibility_description(mt_kahypar_hypergraph_t hypergraph) {
  std::stringstream ss;
  switch ( hypergraph.type ) {
    case STATIC_GRAPH:
      ss << "The hypergraph uses the static graph data structure which can be only used "
         << "in combination with the following presets: "
         << "DEFAULT, QUALITY, DETERMINISTIC and LARGE_K"; break;
    case DYNAMIC_GRAPH:
      ss << "The hypergraph uses the dynamic graph data structure which can be only used "
         << "in combination with the following preset: "
         << "HIGHEST_QUALITY"; break;
    case STATIC_HYPERGRAPH:
      ss << "The hypergraph uses the static hypergraph data structure which can be only used "
         << "in combination with the following presets: "
         << "DEFAULT, QUALITY, DETERMINISTIC and LARGE_K"; break;
    case DYNAMIC_HYPERGRAPH:
      ss << "The hypergraph uses the dynamic hypergraph data structure which can be only used "
         << "in combination with the following preset: "
         << "HIGHEST_QUALITY"; break;
    case NULLPTR_HYPERGRAPH:
      ss << "The hypergraph holds a nullptr. "
         << "Did you forgot to construct or load a hypergraph?"; break;
  }
  return ss.str();
}

std::string incompatibility_description(mt_kahypar_partitioned_hypergraph_t partitioned_hg) {
  std::stringstream ss;
  switch ( partitioned_hg.type ) {
    case MULTILEVEL_GRAPH_PARTITIONING:
      ss << "The partitioned hypergraph uses the data structures for multilevel graph partitioning "
         << "which can be only used in combination with the following presets: "
         << "DEFAULT, QUALITY, DETERMINISTIC, and LARGE_K"; break;
    case N_LEVEL_GRAPH_PARTITIONING:
      ss << "The partitioned hypergraph uses the data structures for n-level graph partitioning "
         << "which can be only used in combination with the following preset: "
         << "HIGHEST_QUALITY"; break;
    case MULTILEVEL_HYPERGRAPH_PARTITIONING:
      ss << "The partitioned hypergraph uses the data structures for multilevel hypergraph partitioning "
         << "which can be only used in combination with the following presets: "
         << "DEFAULT, QUALITY, and DETERMINISTIC"; break;
    case N_LEVEL_HYPERGRAPH_PARTITIONING:
      ss << "The partitioned hypergraph uses the data structures for n-level hypergraph partitioning "
         << "which can be only used in combination with the following preset: "
         << "HIGHEST_QUALITY"; break;
    case LARGE_K_PARTITIONING:
      ss << "The partitioned hypergraph uses the data structures for large k hypergraph partitioning "
         << "which can be only used in combination with the following preset: "
         << "LARGE_K"; break;
    case NULLPTR_PARTITION:
      ss << "The hypergraph holds a nullptr. "
         << "Did you forgot to construct or load a hypergraph?"; break;
  }
  return ss.str();
}

template<typename PartitionedHypergraph, typename Hypergraph>
mt_kahypar_partitioned_hypergraph_t create_partitoned_hypergraph(Hypergraph& hg,
                                                                 const mt_kahypar_partition_id_t num_blocks,
                                                                 const mt_kahypar_partition_id_t* partition) {
  PartitionedHypergraph partitioned_hg(num_blocks, hg, parallel_tag_t { });
  const mt_kahypar::HypernodeID num_nodes = hg.initialNumNodes();
  tbb::parallel_for(ID(0), num_nodes, [&](const mt_kahypar::HypernodeID& hn) {
    partitioned_hg.setOnlyNodePart(hn, partition[hn]);
  });
  partitioned_hg.initializePartition();
  return mt_kahypar_partitioned_hypergraph_t { reinterpret_cast<mt_kahypar_partitioned_hypergraph_s*>(
    new PartitionedHypergraph(std::move(partitioned_hg))), PartitionedHypergraph::TYPE };
}

template<typename PartitionedHypergraph>
void get_partition(const PartitionedHypergraph& partitioned_hg,
                   mt_kahypar_partition_id_t* partition) {
  ASSERT(partition != nullptr);
  partitioned_hg.doParallelForAllNodes([&](const HypernodeID& hn) {
    partition[hn] = partitioned_hg.partID(hn);
  });
}

template<typename PartitionedHypergraph>
void get_block_weights(const PartitionedHypergraph& partitioned_hg,
                       mt_kahypar_hypernode_weight_t* block_weights) {
  ASSERT(block_weights != nullptr);
  for ( PartitionID i = 0; i < partitioned_hg.k(); ++i ) {
    block_weights[i] = partitioned_hg.partWeight(i);
  }
}

} // namespace lib