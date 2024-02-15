/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2020 Lars Gottesbüren <lars.gottesbueren@kit.edu>
 * Copyright (C) 2020 Tobias Heuer <tobias.heuer@kit.edu>
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

#include "include/libmtkahypar.h"
#include "include/libmtkahypartypes.h"
#include "include/helper_functions.h"

#include "tbb/parallel_for.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/partitioner_facade.h"
#include "mt-kahypar/partition/metrics.h"
#include "mt-kahypar/partition/conversion.h"
#include "mt-kahypar/partition/mapping/target_graph.h"
#include "mt-kahypar/parallel/tbb_initializer.h"
#include "mt-kahypar/parallel/stl/scalable_vector.h"
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/io/hypergraph_io.h"
#include "mt-kahypar/io/command_line_options.h"
#include "mt-kahypar/macros.h"
#include "mt-kahypar/utils/cast.h"
#include "mt-kahypar/utils/delete.h"

using namespace mt_kahypar;

namespace {
  using StaticHypergraphFactory = typename ds::StaticHypergraph::Factory;
  using DynamicHypergraphFactory = typename ds::DynamicHypergraph::Factory;
  using StaticGraphFactory = typename ds::StaticGraph::Factory;
  using DynamicGraphFactory = typename ds::DynamicGraph::Factory;

  using StaticPartitionedHypergraph = typename StaticHypergraphTypeTraits::PartitionedHypergraph;
  using DynamicPartitionedHypergraph = typename DynamicHypergraphTypeTraits::PartitionedHypergraph;
  using SparsePartitionedHypergraph = typename LargeKHypergraphTypeTraits::PartitionedHypergraph;
  using StaticPartitionedGraph = typename StaticGraphTypeTraits::PartitionedHypergraph;
  using DynamicPartitionedGraph = typename DynamicGraphTypeTraits::PartitionedHypergraph;

  PresetType to_preset_type(mt_kahypar_preset_type_t preset) {
    switch ( preset ) {
      case DETERMINISTIC: return PresetType::deterministic;
      case LARGE_K: return PresetType::large_k;
      case DEFAULT: return PresetType::default_preset;
      case QUALITY: return PresetType::quality;
      case HIGHEST_QUALITY: return PresetType::highest_quality;
    }
    return PresetType::UNDEFINED;
  }

}


mt_kahypar_context_t* mt_kahypar_context_new() {
  return reinterpret_cast<mt_kahypar_context_t*>(new Context(false));
}

void mt_kahypar_free_context(mt_kahypar_context_t* context) {
  if (context == nullptr) {
    return;
  }
  delete reinterpret_cast<Context*>(context);
}

void mt_kahypar_configure_context_from_file(mt_kahypar_context_t* kahypar_context,
                                            const char* ini_file_name) {
  try {
    parseIniToContext(*reinterpret_cast<Context*>(kahypar_context), ini_file_name);
  } catch ( std::exception& ex ) {
    LOG << ex.what();
  }
}

void mt_kahypar_load_preset(mt_kahypar_context_t* context,
                            const mt_kahypar_preset_type_t preset) {
  Context& c = *reinterpret_cast<Context*>(context);
  switch(preset) {
    case DETERMINISTIC:
      c.load_deterministic_preset();
      break;
    case LARGE_K:
      c.load_large_k_preset();
      break;
    case DEFAULT:
      c.load_default_preset();
      break;
    case QUALITY:
      c.load_quality_preset();
      break;
    case HIGHEST_QUALITY:
      c.load_highest_quality_preset();
      break;
  }
}

int mt_kahypar_set_context_parameter(mt_kahypar_context_t* context,
                                     const mt_kahypar_context_parameter_type_t type,
                                     const char* value) {
  Context& c = *reinterpret_cast<Context*>(context);
  switch(type) {
    case NUM_BLOCKS:
      c.partition.k = atoi(value);
      if ( c.partition.k > 0 ) return 0; /** success **/
      else return 2; /** integer conversion error **/
    case EPSILON:
      c.partition.epsilon = atof(value);
      return 0;
    case OBJECTIVE:
      {
        std::string objective(value);
        if ( objective == "km1" ) {
          c.partition.objective = Objective::km1;
          return 0;
        } else if ( objective == "cut" ) {
          c.partition.objective = Objective::cut;
          return 0;
        } else if ( objective == "soed" ) {
          c.partition.objective = Objective::soed;
        }
        return 3;
      }
    case NUM_VCYCLES:
      c.partition.num_vcycles = atoi(value);
      return 0;
    case VERBOSE:
      c.partition.verbose_output = atoi(value);
      return 0;
  }
  return 1; /** no valid parameter type **/
}

void mt_kahypar_set_partitioning_parameters(mt_kahypar_context_t* context,
                                            const mt_kahypar_partition_id_t num_blocks,
                                            const double epsilon,
                                            const mt_kahypar_objective_t objective) {
  Context& c = *reinterpret_cast<Context*>(context);
  c.partition.k = num_blocks;
  c.partition.epsilon = epsilon;
  switch ( objective ) {
    case CUT:
      c.partition.objective = Objective::cut; break;
    case KM1:
      c.partition.objective = Objective::km1; break;
    case SOED:
      c.partition.objective = Objective::soed; break;
  }
}

void mt_kahypar_set_seed(const size_t seed) {
  utils::Randomize::instance().setSeed(seed);
}

void mt_kahypar_set_individual_target_block_weights(mt_kahypar_context_t* context,
                                                    const mt_kahypar_partition_id_t num_blocks,
                                                    const mt_kahypar_hypernode_weight_t* block_weights) {
  Context& c = *reinterpret_cast<Context*>(context);
  c.partition.use_individual_part_weights = true;
  c.partition.max_part_weights.assign(num_blocks, 0);
  for ( mt_kahypar_partition_id_t i = 0; i < num_blocks; ++i ) {
    c.partition.max_part_weights[i] = block_weights[i];
  }
}

void mt_kahypar_initialize_thread_pool(const size_t num_threads,
                                       const bool interleaved_allocations) {
  size_t P = num_threads;
  size_t num_available_cpus = HardwareTopology::instance().num_cpus();
  if ( num_available_cpus < num_threads ) {
    WARNING("There are currently only" << num_available_cpus << "cpus available."
      << "Setting number of threads from" << num_threads
      << "to" << num_available_cpus);
    P = num_available_cpus;
  }

  // Initialize TBB task arenas on numa nodes
  TBBInitializer::instance(P);

  if ( interleaved_allocations ) {
    // We set the membind policy to interleaved allocations in order to
    // distribute allocations evenly across NUMA nodes
    hwloc_cpuset_t cpuset = TBBInitializer::instance().used_cpuset();
    parallel::HardwareTopology<>::instance().activate_interleaved_membind_policy(cpuset);
    hwloc_bitmap_free(cpuset);
  }
}

mt_kahypar_hypergraph_t mt_kahypar_read_hypergraph_from_file(const char* file_name,
                                                             const mt_kahypar_preset_type_t preset,
                                                             const mt_kahypar_file_format_type_t file_format) {
  const PresetType config = to_preset_type(preset);
  const InstanceType instance = file_format == HMETIS ? InstanceType::hypergraph : InstanceType::graph;
  const FileFormat format = file_format == HMETIS ? FileFormat::hMetis : FileFormat::Metis;
  const bool stable_construction = preset == DETERMINISTIC ? true : false;
  try {
    return io::readInputFile(file_name, config, instance, format, stable_construction);
  } catch ( std::exception& ex ) {
    LOG << ex.what();
  }
  return mt_kahypar_hypergraph_t { nullptr, NULLPTR_HYPERGRAPH };
}

mt_kahypar_target_graph_t* mt_kahypar_read_target_graph_from_file(const char* file_name) {
  TargetGraph* target_graph = nullptr;
  try {
    ds::StaticGraph graph = io::readInputFile<ds::StaticGraph>(file_name, FileFormat::Metis, true);
    target_graph = new TargetGraph(std::move(graph));
  } catch ( std::exception& ex ) {
    LOG << ex.what();
  }
  return reinterpret_cast<mt_kahypar_target_graph_t*>(target_graph);
}


mt_kahypar_hypergraph_t mt_kahypar_create_hypergraph(const mt_kahypar_preset_type_t preset,
                                                     const mt_kahypar_hypernode_id_t num_vertices,
                                                     const mt_kahypar_hyperedge_id_t num_hyperedges,
                                                     const size_t* hyperedge_indices,
                                                     const mt_kahypar_hyperedge_id_t* hyperedges,
                                                     const mt_kahypar_hyperedge_weight_t* hyperedge_weights,
                                                     const mt_kahypar_hypernode_weight_t* vertex_weights) {
  // Transform adjacence array into adjacence list
  vec<vec<HypernodeID>> edge_vector(num_hyperedges);
  tbb::parallel_for<HyperedgeID>(0, num_hyperedges, [&](const mt_kahypar::HyperedgeID& he) {
    const size_t num_pins = hyperedge_indices[he + 1] - hyperedge_indices[he];
    edge_vector[he].resize(num_pins);
    for ( size_t i = 0; i < num_pins; ++i ) {
      edge_vector[he][i] = hyperedges[hyperedge_indices[he] + i];
    }
  });

  try {
    switch ( preset ) {
      case DETERMINISTIC:
      case LARGE_K:
      case DEFAULT:
      case QUALITY:
        return mt_kahypar_hypergraph_t {
          reinterpret_cast<mt_kahypar_hypergraph_s*>(new ds::StaticHypergraph(
            StaticHypergraphFactory::construct(num_vertices, num_hyperedges,
              edge_vector, hyperedge_weights, vertex_weights))), STATIC_HYPERGRAPH };
      case HIGHEST_QUALITY:
        return mt_kahypar_hypergraph_t {
          reinterpret_cast<mt_kahypar_hypergraph_s*>(new ds::DynamicHypergraph(
            DynamicHypergraphFactory::construct(num_vertices, num_hyperedges,
              edge_vector, hyperedge_weights, vertex_weights))), DYNAMIC_HYPERGRAPH };
    }
  } catch ( std::exception& ex ) {
    LOG << ex.what();
  }
  return mt_kahypar_hypergraph_t { nullptr, NULLPTR_HYPERGRAPH };
}

mt_kahypar_hypergraph_t mt_kahypar_create_graph(const mt_kahypar_preset_type_t preset,
                                                const mt_kahypar_hypernode_id_t num_vertices,
                                                const mt_kahypar_hyperedge_id_t num_edges,
                                                const mt_kahypar_hypernode_id_t* edges,
                                                const mt_kahypar_hyperedge_weight_t* edge_weights,
                                                const mt_kahypar_hypernode_weight_t* vertex_weights) {
  // Transform adjacence array into adjacence list
  vec<std::pair<mt_kahypar::HypernodeID, mt_kahypar::HypernodeID>> edge_vector(num_edges);
  tbb::parallel_for<mt_kahypar::HyperedgeID>(0, num_edges, [&](const mt_kahypar::HyperedgeID& he) {
    edge_vector[he] = std::make_pair(edges[2*he], edges[2*he + 1]);
  });

  try {
    switch ( preset ) {
      case DETERMINISTIC:
      case LARGE_K:
      case DEFAULT:
      case QUALITY:
        return mt_kahypar_hypergraph_t {
          reinterpret_cast<mt_kahypar_hypergraph_s*>(new ds::StaticGraph(
            StaticGraphFactory::construct_from_graph_edges(num_vertices, num_edges,
              edge_vector, edge_weights, vertex_weights))), STATIC_GRAPH };
      case HIGHEST_QUALITY:
        return mt_kahypar_hypergraph_t {
          reinterpret_cast<mt_kahypar_hypergraph_s*>(new ds::DynamicGraph(
            DynamicGraphFactory::construct_from_graph_edges(num_vertices, num_edges,
              edge_vector, edge_weights, vertex_weights))), DYNAMIC_GRAPH };
    }
  } catch ( std::exception& ex ) {
    LOG << ex.what();
  }
  return mt_kahypar_hypergraph_t { nullptr, NULLPTR_HYPERGRAPH };
}

mt_kahypar_target_graph_t* mt_kahypar_create_target_graph(const mt_kahypar_hypernode_id_t num_vertices,
                                                          const mt_kahypar_hyperedge_id_t num_edges,
                                                          const mt_kahypar_hypernode_id_t* edges,
                                                          const mt_kahypar_hyperedge_weight_t* edge_weights) {
  // Transform adjacence array into adjacence list
  vec<std::pair<mt_kahypar::HypernodeID, mt_kahypar::HypernodeID>> edge_vector(num_edges);
  tbb::parallel_for<mt_kahypar::HyperedgeID>(0, num_edges, [&](const mt_kahypar::HyperedgeID& he) {
    edge_vector[he] = std::make_pair(edges[2*he], edges[2*he + 1]);
  });

  TargetGraph* target_graph = nullptr;
  try {
    ds::StaticGraph graph = StaticGraphFactory::construct_from_graph_edges(
      num_vertices, num_edges, edge_vector, edge_weights, nullptr);
    target_graph = new TargetGraph(std::move(graph));
  } catch ( std::exception& ex ) {
    LOG << ex.what();
  }
  return reinterpret_cast<mt_kahypar_target_graph_t*>(target_graph);
}


void mt_kahypar_free_hypergraph(mt_kahypar_hypergraph_t hypergraph) {
  utils::delete_hypergraph(hypergraph);
}

void mt_kahypar_free_target_graph(mt_kahypar_target_graph_t* target_graph) {
  if ( target_graph ) {
    delete reinterpret_cast<TargetGraph*>(target_graph);
  }
}

mt_kahypar_hypernode_id_t mt_kahypar_num_hypernodes(mt_kahypar_hypergraph_t hypergraph) {
  switch ( hypergraph.type ) {
    case STATIC_GRAPH: return utils::cast<ds::StaticGraph>(hypergraph).initialNumNodes();
    case DYNAMIC_GRAPH: return utils::cast<ds::DynamicGraph>(hypergraph).initialNumNodes();
    case STATIC_HYPERGRAPH: return utils::cast<ds::StaticHypergraph>(hypergraph).initialNumNodes();
    case DYNAMIC_HYPERGRAPH: return utils::cast<ds::DynamicHypergraph>(hypergraph).initialNumNodes();
    case NULLPTR_HYPERGRAPH: return 0;
  }
  return 0;
}

mt_kahypar_hyperedge_id_t mt_kahypar_num_hyperedges(mt_kahypar_hypergraph_t hypergraph) {
  switch ( hypergraph.type ) {
    case STATIC_GRAPH: return utils::cast<ds::StaticGraph>(hypergraph).initialNumEdges() / 2;
    case DYNAMIC_GRAPH: return utils::cast<ds::DynamicGraph>(hypergraph).initialNumEdges() / 2;
    case STATIC_HYPERGRAPH: return utils::cast<ds::StaticHypergraph>(hypergraph).initialNumEdges();
    case DYNAMIC_HYPERGRAPH: return utils::cast<ds::DynamicHypergraph>(hypergraph).initialNumEdges();
    case NULLPTR_HYPERGRAPH: return 0;
  }
  return 0;
}

mt_kahypar_hypernode_id_t mt_kahypar_num_pins(mt_kahypar_hypergraph_t hypergraph) {
  switch ( hypergraph.type ) {
    case STATIC_GRAPH: return utils::cast<ds::StaticGraph>(hypergraph).initialNumPins();
    case DYNAMIC_GRAPH: return utils::cast<ds::DynamicGraph>(hypergraph).initialNumPins();
    case STATIC_HYPERGRAPH: return utils::cast<ds::StaticHypergraph>(hypergraph).initialNumPins();
    case DYNAMIC_HYPERGRAPH: return utils::cast<ds::DynamicHypergraph>(hypergraph).initialNumPins();
    case NULLPTR_HYPERGRAPH: return 0;
  }
  return 0;
}

mt_kahypar_hypernode_id_t mt_kahypar_hypergraph_weight(mt_kahypar_hypergraph_t hypergraph) {
  switch ( hypergraph.type ) {
    case STATIC_GRAPH: return utils::cast<ds::StaticGraph>(hypergraph).totalWeight();
    case DYNAMIC_GRAPH: return utils::cast<ds::DynamicGraph>(hypergraph).totalWeight();
    case STATIC_HYPERGRAPH: return utils::cast<ds::StaticHypergraph>(hypergraph).totalWeight();
    case DYNAMIC_HYPERGRAPH: return utils::cast<ds::DynamicHypergraph>(hypergraph).totalWeight();
    case NULLPTR_HYPERGRAPH: return 0;
  }
  return 0;
}

void mt_kahypar_free_partitioned_hypergraph(mt_kahypar_partitioned_hypergraph_t partitioned_hg) {
  utils::delete_partitioned_hypergraph(partitioned_hg);
}

void mt_kahypar_add_fixed_vertices(mt_kahypar_hypergraph_t hypergraph,
                                   mt_kahypar_partition_id_t* fixed_vertices,
                                   mt_kahypar_partition_id_t num_blocks) {
  try {
    io::addFixedVertices(hypergraph, fixed_vertices, num_blocks);
  } catch ( std::exception& ex ) {
    LOG << ex.what();
  }
}

void mt_kahypar_read_fixed_vertices_from_file(const char* file_name,
                                              mt_kahypar_partition_id_t* fixed_vertices) {
  try {
    io::readPartitionFile(file_name, fixed_vertices);
  } catch ( std::exception& ex ) {
    LOG << ex.what();
  }
}


void mt_kahypar_add_fixed_vertices_from_file(mt_kahypar_hypergraph_t hypergraph,
                                             const char* file_name,
                                             mt_kahypar_partition_id_t num_blocks) {
  try {
    io::addFixedVerticesFromFile(hypergraph, file_name, num_blocks);
  } catch ( std::exception& ex ) {
    LOG << ex.what();
  }
}

void mt_kahypar_remove_fixed_vertices(mt_kahypar_hypergraph_t hypergraph) {
  io::removeFixedVertices(hypergraph);
}

bool mt_kahypar_check_compatibility(mt_kahypar_hypergraph_t hypergraph,
                                                   mt_kahypar_preset_type_t preset) {
  return lib::check_compatibility(hypergraph, preset);
}

mt_kahypar_partitioned_hypergraph_t mt_kahypar_partition(mt_kahypar_hypergraph_t hypergraph,
                                                         mt_kahypar_context_t* context) {
  Context& c = *reinterpret_cast<Context*>(context);
  if ( lib::check_if_all_relavant_parameters_are_set(c) ) {
    if ( mt_kahypar_check_compatibility(hypergraph, lib::get_preset_c_type(c.partition.preset_type)) ) {
      c.partition.instance_type = lib::get_instance_type(hypergraph);
      c.partition.partition_type = to_partition_c_type(
        c.partition.preset_type, c.partition.instance_type);
      lib::prepare_context(c);
      c.partition.num_vcycles = 0;
      try {
        return PartitionerFacade::partition(hypergraph, c);
      } catch ( std::exception& ex ) {
        LOG << ex.what();
      }
    } else {
      WARNING(lib::incompatibility_description(hypergraph));
    }
  }
  return mt_kahypar_partitioned_hypergraph_t { nullptr, NULLPTR_PARTITION };
}

mt_kahypar_partitioned_hypergraph_t mt_kahypar_map(mt_kahypar_hypergraph_t hypergraph,
                                                   mt_kahypar_target_graph_t* target_graph,
                                                   mt_kahypar_context_t* context) {
  Context& c = *reinterpret_cast<Context*>(context);
  if ( lib::check_if_all_relavant_parameters_are_set(c) ) {
    if ( mt_kahypar_check_compatibility(hypergraph, lib::get_preset_c_type(c.partition.preset_type)) ) {
      c.partition.instance_type = lib::get_instance_type(hypergraph);
      c.partition.partition_type = to_partition_c_type(
        c.partition.preset_type, c.partition.instance_type);
      lib::prepare_context(c);
      c.partition.num_vcycles = 0;
      c.partition.objective = Objective::steiner_tree;
      TargetGraph* target = reinterpret_cast<TargetGraph*>(target_graph);
      try {
        return PartitionerFacade::partition(hypergraph, c, target);
      } catch ( std::exception& ex ) {
        LOG << ex.what();
      }
    } else {
      WARNING(lib::incompatibility_description(hypergraph));
    }
  }
  return mt_kahypar_partitioned_hypergraph_t { nullptr, NULLPTR_PARTITION };
}

MT_KAHYPAR_API bool mt_kahypar_check_partition_compatibility(mt_kahypar_partitioned_hypergraph_t partitioned_hg,
                                                             mt_kahypar_preset_type_t preset) {
  return lib::check_compatibility(partitioned_hg, preset);
}

void mt_kahypar_improve_partition(mt_kahypar_partitioned_hypergraph_t partitioned_hg,
                                  mt_kahypar_context_t* context,
                                  const size_t num_vcycles) {
  Context& c = *reinterpret_cast<Context*>(context);
  if ( lib::check_if_all_relavant_parameters_are_set(c) ) {
    if ( mt_kahypar_check_partition_compatibility(
          partitioned_hg, lib::get_preset_c_type(c.partition.preset_type)) ) {
      c.partition.instance_type = lib::get_instance_type(partitioned_hg);
      c.partition.partition_type = to_partition_c_type(
        c.partition.preset_type, c.partition.instance_type);
      lib::prepare_context(c);
      c.partition.num_vcycles = num_vcycles;
      try {
        PartitionerFacade::improve(partitioned_hg, c);
      } catch ( std::exception& ex ) {
        LOG << ex.what();
      }
    } else {
      WARNING(lib::incompatibility_description(partitioned_hg));
    }
  }
}

void mt_kahypar_improve_mapping(mt_kahypar_partitioned_hypergraph_t partitioned_hg,
                                               mt_kahypar_target_graph_t* target_graph,
                                               mt_kahypar_context_t* context,
                                               const size_t num_vcycles) {
  Context& c = *reinterpret_cast<Context*>(context);
  if ( lib::check_if_all_relavant_parameters_are_set(c) ) {
    if ( mt_kahypar_check_partition_compatibility(
          partitioned_hg, lib::get_preset_c_type(c.partition.preset_type)) ) {
      c.partition.instance_type = lib::get_instance_type(partitioned_hg);
      c.partition.partition_type = to_partition_c_type(
        c.partition.preset_type, c.partition.instance_type);
      lib::prepare_context(c);
      c.partition.num_vcycles = num_vcycles;
      c.partition.objective = Objective::steiner_tree;
      TargetGraph* target = reinterpret_cast<TargetGraph*>(target_graph);
      try {
        PartitionerFacade::improve(partitioned_hg, c, target);
      } catch ( std::exception& ex ) {
        LOG << ex.what();
      }
    } else {
      WARNING(lib::incompatibility_description(partitioned_hg));
    }
  }
}

mt_kahypar_partitioned_hypergraph_t mt_kahypar_create_partitioned_hypergraph(mt_kahypar_hypergraph_t hypergraph,
                                                                             const mt_kahypar_preset_type_t preset,
                                                                             const mt_kahypar_partition_id_t num_blocks,
                                                                             const mt_kahypar_partition_id_t* partition) {
  if ( hypergraph.type == STATIC_GRAPH || hypergraph.type == DYNAMIC_GRAPH ) {
    switch ( preset ) {
      case LARGE_K:
      case DETERMINISTIC:
      case DEFAULT:
      case QUALITY:
        ASSERT(hypergraph.type == STATIC_GRAPH);
        return lib::create_partitoned_hypergraph<StaticPartitionedGraph>(
          utils::cast<ds::StaticGraph>(hypergraph), num_blocks, partition);
      case HIGHEST_QUALITY:
        ASSERT(hypergraph.type == DYNAMIC_GRAPH);
        return lib::create_partitoned_hypergraph<DynamicPartitionedGraph>(
          utils::cast<ds::DynamicGraph>(hypergraph), num_blocks, partition);
    }
  } else {
    switch ( preset ) {
      case LARGE_K:
        ASSERT(hypergraph.type == STATIC_HYPERGRAPH);
        return lib::create_partitoned_hypergraph<SparsePartitionedHypergraph>(
          utils::cast<ds::StaticHypergraph>(hypergraph), num_blocks, partition);
      case DETERMINISTIC:
      case DEFAULT:
      case QUALITY:
        ASSERT(hypergraph.type == STATIC_HYPERGRAPH);
        return lib::create_partitoned_hypergraph<StaticPartitionedHypergraph>(
          utils::cast<ds::StaticHypergraph>(hypergraph), num_blocks, partition);
      case HIGHEST_QUALITY:
        ASSERT(hypergraph.type == DYNAMIC_HYPERGRAPH);
        return lib::create_partitoned_hypergraph<DynamicPartitionedHypergraph>(
          utils::cast<ds::DynamicHypergraph>(hypergraph), num_blocks, partition);
    }
  }
  return mt_kahypar_partitioned_hypergraph_t { nullptr, NULLPTR_PARTITION };
}

mt_kahypar_partitioned_hypergraph_t mt_kahypar_read_partition_from_file(mt_kahypar_hypergraph_t hypergraph,
                                                                        const mt_kahypar_preset_type_t preset,
                                                                        const mt_kahypar_partition_id_t num_blocks,
                                                                        const char* partition_file) {
  std::vector<PartitionID> partition;
  io::readPartitionFile(partition_file, partition);
  return mt_kahypar_create_partitioned_hypergraph(hypergraph, preset, num_blocks, partition.data());
}

void mt_kahypar_write_partition_to_file(const mt_kahypar_partitioned_hypergraph_t partitioned_hg,
                                        const char* partition_file) {
  switch ( partitioned_hg.type ) {
    case MULTILEVEL_GRAPH_PARTITIONING:
      io::writePartitionFile(utils::cast<StaticPartitionedGraph>(partitioned_hg), partition_file); break;
    case N_LEVEL_GRAPH_PARTITIONING:
      io::writePartitionFile(utils::cast<DynamicPartitionedGraph>(partitioned_hg), partition_file); break;
    case MULTILEVEL_HYPERGRAPH_PARTITIONING:
      io::writePartitionFile(utils::cast<StaticPartitionedHypergraph>(partitioned_hg), partition_file); break;
    case N_LEVEL_HYPERGRAPH_PARTITIONING:
      io::writePartitionFile(utils::cast<DynamicPartitionedHypergraph>(partitioned_hg), partition_file); break;
    case LARGE_K_PARTITIONING:
      io::writePartitionFile(utils::cast<SparsePartitionedHypergraph>(partitioned_hg), partition_file); break;
    case NULLPTR_PARTITION: break;
  }
}

void mt_kahypar_get_partition(const mt_kahypar_partitioned_hypergraph_t partitioned_hg,
                              mt_kahypar_partition_id_t* partition) {
  switch ( partitioned_hg.type ) {
    case MULTILEVEL_GRAPH_PARTITIONING:
      lib::get_partition(utils::cast<StaticPartitionedGraph>(partitioned_hg), partition); break;
    case N_LEVEL_GRAPH_PARTITIONING:
      lib::get_partition(utils::cast<DynamicPartitionedGraph>(partitioned_hg), partition); break;
    case MULTILEVEL_HYPERGRAPH_PARTITIONING:
      lib::get_partition(utils::cast<StaticPartitionedHypergraph>(partitioned_hg), partition); break;
    case N_LEVEL_HYPERGRAPH_PARTITIONING:
      lib::get_partition(utils::cast<DynamicPartitionedHypergraph>(partitioned_hg), partition); break;
    case LARGE_K_PARTITIONING:
      lib::get_partition(utils::cast<SparsePartitionedHypergraph>(partitioned_hg), partition); break;
    case NULLPTR_PARTITION: break;
  }
}

void mt_kahypar_get_block_weights(const mt_kahypar_partitioned_hypergraph_t partitioned_hg,
                                  mt_kahypar_hypernode_weight_t* block_weights) {
  switch ( partitioned_hg.type ) {
    case MULTILEVEL_GRAPH_PARTITIONING:
      lib::get_block_weights(utils::cast<StaticPartitionedGraph>(partitioned_hg), block_weights); break;
    case N_LEVEL_GRAPH_PARTITIONING:
      lib::get_block_weights(utils::cast<DynamicPartitionedGraph>(partitioned_hg), block_weights); break;
    case MULTILEVEL_HYPERGRAPH_PARTITIONING:
      lib::get_block_weights(utils::cast<StaticPartitionedHypergraph>(partitioned_hg), block_weights); break;
    case N_LEVEL_HYPERGRAPH_PARTITIONING:
      lib::get_block_weights(utils::cast<DynamicPartitionedHypergraph>(partitioned_hg), block_weights); break;
    case LARGE_K_PARTITIONING:
      lib::get_block_weights(utils::cast<SparsePartitionedHypergraph>(partitioned_hg), block_weights); break;
    case NULLPTR_PARTITION: break;
  }
}

double mt_kahypar_imbalance(const mt_kahypar_partitioned_hypergraph_t partitioned_hg,
                            const mt_kahypar_context_t* context) {
  const Context& c = *reinterpret_cast<const Context*>(context);
  switch ( partitioned_hg.type ) {
    case MULTILEVEL_GRAPH_PARTITIONING:
      return metrics::imbalance(utils::cast_const<StaticPartitionedGraph>(partitioned_hg), c);
    case N_LEVEL_GRAPH_PARTITIONING:
      return metrics::imbalance(utils::cast_const<DynamicPartitionedGraph>(partitioned_hg), c);
    case MULTILEVEL_HYPERGRAPH_PARTITIONING:
      return metrics::imbalance(utils::cast_const<StaticPartitionedHypergraph>(partitioned_hg), c);
    case N_LEVEL_HYPERGRAPH_PARTITIONING:
      return metrics::imbalance(utils::cast_const<DynamicPartitionedHypergraph>(partitioned_hg), c);
    case LARGE_K_PARTITIONING:
      return metrics::imbalance(utils::cast_const<SparsePartitionedHypergraph>(partitioned_hg), c);
    case NULLPTR_PARTITION: return 0;
  }
  return 0;
}

mt_kahypar_hyperedge_weight_t mt_kahypar_cut(const mt_kahypar_partitioned_hypergraph_t partitioned_hg) {
  switch ( partitioned_hg.type ) {
    case MULTILEVEL_GRAPH_PARTITIONING:
      return metrics::quality(utils::cast<StaticPartitionedGraph>(partitioned_hg), Objective::cut);
    case N_LEVEL_GRAPH_PARTITIONING:
      return metrics::quality(utils::cast<DynamicPartitionedGraph>(partitioned_hg), Objective::cut);
    case MULTILEVEL_HYPERGRAPH_PARTITIONING:
      return metrics::quality(utils::cast<StaticPartitionedHypergraph>(partitioned_hg), Objective::cut);
    case N_LEVEL_HYPERGRAPH_PARTITIONING:
      return metrics::quality(utils::cast<DynamicPartitionedHypergraph>(partitioned_hg), Objective::cut);
    case LARGE_K_PARTITIONING:
      return metrics::quality(utils::cast<SparsePartitionedHypergraph>(partitioned_hg), Objective::cut);
    case NULLPTR_PARTITION: return 0;
  }
  return 0;
}

mt_kahypar_hyperedge_weight_t mt_kahypar_km1(const mt_kahypar_partitioned_hypergraph_t partitioned_hg) {
  switch ( partitioned_hg.type ) {
    case MULTILEVEL_GRAPH_PARTITIONING:
      return metrics::quality(utils::cast<StaticPartitionedGraph>(partitioned_hg), Objective::km1);
    case N_LEVEL_GRAPH_PARTITIONING:
      return metrics::quality(utils::cast<DynamicPartitionedGraph>(partitioned_hg), Objective::km1);
    case MULTILEVEL_HYPERGRAPH_PARTITIONING:
      return metrics::quality(utils::cast<StaticPartitionedHypergraph>(partitioned_hg), Objective::km1);
    case N_LEVEL_HYPERGRAPH_PARTITIONING:
      return metrics::quality(utils::cast<DynamicPartitionedHypergraph>(partitioned_hg), Objective::km1);
    case LARGE_K_PARTITIONING:
      return metrics::quality(utils::cast<SparsePartitionedHypergraph>(partitioned_hg), Objective::km1);
    case NULLPTR_PARTITION: return 0;
  }
  return 0;
}

mt_kahypar_hyperedge_weight_t mt_kahypar_soed(const mt_kahypar_partitioned_hypergraph_t partitioned_hg) {
  switch ( partitioned_hg.type ) {
    case MULTILEVEL_GRAPH_PARTITIONING:
      return metrics::quality(utils::cast<StaticPartitionedGraph>(partitioned_hg), Objective::soed);
    case N_LEVEL_GRAPH_PARTITIONING:
      return metrics::quality(utils::cast<DynamicPartitionedGraph>(partitioned_hg), Objective::soed);
    case MULTILEVEL_HYPERGRAPH_PARTITIONING:
      return metrics::quality(utils::cast<StaticPartitionedHypergraph>(partitioned_hg), Objective::soed);
    case N_LEVEL_HYPERGRAPH_PARTITIONING:
      return metrics::quality(utils::cast<DynamicPartitionedHypergraph>(partitioned_hg), Objective::soed);
    case LARGE_K_PARTITIONING:
      return metrics::quality(utils::cast<SparsePartitionedHypergraph>(partitioned_hg), Objective::soed);
    case NULLPTR_PARTITION: return 0;
  }
  return 0;
}

mt_kahypar_hyperedge_weight_t mt_kahypar_steiner_tree(const mt_kahypar_partitioned_hypergraph_t partitioned_hg,
                                                      mt_kahypar_target_graph_t* target_graph) {
  TargetGraph* target = reinterpret_cast<TargetGraph*>(target_graph);
  if ( !target->isInitialized() ) {
    target->precomputeDistances(4);
  }

  switch ( partitioned_hg.type ) {
    case MULTILEVEL_GRAPH_PARTITIONING:
      {
        StaticPartitionedGraph& phg = utils::cast<StaticPartitionedGraph>(partitioned_hg);
        phg.setTargetGraph(target);
        return metrics::quality(phg, Objective::steiner_tree);
      }
    case N_LEVEL_GRAPH_PARTITIONING:
      {
        DynamicPartitionedGraph& phg = utils::cast<DynamicPartitionedGraph>(partitioned_hg);
        phg.setTargetGraph(target);
        return metrics::quality(phg, Objective::steiner_tree);
      }
    case MULTILEVEL_HYPERGRAPH_PARTITIONING:
      {
        StaticPartitionedHypergraph& phg = utils::cast<StaticPartitionedHypergraph>(partitioned_hg);
        phg.setTargetGraph(target);
        return metrics::quality(phg, Objective::steiner_tree);
      }
    case N_LEVEL_HYPERGRAPH_PARTITIONING:
      {
        DynamicPartitionedHypergraph& phg = utils::cast<DynamicPartitionedHypergraph>(partitioned_hg);
        phg.setTargetGraph(target);
        return metrics::quality(phg, Objective::steiner_tree);
      }
    case LARGE_K_PARTITIONING:
      {
        SparsePartitionedHypergraph& phg = utils::cast<SparsePartitionedHypergraph>(partitioned_hg);
        phg.setTargetGraph(target);
        return metrics::quality(phg, Objective::steiner_tree);
      }
    case NULLPTR_PARTITION: return 0;
  }
  return 0;
}