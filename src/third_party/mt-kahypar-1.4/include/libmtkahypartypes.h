#ifndef TYPEDEFS_H
#define TYPEDEFS_H

typedef enum {
  STATIC_GRAPH,
  DYNAMIC_GRAPH,
  STATIC_HYPERGRAPH,
  DYNAMIC_HYPERGRAPH,
  NULLPTR_HYPERGRAPH
} mt_kahypar_hypergraph_type_t;

typedef enum {
  MULTILEVEL_GRAPH_PARTITIONING,
  N_LEVEL_GRAPH_PARTITIONING,
  MULTILEVEL_HYPERGRAPH_PARTITIONING,
  N_LEVEL_HYPERGRAPH_PARTITIONING,
  LARGE_K_PARTITIONING,
  NULLPTR_PARTITION
} mt_kahypar_partition_type_t;

struct mt_kahypar_context_s;
typedef struct mt_kahypar_context_s mt_kahypar_context_t;
struct mt_kahypar_target_graph_s;
typedef struct mt_kahypar_target_graph_s mt_kahypar_target_graph_t;

struct mt_kahypar_hypergraph_s;
typedef struct {
  mt_kahypar_hypergraph_s* hypergraph;
  mt_kahypar_hypergraph_type_t type;
} mt_kahypar_hypergraph_t;

typedef struct {
  const mt_kahypar_hypergraph_s* hypergraph;
  mt_kahypar_hypergraph_type_t type;
} mt_kahypar_hypergraph_const_t;

struct mt_kahypar_partitioned_hypergraph_s;
typedef struct {
  mt_kahypar_partitioned_hypergraph_s* partitioned_hg;
  mt_kahypar_partition_type_t type;
} mt_kahypar_partitioned_hypergraph_t;

typedef struct {
  const mt_kahypar_partitioned_hypergraph_s* partitioned_hg;
  mt_kahypar_partition_type_t type;
} mt_kahypar_partitioned_hypergraph_const_t;

typedef unsigned long int mt_kahypar_hypernode_id_t;
typedef unsigned long int mt_kahypar_hyperedge_id_t;
typedef int mt_kahypar_hypernode_weight_t;
typedef int mt_kahypar_hyperedge_weight_t;
typedef int mt_kahypar_partition_id_t;

/**
 * Configurable parameters of the partitioning context.
 */
typedef enum {
  // number of blocks of the partition
  NUM_BLOCKS,
  // imbalance factor
  EPSILON,
  // objective function (either 'cut' or 'km1')
  OBJECTIVE,
  // number of V-cycles
  NUM_VCYCLES,
  // disables or enables logging
  VERBOSE
} mt_kahypar_context_parameter_type_t;

/**
 * Supported objective functions.
 */
typedef enum {
  CUT,
  KM1,
  SOED
} mt_kahypar_objective_t;

/**
 * Preset types for partitioning context.
 */
typedef enum {
  // deterministic partitioning mode (corresponds to Mt-KaHyPar-SDet)
  DETERMINISTIC,
  // partitioning mode for partitioning a (hyper)graph into a large number of blocks
  LARGE_K,
  // computes good partitions very fast (corresponds to Mt-KaHyPar-D)
  DEFAULT,
  // extends default preset with flow-based refinement
  // -> computes high-quality partitions (corresponds to Mt-KaHyPar-D-F)
  QUALITY,
  // n-level code with flow-based refinement
  // => highest quality configuration (corresponds to Mt-KaHyPar-Q-F)
  HIGHEST_QUALITY
} mt_kahypar_preset_type_t;

/**
 * Supported (hyper)graph file formats.
 */
typedef enum {
  // Standard file format for graphs
  METIS,
  // Standard file format for hypergraphs
  HMETIS
} mt_kahypar_file_format_type_t;

#ifndef MT_KAHYPAR_API
#   if __GNUC__ >= 4
#       define MT_KAHYPAR_API __attribute__ ((visibility("default")))
#   else
#       define MT_KAHYPAR_API
#   endif
#endif

#endif // TYPEDEFS_H