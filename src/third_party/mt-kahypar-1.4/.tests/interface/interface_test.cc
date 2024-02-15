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

#include <thread>

#include "tbb/parallel_invoke.h"

#include "libmtkahypar.h"
#include "mt-kahypar/macros.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/io/hypergraph_io.h"

using ::testing::Test;

namespace mt_kahypar {

  static constexpr bool debug = false;

  TEST(MtKaHyPar, ReadHypergraphFile) {
    mt_kahypar_hypergraph_t hypergraph =
      mt_kahypar_read_hypergraph_from_file("test_instances/ibm01.hgr", DEFAULT, HMETIS);

    ASSERT_EQ(12752, mt_kahypar_num_hypernodes(hypergraph));
    ASSERT_EQ(14111, mt_kahypar_num_hyperedges(hypergraph));
    ASSERT_EQ(50566, mt_kahypar_num_pins(hypergraph));
    ASSERT_EQ(12752, mt_kahypar_hypergraph_weight(hypergraph));

    mt_kahypar_free_hypergraph(hypergraph);
  }

  TEST(MtKaHyPar, ReadGraphFile) {
    mt_kahypar_hypergraph_t graph =
      mt_kahypar_read_hypergraph_from_file("test_instances/delaunay_n15.graph", DEFAULT, METIS);

    ASSERT_EQ(32768,  mt_kahypar_num_hypernodes(graph));
    ASSERT_EQ(98274,  mt_kahypar_num_hyperedges(graph));
    ASSERT_EQ(196548, mt_kahypar_num_pins(graph));
    ASSERT_EQ(32768,  mt_kahypar_hypergraph_weight(graph));

    mt_kahypar_free_hypergraph(graph);
  }

  TEST(MtKaHyPar, ConstructUnweightedStaticHypergraph) {
    const mt_kahypar_hypernode_id_t num_vertices = 7;
    const mt_kahypar_hyperedge_id_t num_hyperedges = 4;

    std::unique_ptr<size_t[]> hyperedge_indices = std::make_unique<size_t[]>(5);
    hyperedge_indices[0] = 0; hyperedge_indices[1] = 2; hyperedge_indices[2] = 6;
    hyperedge_indices[3] = 9; hyperedge_indices[4] = 12;

    std::unique_ptr<mt_kahypar_hyperedge_id_t[]> hyperedges = std::make_unique<mt_kahypar_hyperedge_id_t[]>(12);
    hyperedges[0] = 0;  hyperedges[1] = 2;                                        // Hyperedge 0
    hyperedges[2] = 0;  hyperedges[3] = 1; hyperedges[4] = 3;  hyperedges[5] = 4; // Hyperedge 1
    hyperedges[6] = 3;  hyperedges[7] = 4; hyperedges[8] = 6;                     // Hyperedge 2
    hyperedges[9] = 2; hyperedges[10] = 5; hyperedges[11] = 6;                    // Hyperedge 3

    mt_kahypar_hypergraph_t hypergraph = mt_kahypar_create_hypergraph(
      DEFAULT, num_vertices, num_hyperedges, hyperedge_indices.get(),
      hyperedges.get(), nullptr, nullptr);
    ASSERT_EQ(hypergraph.type, STATIC_HYPERGRAPH);

    ASSERT_EQ(7, mt_kahypar_num_hypernodes(hypergraph));
    ASSERT_EQ(4, mt_kahypar_num_hyperedges(hypergraph));
    ASSERT_EQ(12, mt_kahypar_num_pins(hypergraph));
    ASSERT_EQ(7, mt_kahypar_hypergraph_weight(hypergraph));

    mt_kahypar_free_hypergraph(hypergraph);
  }

  TEST(MtKaHyPar, ConstructUnweightedDynamicHypergraph) {
    const mt_kahypar_hypernode_id_t num_vertices = 7;
    const mt_kahypar_hyperedge_id_t num_hyperedges = 4;

    std::unique_ptr<size_t[]> hyperedge_indices = std::make_unique<size_t[]>(5);
    hyperedge_indices[0] = 0; hyperedge_indices[1] = 2; hyperedge_indices[2] = 6;
    hyperedge_indices[3] = 9; hyperedge_indices[4] = 12;

    std::unique_ptr<mt_kahypar_hyperedge_id_t[]> hyperedges = std::make_unique<mt_kahypar_hyperedge_id_t[]>(12);
    hyperedges[0] = 0;  hyperedges[1] = 2;                                        // Hyperedge 0
    hyperedges[2] = 0;  hyperedges[3] = 1; hyperedges[4] = 3;  hyperedges[5] = 4; // Hyperedge 1
    hyperedges[6] = 3;  hyperedges[7] = 4; hyperedges[8] = 6;                     // Hyperedge 2
    hyperedges[9] = 2; hyperedges[10] = 5; hyperedges[11] = 6;                    // Hyperedge 3

    mt_kahypar_hypergraph_t hypergraph = mt_kahypar_create_hypergraph(
      HIGHEST_QUALITY, num_vertices, num_hyperedges, hyperedge_indices.get(),
      hyperedges.get(), nullptr, nullptr);
    ASSERT_EQ(hypergraph.type, DYNAMIC_HYPERGRAPH);

    ASSERT_EQ(7, mt_kahypar_num_hypernodes(hypergraph));
    ASSERT_EQ(4, mt_kahypar_num_hyperedges(hypergraph));
    ASSERT_EQ(12, mt_kahypar_num_pins(hypergraph));
    ASSERT_EQ(7, mt_kahypar_hypergraph_weight(hypergraph));

    mt_kahypar_free_hypergraph(hypergraph);
  }

  TEST(MtKaHyPar, ConstructUnweightedStaticGraph) {
    const mt_kahypar_hypernode_id_t num_vertices = 5;
    const mt_kahypar_hyperedge_id_t num_hyperedges = 6;

    std::unique_ptr<mt_kahypar_hypernode_id_t[]> edges =
      std::make_unique<mt_kahypar_hypernode_id_t[]>(12);
    edges[0] = 0;  edges[1] = 1;
    edges[2] = 0;  edges[3] = 2;
    edges[4] = 1;  edges[5] = 2;
    edges[6] = 1;  edges[7] = 3;
    edges[8] = 2;  edges[9] = 3;
    edges[10] = 3; edges[11] = 4;

    mt_kahypar_hypergraph_t graph = mt_kahypar_create_graph(
      DEFAULT, num_vertices, num_hyperedges, edges.get(), nullptr, nullptr);
    ASSERT_EQ(graph.type, STATIC_GRAPH);

    ASSERT_EQ(5, mt_kahypar_num_hypernodes(graph));
    ASSERT_EQ(6, mt_kahypar_num_hyperedges(graph));
    ASSERT_EQ(5, mt_kahypar_hypergraph_weight(graph));

    mt_kahypar_free_hypergraph(graph);
  }

    TEST(MtKaHyPar, ConstructUnweightedDynamicGraph) {
    const mt_kahypar_hypernode_id_t num_vertices = 5;
    const mt_kahypar_hyperedge_id_t num_hyperedges = 6;

    std::unique_ptr<mt_kahypar_hypernode_id_t[]> edges =
      std::make_unique<mt_kahypar_hypernode_id_t[]>(12);
    edges[0] = 0;  edges[1] = 1;
    edges[2] = 0;  edges[3] = 2;
    edges[4] = 1;  edges[5] = 2;
    edges[6] = 1;  edges[7] = 3;
    edges[8] = 2;  edges[9] = 3;
    edges[10] = 3; edges[11] = 4;

    mt_kahypar_hypergraph_t graph = mt_kahypar_create_graph(
      HIGHEST_QUALITY, num_vertices, num_hyperedges, edges.get(), nullptr, nullptr);
    ASSERT_EQ(graph.type, DYNAMIC_GRAPH);

    ASSERT_EQ(5, mt_kahypar_num_hypernodes(graph));
    ASSERT_EQ(6, mt_kahypar_num_hyperedges(graph));
    ASSERT_EQ(5, mt_kahypar_hypergraph_weight(graph));

    mt_kahypar_free_hypergraph(graph);
  }

  TEST(MtKaHyPar, ConstructHypergraphWithNodeWeights) {
    const mt_kahypar_hypernode_id_t num_vertices = 7;
    const mt_kahypar_hyperedge_id_t num_hyperedges = 4;

    std::unique_ptr<mt_kahypar_hypernode_weight_t[]> vertex_weights =
      std::make_unique<mt_kahypar_hypernode_weight_t[]>(7);
    vertex_weights[0] = 1; vertex_weights[1] = 2; vertex_weights[2] = 3; vertex_weights[3] = 4;
    vertex_weights[4] = 5; vertex_weights[5] = 6; vertex_weights[6] = 7;

    std::unique_ptr<size_t[]> hyperedge_indices = std::make_unique<size_t[]>(5);
    hyperedge_indices[0] = 0; hyperedge_indices[1] = 2; hyperedge_indices[2] = 6;
    hyperedge_indices[3] = 9; hyperedge_indices[4] = 12;

    std::unique_ptr<mt_kahypar_hyperedge_id_t[]> hyperedges = std::make_unique<mt_kahypar_hyperedge_id_t[]>(12);
    hyperedges[0] = 0;  hyperedges[1] = 2;                                        // Hyperedge 0
    hyperedges[2] = 0;  hyperedges[3] = 1; hyperedges[4] = 3;  hyperedges[5] = 4; // Hyperedge 1
    hyperedges[6] = 3;  hyperedges[7] = 4; hyperedges[8] = 6;                     // Hyperedge 2
    hyperedges[9] = 2; hyperedges[10] = 5; hyperedges[11] = 6;                    // Hyperedge 3

    mt_kahypar_hypergraph_t hypergraph = mt_kahypar_create_hypergraph(
      DEFAULT, num_vertices, num_hyperedges, hyperedge_indices.get(),
      hyperedges.get(), nullptr, vertex_weights.get());

    ASSERT_EQ(7, mt_kahypar_num_hypernodes(hypergraph));
    ASSERT_EQ(4, mt_kahypar_num_hyperedges(hypergraph));
    ASSERT_EQ(12, mt_kahypar_num_pins(hypergraph));
    ASSERT_EQ(28, mt_kahypar_hypergraph_weight(hypergraph));

    mt_kahypar_free_hypergraph(hypergraph);
  }

  TEST(MtKaHyPar, ConstructGraphWithNodeWeights) {
    const mt_kahypar_hypernode_id_t num_vertices = 5;
    const mt_kahypar_hyperedge_id_t num_hyperedges = 6;

    std::unique_ptr<mt_kahypar_hypernode_weight_t[]> vertex_weights =
      std::make_unique<mt_kahypar_hypernode_weight_t[]>(7);
    vertex_weights[0] = 1; vertex_weights[1] = 2; vertex_weights[2] = 3;
    vertex_weights[3] = 4; vertex_weights[4] = 5;

    std::unique_ptr<mt_kahypar_hypernode_id_t[]> edges =
      std::make_unique<mt_kahypar_hypernode_id_t[]>(12);
    edges[0] = 0;  edges[1] = 1;
    edges[2] = 0;  edges[3] = 2;
    edges[4] = 1;  edges[5] = 2;
    edges[6] = 1;  edges[7] = 3;
    edges[8] = 2;  edges[9] = 3;
    edges[10] = 3; edges[11] = 4;

    mt_kahypar_hypergraph_t graph = mt_kahypar_create_graph(
      DEFAULT, num_vertices, num_hyperedges, edges.get(), nullptr, vertex_weights.get());

    ASSERT_EQ(5, mt_kahypar_num_hypernodes(graph));
    ASSERT_EQ(6, mt_kahypar_num_hyperedges(graph));
    ASSERT_EQ(15, mt_kahypar_hypergraph_weight(graph));

    mt_kahypar_free_hypergraph(graph);
  }

  TEST(MtKaHyPar, CreatesPartitionedHypergraph) {
    const mt_kahypar_hypernode_id_t num_vertices = 7;
    const mt_kahypar_hyperedge_id_t num_hyperedges = 4;

    std::unique_ptr<size_t[]> hyperedge_indices = std::make_unique<size_t[]>(5);
    hyperedge_indices[0] = 0; hyperedge_indices[1] = 2; hyperedge_indices[2] = 6;
    hyperedge_indices[3] = 9; hyperedge_indices[4] = 12;

    std::unique_ptr<mt_kahypar_hyperedge_id_t[]> hyperedges = std::make_unique<mt_kahypar_hyperedge_id_t[]>(12);
    hyperedges[0] = 0;  hyperedges[1] = 2;                                        // Hyperedge 0
    hyperedges[2] = 0;  hyperedges[3] = 1; hyperedges[4] = 3;  hyperedges[5] = 4; // Hyperedge 1
    hyperedges[6] = 3;  hyperedges[7] = 4; hyperedges[8] = 6;                     // Hyperedge 2
    hyperedges[9] = 2; hyperedges[10] = 5; hyperedges[11] = 6;                    // Hyperedge 3

    mt_kahypar_hypergraph_t hypergraph = mt_kahypar_create_hypergraph(
      DEFAULT, num_vertices, num_hyperedges, hyperedge_indices.get(), hyperedges.get(), nullptr, nullptr);

    std::unique_ptr<mt_kahypar_partition_id_t[]> partition =
      std::make_unique<mt_kahypar_partition_id_t[]>(7);
    partition[0] = 0; partition[1] = 0; partition[2] = 0;
    partition[3] = 1; partition[4] = 1; partition[5] = 1; partition[6] = 1;

    mt_kahypar_partitioned_hypergraph_t partitioned_hg =
      mt_kahypar_create_partitioned_hypergraph(hypergraph, DEFAULT, 2, partition.get());
    ASSERT_EQ(partitioned_hg.type, MULTILEVEL_HYPERGRAPH_PARTITIONING);

    std::unique_ptr<mt_kahypar_partition_id_t[]> actual_partition =
      std::make_unique<mt_kahypar_partition_id_t[]>(7);
    mt_kahypar_get_partition(partitioned_hg, actual_partition.get());

    ASSERT_EQ(2, mt_kahypar_km1(partitioned_hg));
    for ( mt_kahypar_hypernode_id_t hn = 0; hn < 7; ++hn ) {
      ASSERT_EQ(partition[hn], actual_partition[hn]);
    }

    mt_kahypar_free_hypergraph(hypergraph);
    mt_kahypar_free_partitioned_hypergraph(partitioned_hg);
  }

  TEST(MtKaHyPar, CreatesPartitionedGraph) {
    const mt_kahypar_hypernode_id_t num_vertices = 5;
    const mt_kahypar_hyperedge_id_t num_hyperedges = 6;

    std::unique_ptr<mt_kahypar_hypernode_weight_t[]> vertex_weights =
      std::make_unique<mt_kahypar_hypernode_weight_t[]>(7);
    vertex_weights[0] = 1; vertex_weights[1] = 2; vertex_weights[2] = 3;
    vertex_weights[3] = 4; vertex_weights[4] = 5;

    std::unique_ptr<mt_kahypar_hypernode_id_t[]> edges =
      std::make_unique<mt_kahypar_hypernode_id_t[]>(12);
    edges[0] = 0;  edges[1] = 1;
    edges[2] = 0;  edges[3] = 2;
    edges[4] = 1;  edges[5] = 2;
    edges[6] = 1;  edges[7] = 3;
    edges[8] = 2;  edges[9] = 3;
    edges[10] = 3; edges[11] = 4;

    mt_kahypar_hypergraph_t graph = mt_kahypar_create_graph(
      DEFAULT, num_vertices, num_hyperedges, edges.get(), nullptr, vertex_weights.get());

    std::unique_ptr<mt_kahypar_partition_id_t[]> partition =
      std::make_unique<mt_kahypar_partition_id_t[]>(7);
    partition[0] = 0; partition[1] = 0; partition[2] = 1;
    partition[3] = 1; partition[4] = 1;

    mt_kahypar_partitioned_hypergraph_t partitioned_graph =
      mt_kahypar_create_partitioned_hypergraph(graph, DEFAULT, 2, partition.get());
    ASSERT_EQ(partitioned_graph.type, MULTILEVEL_GRAPH_PARTITIONING);

    std::unique_ptr<mt_kahypar_partition_id_t[]> actual_partition =
      std::make_unique<mt_kahypar_partition_id_t[]>(7);
    mt_kahypar_get_partition(partitioned_graph, actual_partition.get());

    ASSERT_EQ(3, mt_kahypar_cut(partitioned_graph));
    for ( mt_kahypar_hypernode_id_t hn = 0; hn < 7; ++hn ) {
      ASSERT_EQ(partition[hn], actual_partition[hn]);
    }

    mt_kahypar_free_hypergraph(graph);
    mt_kahypar_free_partitioned_hypergraph(partitioned_graph);
  }

  TEST(MtKaHyPar, WritesAndLoadsHypergraphPartitionFile) {
    const mt_kahypar_hypernode_id_t num_vertices = 7;
    const mt_kahypar_hyperedge_id_t num_hyperedges = 4;

    std::unique_ptr<size_t[]> hyperedge_indices = std::make_unique<size_t[]>(5);
    hyperedge_indices[0] = 0; hyperedge_indices[1] = 2; hyperedge_indices[2] = 6;
    hyperedge_indices[3] = 9; hyperedge_indices[4] = 12;

    std::unique_ptr<mt_kahypar_hyperedge_id_t[]> hyperedges = std::make_unique<mt_kahypar_hyperedge_id_t[]>(12);
    hyperedges[0] = 0;  hyperedges[1] = 2;                                        // Hyperedge 0
    hyperedges[2] = 0;  hyperedges[3] = 1; hyperedges[4] = 3;  hyperedges[5] = 4; // Hyperedge 1
    hyperedges[6] = 3;  hyperedges[7] = 4; hyperedges[8] = 6;                     // Hyperedge 2
    hyperedges[9] = 2; hyperedges[10] = 5; hyperedges[11] = 6;                    // Hyperedge 3

    mt_kahypar_hypergraph_t hypergraph = mt_kahypar_create_hypergraph(
      DEFAULT, num_vertices, num_hyperedges, hyperedge_indices.get(), hyperedges.get(), nullptr, nullptr);

    std::unique_ptr<mt_kahypar_partition_id_t[]> partition = std::make_unique<mt_kahypar_partition_id_t[]>(7);
    partition[0] = 0; partition[1] = 0; partition[2] = 0;
    partition[3] = 1; partition[4] = 1; partition[5] = 1; partition[6] = 1;

    mt_kahypar_partitioned_hypergraph_t partitioned_hg =
      mt_kahypar_create_partitioned_hypergraph(hypergraph, DEFAULT, 2, partition.get());

    mt_kahypar_write_partition_to_file(partitioned_hg, "tmp.partition");

    mt_kahypar_partitioned_hypergraph_t partitioned_hg_2 =
      mt_kahypar_read_partition_from_file(hypergraph, DEFAULT, 2, "tmp.partition");

    std::unique_ptr<mt_kahypar_partition_id_t[]> actual_partition =
      std::make_unique<mt_kahypar_partition_id_t[]>(7);
    mt_kahypar_get_partition(partitioned_hg_2, actual_partition.get());

    ASSERT_EQ(2, mt_kahypar_km1(partitioned_hg_2));
    for ( mt_kahypar_hypernode_id_t hn = 0; hn < 5; ++hn ) {
      ASSERT_EQ(partition[hn], actual_partition[hn]);
    }

    mt_kahypar_free_hypergraph(hypergraph);
    mt_kahypar_free_partitioned_hypergraph(partitioned_hg);
    mt_kahypar_free_partitioned_hypergraph(partitioned_hg_2);
  }

  TEST(MtKaHyPar, WritesAndLoadsGraphPartitionFile) {
    const mt_kahypar_hypernode_id_t num_vertices = 5;
    const mt_kahypar_hyperedge_id_t num_hyperedges = 6;

    std::unique_ptr<mt_kahypar_hypernode_weight_t[]> vertex_weights =
      std::make_unique<mt_kahypar_hypernode_weight_t[]>(7);
    vertex_weights[0] = 1; vertex_weights[1] = 2; vertex_weights[2] = 3;
    vertex_weights[3] = 4; vertex_weights[4] = 5;

    std::unique_ptr<mt_kahypar_hypernode_id_t[]> edges =
      std::make_unique<mt_kahypar_hypernode_id_t[]>(12);
    edges[0] = 0;  edges[1] = 1;
    edges[2] = 0;  edges[3] = 2;
    edges[4] = 1;  edges[5] = 2;
    edges[6] = 1;  edges[7] = 3;
    edges[8] = 2;  edges[9] = 3;
    edges[10] = 3; edges[11] = 4;

    mt_kahypar_hypergraph_t graph = mt_kahypar_create_graph(
      DEFAULT, num_vertices, num_hyperedges, edges.get(), nullptr, vertex_weights.get());

    std::unique_ptr<mt_kahypar_partition_id_t[]> partition =
      std::make_unique<mt_kahypar_partition_id_t[]>(7);
    partition[0] = 0; partition[1] = 0; partition[2] = 1;
    partition[3] = 1; partition[4] = 1;

    mt_kahypar_partitioned_hypergraph_t partitioned_graph =
      mt_kahypar_create_partitioned_hypergraph(graph, DEFAULT, 2, partition.get());

    mt_kahypar_write_partition_to_file(partitioned_graph, "tmp.partition");

    mt_kahypar_partitioned_hypergraph_t partitioned_graph_2 =
      mt_kahypar_read_partition_from_file(graph, DEFAULT, 2, "tmp.partition");

    std::unique_ptr<mt_kahypar_partition_id_t[]> actual_partition =
      std::make_unique<mt_kahypar_partition_id_t[]>(7);
    mt_kahypar_get_partition(partitioned_graph_2, actual_partition.get());

    ASSERT_EQ(3, mt_kahypar_cut(partitioned_graph_2));
    for ( mt_kahypar_hypernode_id_t hn = 0; hn < 5; ++hn ) {
      ASSERT_EQ(partition[hn], actual_partition[hn]);
    }

    mt_kahypar_free_hypergraph(graph);
    mt_kahypar_free_partitioned_hypergraph(partitioned_graph);
    mt_kahypar_free_partitioned_hypergraph(partitioned_graph_2);
  }

  class APartitioner : public Test {
    private:
      static constexpr bool debug = false;

    public:
      static constexpr char HYPERGRAPH_FILE[] = "test_instances/ibm01.hgr";
      static constexpr char GRAPH_FILE[] = "test_instances/delaunay_n15.graph";
      static constexpr char HYPERGRAPH_FIX_FILE[] = "test_instances/ibm01.k4.p1.fix";
      static constexpr char GRAPH_FIX_FILE[] = "test_instances/delaunay_n15.k4.p1.fix";
      static constexpr char TARGET_GRAPH_FILE[] = "test_instances/target.graph";

      APartitioner() :
        context(nullptr),
        hypergraph(mt_kahypar_hypergraph_t { nullptr, NULLPTR_HYPERGRAPH }),
        partitioned_hg(mt_kahypar_partitioned_hypergraph_t { nullptr, NULLPTR_PARTITION }),
        target_graph(nullptr) {
      mt_kahypar_set_seed(42);
    }

    void Partition(const char* filename,
                   const mt_kahypar_file_format_type_t format,
                   const mt_kahypar_preset_type_t preset,
                   const mt_kahypar_partition_id_t num_blocks,
                   const double epsilon,
                   const mt_kahypar_objective_t objective,
                   const bool verbose = false,
                   const bool add_fixed_vertices = false) {
      SetUpContext(preset, num_blocks, epsilon, objective, verbose);
      Load(filename, preset, format);
      if ( add_fixed_vertices ) addFixedVertices(num_blocks);
      partition(hypergraph, &partitioned_hg, context, num_blocks, epsilon, nullptr);
    }

    void Map(const char* filename,
            const mt_kahypar_file_format_type_t format,
            const mt_kahypar_preset_type_t preset,
            const double epsilon,
            const bool verbose = false) {
      SetUpContext(preset, 8, epsilon, KM1, verbose);
      Load(filename, preset, format);
      partition(hypergraph, &partitioned_hg, context, 8, epsilon, target_graph);
    }

    void PartitionAnotherHypergraph(const char* filename,
                                    const mt_kahypar_file_format_type_t format,
                                    const mt_kahypar_preset_type_t preset,
                                    const mt_kahypar_partition_id_t num_blocks,
                                    const double epsilon,
                                    const mt_kahypar_objective_t objective,
                                    const bool verbose = false) {
      mt_kahypar_context_t* c = mt_kahypar_context_new();
      mt_kahypar_load_preset(c, preset);
      mt_kahypar_set_partitioning_parameters(c, num_blocks, epsilon, objective);
      mt_kahypar_set_context_parameter(c, VERBOSE, ( debug || verbose ) ? "1" : "0");

      mt_kahypar_hypergraph_t hg = mt_kahypar_read_hypergraph_from_file(filename, preset, format);
      partition(hg, nullptr, c, num_blocks, epsilon, nullptr);

      mt_kahypar_free_context(c);
      mt_kahypar_free_hypergraph(hg);
    }

    void ImprovePartition(const mt_kahypar_preset_type_t preset,
                          const size_t num_vcycles,
                          const bool verbose = false) {
      mt_kahypar_load_preset(context, preset);
      mt_kahypar_set_context_parameter(context, VERBOSE, ( debug || verbose ) ? "1" : "0");

      mt_kahypar_hyperedge_weight_t before = mt_kahypar_km1(partitioned_hg);
      mt_kahypar_improve_partition(partitioned_hg, context, num_vcycles);
      mt_kahypar_hyperedge_weight_t after = mt_kahypar_km1(partitioned_hg);
      ASSERT_LE(after, before);
    }

    void ImproveMapping(const mt_kahypar_preset_type_t preset,
                        const size_t num_vcycles,
                        const bool verbose = false) {
      mt_kahypar_load_preset(context, preset);
      mt_kahypar_set_context_parameter(context, VERBOSE, ( debug || verbose ) ? "1" : "0");

      mt_kahypar_hyperedge_weight_t before = mt_kahypar_steiner_tree(partitioned_hg, target_graph);
      mt_kahypar_improve_mapping(partitioned_hg, target_graph, context, num_vcycles);
      mt_kahypar_hyperedge_weight_t after = mt_kahypar_steiner_tree(partitioned_hg, target_graph);
      ASSERT_LE(after, before);
    }

    void verifyFixedVertexAssignment(const char* fixed_vertex_file) {
      std::vector<PartitionID> fixed_vertices;
      io::readPartitionFile(fixed_vertex_file, fixed_vertices);
      vec<PartitionID> partition(mt_kahypar_num_hypernodes(hypergraph), kInvalidPartition);
      mt_kahypar_get_partition(partitioned_hg, partition.data());

      for ( HypernodeID hn = 0; hn < mt_kahypar_num_hypernodes(hypergraph); ++hn ) {
        if ( fixed_vertices[hn] != -1 ) {
          ASSERT_EQ(fixed_vertices[hn], partition[hn]);
        }
      }
    }

    void SetUp()  {
      mt_kahypar_initialize_thread_pool(std::thread::hardware_concurrency(), false);
      context = mt_kahypar_context_new();
      target_graph = mt_kahypar_read_target_graph_from_file(TARGET_GRAPH_FILE);
    }

    void TearDown() {
      mt_kahypar_free_context(context);
      mt_kahypar_free_hypergraph(hypergraph);
      mt_kahypar_free_partitioned_hypergraph(partitioned_hg);
      mt_kahypar_free_target_graph(target_graph);
    }

    mt_kahypar_context_t* context;
    mt_kahypar_hypergraph_t hypergraph;
    mt_kahypar_partitioned_hypergraph_t partitioned_hg;
    mt_kahypar_target_graph_t* target_graph;

   private:
    void Load(const char* filename,
              const mt_kahypar_preset_type_t preset,
              const mt_kahypar_file_format_type_t format) {
      if ( hypergraph.type != NULLPTR_HYPERGRAPH ) {
        mt_kahypar_free_hypergraph(hypergraph);
      }
      hypergraph = mt_kahypar_read_hypergraph_from_file(filename, preset, format);
    }

    void addFixedVertices(const mt_kahypar_partition_id_t num_blocks ) {
      if ( hypergraph.type == STATIC_HYPERGRAPH ||
            hypergraph.type == DYNAMIC_HYPERGRAPH ) {
        mt_kahypar_add_fixed_vertices_from_file(hypergraph,
          HYPERGRAPH_FIX_FILE, num_blocks);
      } else if ( hypergraph.type == STATIC_GRAPH ||
                  hypergraph.type == DYNAMIC_GRAPH ) {
        mt_kahypar_add_fixed_vertices_from_file(hypergraph,
          GRAPH_FIX_FILE, num_blocks);
      }
    }

    void SetUpContext(const mt_kahypar_preset_type_t preset,
                      const mt_kahypar_partition_id_t num_blocks,
                      const double epsilon,
                      const mt_kahypar_objective_t objective,
                      const bool verbose = false) {
      mt_kahypar_load_preset(context, preset);
      mt_kahypar_set_partitioning_parameters(context, num_blocks, epsilon, objective);
      mt_kahypar_set_context_parameter(context, VERBOSE, ( debug || verbose ) ? "1" : "0");
    }

    void partition(mt_kahypar_hypergraph_t hg,
                   mt_kahypar_partitioned_hypergraph_t* phg,
                   mt_kahypar_context_t * c,
                   const mt_kahypar_partition_id_t num_blocks,
                   const double epsilon,
                   mt_kahypar_target_graph_t* target_graph) {
      mt_kahypar_partitioned_hypergraph_t p_hg { nullptr, NULLPTR_PARTITION };
      if ( target_graph ) {
        p_hg = mt_kahypar_map(hg, target_graph, c);
      } else {
        p_hg = mt_kahypar_partition(hg, c);
      }

      double imbalance = mt_kahypar_imbalance(p_hg, c);
      mt_kahypar_hyperedge_weight_t km1 = mt_kahypar_km1(p_hg);
      if ( debug ) {
        LOG << " imbalance =" << imbalance << "\n"
            << "cut =" << mt_kahypar_cut(p_hg) << "\n"
            << "km1 =" << km1 << "\n"
            << "soed =" << mt_kahypar_soed(p_hg) << "\n"
            << (target_graph ? "steiner_tree = " + std::to_string(mt_kahypar_steiner_tree(p_hg, target_graph)) : "");

      }
      ASSERT_LE(imbalance, epsilon);

      // Verify Partition IDs
      std::unique_ptr<mt_kahypar_partition_id_t[]> partition =
        std::make_unique<mt_kahypar_partition_id_t[]>(mt_kahypar_num_hypernodes(hg));
      mt_kahypar_get_partition(p_hg, partition.get());
      std::vector<mt_kahypar_hypernode_weight_t> expected_block_weights(num_blocks);
      for ( mt_kahypar_hypernode_id_t hn = 0; hn < mt_kahypar_num_hypernodes(hg); ++hn ) {
        ASSERT_GE(partition[hn], 0);
        ASSERT_LT(partition[hn], num_blocks);
        ++expected_block_weights[partition[hn]];
      }

      // Verify Block Weights
      std::unique_ptr<mt_kahypar_hypernode_weight_t[]> block_weights =
        std::make_unique<mt_kahypar_hypernode_weight_t[]>(num_blocks);
      mt_kahypar_get_block_weights(p_hg, block_weights.get());
      for ( mt_kahypar_partition_id_t i = 0; i < num_blocks; ++i ) {
        EXPECT_EQ(expected_block_weights[i], block_weights[i]);
      }

      if ( phg ) {
        *phg = p_hg;
      } else {
        mt_kahypar_free_partitioned_hypergraph(p_hg);
      }
    }
  };

  TEST_F(APartitioner, PartitionsAHypergraphInTwoBlocksWithDefaultPresetKm1) {
    Partition(HYPERGRAPH_FILE, HMETIS, DEFAULT, 2, 0.03, KM1, false);
  }

  TEST_F(APartitioner, PartitionsAHypergraphInTwoBlocksWithDefaultPresetSoed) {
    Partition(HYPERGRAPH_FILE, HMETIS, DEFAULT, 2, 0.03, SOED, false);
  }

  TEST_F(APartitioner, PartitionsAGraphInTwoBlocksWithDefaultPreset) {
    Partition(GRAPH_FILE, METIS, DEFAULT, 2, 0.03, CUT, false);
  }

  TEST_F(APartitioner, PartitionsAHypergraphInFourBlocksWithDefaultPresetKm1) {
    Partition(HYPERGRAPH_FILE, HMETIS, DEFAULT, 4, 0.03, KM1, false);
  }

  TEST_F(APartitioner, PartitionsAHypergraphInFourBlocksWithDefaultPresetSoed) {
    Partition(HYPERGRAPH_FILE, HMETIS, DEFAULT, 4, 0.03, SOED, false);
  }

  TEST_F(APartitioner, PartitionsAGraphInFourBlocksWithDefaultPreset) {
    Partition(GRAPH_FILE, METIS, DEFAULT, 4, 0.03, CUT, false);
  }

  TEST_F(APartitioner, PartitionsAHypergraphInTwoBlocksWithQualityPreset) {
    Partition(HYPERGRAPH_FILE, HMETIS, QUALITY, 2, 0.03, KM1, false);
  }

  TEST_F(APartitioner, PartitionsAGraphInTwoBlocksWithQualityPreset) {
    Partition(GRAPH_FILE, METIS, QUALITY, 2, 0.03, CUT, false);
  }

  TEST_F(APartitioner, PartitionsAHypergraphInFourBlocksWithQualityPreset) {
    Partition(HYPERGRAPH_FILE, HMETIS, QUALITY, 4, 0.03, KM1, false);
  }

  TEST_F(APartitioner, PartitionsAGraphInFourBlocksWithQualityPreset) {
    Partition(GRAPH_FILE, METIS, QUALITY, 4, 0.03, CUT, false);
  }

  TEST_F(APartitioner, PartitionsAHypergraphInTwoBlocksWithDeterministicPreset) {
    Partition(HYPERGRAPH_FILE, HMETIS, DETERMINISTIC, 2, 0.03, KM1, false);
  }

  TEST_F(APartitioner, PartitionsAGraphInTwoBlocksWithDeterministicPreset) {
    Partition(GRAPH_FILE, METIS, DETERMINISTIC, 2, 0.03, CUT, false);
  }

  TEST_F(APartitioner, PartitionsAHypergraphInFourBlocksWithDeterministicPreset) {
    Partition(HYPERGRAPH_FILE, HMETIS, DETERMINISTIC, 4, 0.03, KM1, false);
  }

  TEST_F(APartitioner, PartitionsAGraphInFourBlocksWithDeterministicPreset) {
    Partition(GRAPH_FILE, METIS, DETERMINISTIC, 4, 0.03, CUT, false);
  }

  TEST_F(APartitioner, PartitionsAHypergraphInTwoBlocksWithLargeKPreset) {
    Partition(HYPERGRAPH_FILE, HMETIS, LARGE_K, 2, 0.03, KM1, false);
  }

  TEST_F(APartitioner, PartitionsAGraphInTwoBlocksWithLargeKPreset) {
    Partition(GRAPH_FILE, METIS, LARGE_K, 2, 0.03, CUT, false);
  }

  TEST_F(APartitioner, PartitionsAHypergraphInFourBlocksWithLargeKPreset) {
    Partition(HYPERGRAPH_FILE, HMETIS, LARGE_K, 4, 0.03, KM1, false);
  }

  TEST_F(APartitioner, PartitionsAGraphInFourBlocksWithLargeKPreset) {
    Partition(GRAPH_FILE, METIS, LARGE_K, 4, 0.03, CUT, false);
  }

  TEST_F(APartitioner, PartitionsAHypergraphInTwoBlocksWithHighestQualityPreset) {
    Partition(HYPERGRAPH_FILE, HMETIS, HIGHEST_QUALITY, 2, 0.03, KM1, false);
  }

  TEST_F(APartitioner, PartitionsAGraphInTwoBlocksWithHighestQualityPreset) {
    Partition(GRAPH_FILE, METIS, HIGHEST_QUALITY, 2, 0.03, CUT, false);
  }

  TEST_F(APartitioner, PartitionsAHypergraphInFourBlocksWithHighestQualityPreset) {
    Partition(HYPERGRAPH_FILE, HMETIS, HIGHEST_QUALITY, 4, 0.03, KM1, false);
  }

  TEST_F(APartitioner, PartitionsAGraphInFourBlocksWithHighestQualityPreset) {
    Partition(GRAPH_FILE, METIS, HIGHEST_QUALITY, 4, 0.03, CUT, false);
  }

  TEST_F(APartitioner, CanPartitionTwoHypergraphsSimultanously) {
    tbb::parallel_invoke([&]() {
      PartitionAnotherHypergraph(HYPERGRAPH_FILE, HMETIS, DETERMINISTIC, 4, 0.03, KM1, false);
    }, [&] {
      PartitionAnotherHypergraph(GRAPH_FILE, METIS, DEFAULT, 8, 0.03, CUT, false);
    });
  }

  TEST_F(APartitioner, CanPartitionFourHypergraphsSimultanously) {
    tbb::parallel_invoke([&]() {
      PartitionAnotherHypergraph(HYPERGRAPH_FILE, HMETIS, DETERMINISTIC, 4, 0.03, KM1, false);
    }, [&] {
      PartitionAnotherHypergraph(GRAPH_FILE, METIS, DEFAULT, 8, 0.03, CUT, false);
    }, [&]() {
      PartitionAnotherHypergraph(HYPERGRAPH_FILE, HMETIS, DEFAULT, 4, 0.03, KM1, false);
    }, [&] {
      PartitionAnotherHypergraph(GRAPH_FILE, METIS, QUALITY, 4, 0.03, CUT, false);
    });
  }

  TEST_F(APartitioner, ChecksIfDeterministicPresetProducesSameResultsForHypergraphs) {
    Partition(HYPERGRAPH_FILE, HMETIS, DETERMINISTIC, 8, 0.03, KM1, false);
    const double objective_1 = mt_kahypar_km1(partitioned_hg);
    Partition(HYPERGRAPH_FILE, HMETIS, DETERMINISTIC, 8, 0.03, KM1, false);
    const double objective_2 = mt_kahypar_km1(partitioned_hg);
    Partition(HYPERGRAPH_FILE, HMETIS, DETERMINISTIC, 8, 0.03, KM1, false);
    const double objective_3 = mt_kahypar_km1(partitioned_hg);
    ASSERT_EQ(objective_1, objective_2);
    ASSERT_EQ(objective_1, objective_3);
  }

  TEST_F(APartitioner, ChecksIfDeterministicPresetProducesSameResultsForGraphs) {
    Partition(GRAPH_FILE, METIS, DETERMINISTIC, 8, 0.03, CUT, false);
    const double objective_1 = mt_kahypar_cut(partitioned_hg);
    Partition(GRAPH_FILE, METIS, DETERMINISTIC, 8, 0.03, CUT, false);
    const double objective_2 = mt_kahypar_cut(partitioned_hg);
    Partition(GRAPH_FILE, METIS, DETERMINISTIC, 8, 0.03, CUT, false);
    const double objective_3 = mt_kahypar_cut(partitioned_hg);
    ASSERT_EQ(objective_1, objective_2);
    ASSERT_EQ(objective_1, objective_3);
  }

  TEST_F(APartitioner, ImprovesHypergraphPartitionWithOneVCycle) {
    Partition(HYPERGRAPH_FILE, HMETIS, DEFAULT, 4, 0.03, KM1, false);
    ImprovePartition(DEFAULT, 1, false);
  }

  TEST_F(APartitioner, ImprovesGraphPartitionWithOneVCycle) {
    Partition(GRAPH_FILE, METIS, DEFAULT, 4, 0.03, CUT, false);
    ImprovePartition(DEFAULT, 1, false);
  }

  TEST_F(APartitioner, ImprovesHypergraphPartitionWithOneVCycleAndDifferentPresetType) {
    Partition(HYPERGRAPH_FILE, HMETIS, DEFAULT, 4, 0.03, KM1, false);
    ImprovePartition(QUALITY, 1, false);
  }

  TEST_F(APartitioner, ImprovesGraphPartitionWithOneVCycleAndDifferentPresetType) {
    Partition(GRAPH_FILE, METIS, DEFAULT, 4, 0.03, CUT, false);
    ImprovePartition(QUALITY, 1, false);
  }

  TEST_F(APartitioner, ImprovesHypergraphPartitionWithThreeVCycles) {
    Partition(HYPERGRAPH_FILE, HMETIS, DEFAULT, 4, 0.03, KM1, false);
    ImprovePartition(DEFAULT, 3, false);
  }

  TEST_F(APartitioner, ImprovesGraphPartitionWithThreeVCycles) {
    Partition(GRAPH_FILE, METIS, DEFAULT, 4, 0.03, CUT, false);
    ImprovePartition(DEFAULT, 3, false);
  }

  TEST_F(APartitioner, PartitionsHypergraphWithIndividualBlockWeights) {
    // Setup Individual Block Weights
    std::unique_ptr<mt_kahypar_hypernode_weight_t[]> block_weights =
      std::make_unique<mt_kahypar_hypernode_weight_t[]>(4);
    block_weights[0] = 2131; block_weights[1] = 1213;
    block_weights[2] = 7287; block_weights[3] = 2501;
    mt_kahypar_set_individual_target_block_weights(context, 4, block_weights.get());

    Partition(HYPERGRAPH_FILE, HMETIS, DEFAULT, 4, 0.03, KM1, false);

    // Verify Block Weights
    std::unique_ptr<mt_kahypar_hypernode_weight_t[]> actual_block_weights =
      std::make_unique<mt_kahypar_hypernode_weight_t[]>(4);
    mt_kahypar_get_block_weights(partitioned_hg, actual_block_weights.get());
    for ( mt_kahypar_partition_id_t i = 0; i < 4; ++i ) {
      ASSERT_LE(actual_block_weights[i], block_weights[i]);
    }
  }

  TEST_F(APartitioner, PartitionsGraphWithIndividualBlockWeights) {
    // Setup Individual Block Weights
    std::unique_ptr<mt_kahypar_hypernode_weight_t[]> block_weights =
      std::make_unique<mt_kahypar_hypernode_weight_t[]>(4);
    block_weights[0] = 11201; block_weights[1] = 4384;
    block_weights[2] = 14174; block_weights[3] = 3989;
    mt_kahypar_set_individual_target_block_weights(context, 4, block_weights.get());

    Partition(GRAPH_FILE, METIS, DEFAULT, 4, 0.03, CUT, false);

    // Verify Block Weights
    std::unique_ptr<mt_kahypar_hypernode_weight_t[]> actual_block_weights =
      std::make_unique<mt_kahypar_hypernode_weight_t[]>(4);
    mt_kahypar_get_block_weights(partitioned_hg, actual_block_weights.get());
    for ( mt_kahypar_partition_id_t i = 0; i < 4; ++i ) {
      ASSERT_LE(actual_block_weights[i], block_weights[i]);
    }
  }

  TEST_F(APartitioner, PartitionsHypergraphWithIndividualBlockWeightsAndVCycle) {
    // Setup Individual Block Weights
    std::unique_ptr<mt_kahypar_hypernode_weight_t[]> block_weights =
      std::make_unique<mt_kahypar_hypernode_weight_t[]>(4);
    block_weights[0] = 2131; block_weights[1] = 1213;
    block_weights[2] = 7287; block_weights[3] = 2501;
    mt_kahypar_set_individual_target_block_weights(context, 4, block_weights.get());

    Partition(HYPERGRAPH_FILE, HMETIS, DEFAULT, 4, 0.03, KM1, false);
    ImprovePartition(DEFAULT, 1, false);

    // Verify Block Weights
    std::unique_ptr<mt_kahypar_hypernode_weight_t[]> actual_block_weights =
      std::make_unique<mt_kahypar_hypernode_weight_t[]>(4);
    mt_kahypar_get_block_weights(partitioned_hg, actual_block_weights.get());
    for ( mt_kahypar_partition_id_t i = 0; i < 4; ++i ) {
      ASSERT_LE(actual_block_weights[i], block_weights[i]);
    }
  }

  TEST_F(APartitioner, PartitionsGraphWithIndividualBlockWeightsAndVCycle) {
    // Setup Individual Block Weights
    std::unique_ptr<mt_kahypar_hypernode_weight_t[]> block_weights =
      std::make_unique<mt_kahypar_hypernode_weight_t[]>(4);
    block_weights[0] = 11201; block_weights[1] = 4384;
    block_weights[2] = 14174; block_weights[3] = 3989;
    mt_kahypar_set_individual_target_block_weights(context, 4, block_weights.get());

    Partition(GRAPH_FILE, METIS, DEFAULT, 4, 0.03, CUT, false);
    ImprovePartition(DEFAULT, 1, false);

    // Verify Block Weights
    std::unique_ptr<mt_kahypar_hypernode_weight_t[]> actual_block_weights =
      std::make_unique<mt_kahypar_hypernode_weight_t[]>(4);
    mt_kahypar_get_block_weights(partitioned_hg, actual_block_weights.get());
    for ( mt_kahypar_partition_id_t i = 0; i < 4; ++i ) {
      ASSERT_LE(actual_block_weights[i], block_weights[i]);
    }
  }

  TEST(MtKaHyPar, CanSetContextParameter) {
    mt_kahypar_context_t* context = mt_kahypar_context_new();
    ASSERT_EQ(0, mt_kahypar_set_context_parameter(context, NUM_BLOCKS, "4"));
    ASSERT_EQ(0, mt_kahypar_set_context_parameter(context, EPSILON, "0.03"));
    ASSERT_EQ(0, mt_kahypar_set_context_parameter(context, OBJECTIVE, "km1"));
    ASSERT_EQ(0, mt_kahypar_set_context_parameter(context, NUM_VCYCLES, "3"));
    ASSERT_EQ(0, mt_kahypar_set_context_parameter(context, VERBOSE, "1"));


    Context& c = *reinterpret_cast<Context*>(context);
    ASSERT_EQ(4, c.partition.k);
    ASSERT_EQ(0.03, c.partition.epsilon);
    ASSERT_EQ(Objective::km1, c.partition.objective);
    ASSERT_EQ(3, c.partition.num_vcycles);
    ASSERT_TRUE(c.partition.verbose_output);

    mt_kahypar_free_context(context);
  }

  TEST_F(APartitioner, MapsAHypergraphOntoATargetGraphWithDefaultPreset) {
    Map(HYPERGRAPH_FILE, HMETIS, DEFAULT, 0.03, false);
  }

  TEST_F(APartitioner, MapsAHypergraphOntoATargetGraphWithQualityPreset) {
    Map(HYPERGRAPH_FILE, HMETIS, QUALITY, 0.03, false);
  }

  TEST_F(APartitioner, MapsAHypergraphOntoATargetGraphWithHighestQualityPreset) {
    Map(HYPERGRAPH_FILE, HMETIS, HIGHEST_QUALITY, 0.03, false);
  }

  TEST_F(APartitioner, MapsAGraphOntoATargetGraphWithDefaultPreset) {
    Map(GRAPH_FILE, METIS, DEFAULT, 0.03, false);
  }

  TEST_F(APartitioner, MapsAGraphOntoATargetGraphWithQualityPreset) {
    Map(GRAPH_FILE, METIS, QUALITY, 0.03, false);
  }

  TEST_F(APartitioner, MapsAGraphOntoATargetGraphWithHighestQualityPreset) {
    Map(GRAPH_FILE, METIS, HIGHEST_QUALITY, 0.03, false);
  }

  TEST_F(APartitioner, ImprovesHypergraphMappingWithOneVCycles) {
    Map(HYPERGRAPH_FILE, HMETIS, DEFAULT, 0.03, false);
    ImproveMapping(DEFAULT, 1, false);
  }

  TEST_F(APartitioner, ImprovesGraphMappingWithOneVCycles) {
    Map(GRAPH_FILE, METIS, DEFAULT, 0.03, false);
    ImproveMapping(DEFAULT, 1, false);
  }

  TEST_F(APartitioner, ImprovesHypergraphMappingWithOneVCyclesWithQualityPreset) {
    Map(HYPERGRAPH_FILE, HMETIS, DEFAULT, 0.03, false);
    ImproveMapping(QUALITY, 1, false);
  }

  TEST_F(APartitioner, ImprovesHypergraphMappingGeneratedByOptimizingKm1Metric) {
    Partition(HYPERGRAPH_FILE, HMETIS, DEFAULT, 8, 0.03, KM1, false);
    ImproveMapping(DEFAULT, 1, false);
  }

  TEST_F(APartitioner, PartitionsAHypergraphWithFixedVerticesAndDefaultPreset) {
    Partition(HYPERGRAPH_FILE, HMETIS, DEFAULT, 4, 0.03, KM1, false, true /* add fixed vertices */);
    verifyFixedVertexAssignment(HYPERGRAPH_FIX_FILE);
  }

  TEST_F(APartitioner, PartitionsAHypergraphWithFixedVerticesAndQualityPreset) {
    Partition(HYPERGRAPH_FILE, HMETIS, QUALITY, 4, 0.03, KM1, false, true /* add fixed vertices */);
    verifyFixedVertexAssignment(HYPERGRAPH_FIX_FILE);
  }

  TEST_F(APartitioner, PartitionsAHypergraphWithFixedVerticesAndHighestQualityPreset) {
    Partition(HYPERGRAPH_FILE, HMETIS, HIGHEST_QUALITY, 4, 0.03, KM1, false, true /* add fixed vertices */);
    verifyFixedVertexAssignment(HYPERGRAPH_FIX_FILE);
  }

  TEST_F(APartitioner, PartitionsAGraphWithFixedVerticesAndDefaultPreset) {
    Partition(GRAPH_FILE, METIS, DEFAULT, 4, 0.03, CUT, false, true /* add fixed vertices */);
    verifyFixedVertexAssignment(GRAPH_FIX_FILE);
  }

  TEST_F(APartitioner, PartitionsGraphWithFixedVerticesAndQualityPreset) {
    Partition(GRAPH_FILE, METIS, QUALITY, 4, 0.03, CUT, false, true /* add fixed vertices */);
    verifyFixedVertexAssignment(GRAPH_FIX_FILE);
  }

  TEST_F(APartitioner, PartitionsAGraphWithFixedVerticesAndHighestQualityPreset) {
    Partition(GRAPH_FILE, METIS, HIGHEST_QUALITY, 4, 0.03, CUT, false, true /* add fixed vertices */);
    verifyFixedVertexAssignment(GRAPH_FIX_FILE);
  }

  TEST_F(APartitioner, ImprovesPartitionWithFixedVertices) {
    Partition(HYPERGRAPH_FILE, HMETIS, DEFAULT, 4, 0.03, KM1, false, true /* add fixed vertices */);
    ImprovePartition(QUALITY, 1, false);
    verifyFixedVertexAssignment(HYPERGRAPH_FIX_FILE);
  }

  TEST_F(APartitioner, PartitionsManyHypergraphsInParallel) {
    std::atomic<size_t> cnt(0);
    size_t max_runs = 100;
    tbb::parallel_for(0U, std::thread::hardware_concurrency(), [&](const int id) {
      while ( cnt.load(std::memory_order_relaxed) < max_runs ) {
        ++cnt;
        PartitionAnotherHypergraph("test_instances/test_instance.hgr", HMETIS, DEFAULT, 4, 0.03, KM1, false);
      }
    });
  }
}