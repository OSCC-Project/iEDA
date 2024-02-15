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

#include "hypergraph_io.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>
#include <memory>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#elif _WIN32
#include <windows.h>
#include <process.h>
#include <memoryapi.h>
#endif


#include "tbb/parallel_for.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/context_enum_classes.h"
#include "mt-kahypar/utils/timer.h"
#include "mt-kahypar/utils/exception.h"

namespace mt_kahypar::io {

  #ifdef __linux__
  struct FileHandle {
    int fd;
    char* mapped_file;
    size_t length;

    void closeHandle() {
      close(fd);
    }
  };
  #elif _WIN32
  struct FileHandle {
    HANDLE hFile;
    HANDLE hMem;
    char* mapped_file;
    size_t length;

    void closeHandle() {
      CloseHandle(hFile);
      CloseHandle(hMem);
    }
  };
  #endif

  size_t file_size(const std::string& filename) {
    struct stat stat_buf;
    const int res = stat( filename.c_str(), &stat_buf);
    if (res < 0) {
      throw InvalidInputException("Could not open:" + filename);
    }
    return static_cast<size_t>(stat_buf.st_size);
  }

  FileHandle mmap_file(const std::string& filename) {
    FileHandle handle;
    handle.length = file_size(filename);

    #ifdef _WIN32
      PSECURITY_DESCRIPTOR pSD;
      SECURITY_ATTRIBUTES  sa;

      /* create security descriptor (needed for Windows NT) */
      pSD = (PSECURITY_DESCRIPTOR) malloc( SECURITY_DESCRIPTOR_MIN_LENGTH );
      if( pSD == NULL ) {
        throw SystemException("Error while creating security descriptor!");
      }

      InitializeSecurityDescriptor(pSD, SECURITY_DESCRIPTOR_REVISION);
      SetSecurityDescriptorDacl(pSD, TRUE, (PACL) NULL, FALSE);

      sa.nLength = sizeof(sa);
      sa.lpSecurityDescriptor = pSD;
      sa.bInheritHandle = TRUE;

      // open file
      handle.hFile = CreateFile ( filename.c_str(), GENERIC_READ, FILE_SHARE_READ,
        &sa, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

      if (handle.hFile == INVALID_HANDLE_VALUE) {
        free( pSD);
        throw InvalidInputException("Invalid file handle when opening: " + filename);
      }

      // Create file mapping
      handle.hMem = CreateFileMapping( handle.hFile, &sa, PAGE_READONLY, 0, handle.length, NULL);
      free(pSD);
      if (handle.hMem == NULL) {
        throw InvalidInputException("Invalid file mapping when opening: " + filename);
      }

      // map file to memory
      handle.mapped_file = (char*) MapViewOfFile(handle.hMem, FILE_MAP_READ, 0, 0, 0);
      if ( handle.mapped_file == NULL ) {
        throw SystemException("Failed to map file to main memory:" + filename);
      }
    #elif __linux__
      handle.fd = open(filename.c_str(), O_RDONLY);
      if ( handle.fd < -1 ) {
        throw InvalidInputException("Could not open: " + filename);
      }
      handle.mapped_file = (char*) mmap(0, handle.length, PROT_READ, MAP_SHARED, handle.fd, 0);
      if ( handle.mapped_file == MAP_FAILED ) {
        close(handle.fd);
        throw SystemException("Error while mapping file to memory");
      }
    #endif

    return handle;
  }

  void munmap_file(FileHandle& handle) {
    #ifdef _WIN32
    UnmapViewOfFile(handle.mapped_file);
    #elif __linux__
    munmap(handle.mapped_file, handle.length);
    #endif
    handle.closeHandle();
  }


  MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE
  bool is_line_ending(char* mapped_file, size_t pos) {
    return mapped_file[pos] == '\r' || mapped_file[pos] == '\n' || mapped_file[pos] == '\0';
  }

  MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE
  void do_line_ending(char* mapped_file, size_t& pos) {
    ASSERT(is_line_ending(mapped_file, pos));
    if (mapped_file[pos] != '\0') {
      if (mapped_file[pos] == '\r') {     // windows line ending
        ++pos;
        ASSERT(mapped_file[pos] == '\n');
      }
      ++pos;
    }
  }

  MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE
  void goto_next_line(char* mapped_file, size_t& pos, const size_t length) {
    for ( ; ; ++pos ) {
      if ( pos == length || is_line_ending(mapped_file, pos) ) {
        do_line_ending(mapped_file, pos);
        break;
      }
    }
  }

  MT_KAHYPAR_ATTRIBUTE_ALWAYS_INLINE
  int64_t read_number(char* mapped_file, size_t& pos, const size_t length) {
    int64_t number = 0;
    while ( mapped_file[pos] == ' ' ) {
      ++pos;
    }
    for ( ; pos < length; ++pos ) {
      if ( mapped_file[pos] == ' ' || is_line_ending(mapped_file, pos) ) {
        while ( mapped_file[pos] == ' ' ) {
          ++pos;
        }
        break;
      }
      ASSERT(mapped_file[pos] >= '0' && mapped_file[pos] <= '9');
      number = number * 10 + (mapped_file[pos] - '0');
    }
    return number;
  }

  void readHGRHeader(char* mapped_file,
                     size_t& pos,
                     const size_t length,
                     HyperedgeID& num_hyperedges,
                     HypernodeID& num_hypernodes,
                     mt_kahypar::Type& type) {
    // Skip comments
    while ( mapped_file[pos] == '%' ) {
      goto_next_line(mapped_file, pos, length);
    }

    num_hyperedges = read_number(mapped_file, pos, length);
    num_hypernodes = read_number(mapped_file, pos, length);
    if (!is_line_ending(mapped_file, pos)) {
      type = static_cast<mt_kahypar::Type>(read_number(mapped_file, pos, length));
    }
    do_line_ending(mapped_file, pos);
  }

  struct HyperedgeRange {
    const size_t start;
    const size_t end;
    const HyperedgeID start_id;
    const HyperedgeID num_hyperedges;
  };

  inline bool isSinglePinHyperedge(char* mapped_file,
                                          size_t pos,
                                          const size_t length,
                                          const bool has_hyperedge_weights) {
    size_t num_spaces = 0;
    for ( ; pos < length; ++pos ) {
      if (is_line_ending(mapped_file, pos)) {
        break;
      } else if ( mapped_file[pos] == ' ' ) {
        ++num_spaces;
      }

      if ( num_spaces == 2 ) {
        break;
      }
    }
    return has_hyperedge_weights ? num_spaces == 1 : num_spaces == 0;
  }

  struct HyperedgeReadResult {
    HyperedgeReadResult() :
      num_removed_single_pin_hyperedges(0),
      num_duplicated_pins(0),
      num_hes_with_duplicated_pins(0) { }

    size_t num_removed_single_pin_hyperedges;
    size_t num_duplicated_pins;
    size_t num_hes_with_duplicated_pins;
  };

  HyperedgeReadResult readHyperedges(char* mapped_file,
                                     size_t& pos,
                                     const size_t length,
                                     const HyperedgeID num_hyperedges,
                                     const mt_kahypar::Type type,
                                     HyperedgeVector& hyperedges,
                                     vec<HyperedgeWeight>& hyperedges_weight,
                                     const bool remove_single_pin_hes) {
    HyperedgeReadResult res;
    const bool has_hyperedge_weights = type == mt_kahypar::Type::EdgeWeights ||
                                       type == mt_kahypar::Type::EdgeAndNodeWeights ?
                                       true : false;

    vec<HyperedgeRange> hyperedge_ranges;
    tbb::parallel_invoke([&] {
      // Sequential pass over all hyperedges to determine ranges in the
      // input file that are read in parallel.
      size_t current_range_start = pos;
      HyperedgeID current_range_start_id = 0;
      HyperedgeID current_range_num_hyperedges = 0;
      HyperedgeID current_num_hyperedges = 0;
      const HyperedgeID num_hyperedges_per_range = std::max(
              (num_hyperedges / ( 2 * std::thread::hardware_concurrency())), ID(1));
      while ( current_num_hyperedges < num_hyperedges ) {
        // Skip Comments
        ASSERT(pos < length);
        while ( mapped_file[pos] == '%' ) {
          goto_next_line(mapped_file, pos, length);
          ASSERT(pos < length);
        }

        // This check is fine even with windows line endings!
        ASSERT(mapped_file[pos - 1] == '\n');
        if ( !remove_single_pin_hes || !isSinglePinHyperedge(mapped_file, pos, length, has_hyperedge_weights) ) {
          ++current_range_num_hyperedges;
        } else {
          ++res.num_removed_single_pin_hyperedges;
        }
        ++current_num_hyperedges;
        goto_next_line(mapped_file, pos, length);

        // If there are enough hyperedges in the current scanned range
        // we store that range, which will be later processed in parallel
        if ( current_range_num_hyperedges == num_hyperedges_per_range ) {
          hyperedge_ranges.push_back(HyperedgeRange {
                  current_range_start, pos, current_range_start_id, current_range_num_hyperedges});
          current_range_start = pos;
          current_range_start_id += current_range_num_hyperedges;
          current_range_num_hyperedges = 0;
        }
      }
      if ( current_range_num_hyperedges > 0 ) {
        hyperedge_ranges.push_back(HyperedgeRange {
                current_range_start, pos, current_range_start_id, current_range_num_hyperedges});
      }
    }, [&] {
      hyperedges.resize(num_hyperedges);
    }, [&] {
      if ( has_hyperedge_weights ) {
        hyperedges_weight.resize(num_hyperedges);
      }
    });

    const HyperedgeID tmp_num_hyperedges = num_hyperedges - res.num_removed_single_pin_hyperedges;
    hyperedges.resize(tmp_num_hyperedges);
    if ( has_hyperedge_weights ) {
      hyperedges_weight.resize(tmp_num_hyperedges);
    }

    // Process all ranges in parallel and build hyperedge vector
    tbb::parallel_for(UL(0), hyperedge_ranges.size(), [&](const size_t i) {
      HyperedgeRange& range = hyperedge_ranges[i];
      size_t current_pos = range.start;
      const size_t current_end = range.end;
      HyperedgeID current_id = range.start_id;
      const HyperedgeID last_id = current_id + range.num_hyperedges;

      while ( current_id < last_id ) {
        // Skip Comments
        ASSERT(current_pos < current_end);
        while ( mapped_file[current_pos] == '%' ) {
          goto_next_line(mapped_file, current_pos, current_end);
          ASSERT(current_pos < current_end);
        }

        if ( !remove_single_pin_hes || !isSinglePinHyperedge(mapped_file, current_pos, current_end, has_hyperedge_weights) ) {
          ASSERT(current_id < hyperedges.size());
          if ( has_hyperedge_weights ) {
            hyperedges_weight[current_id] = read_number(mapped_file, current_pos, current_end);
          }

          Hyperedge& hyperedge = hyperedges[current_id];
          // Note, a hyperedge line must contain at least one pin
          HypernodeID pin = read_number(mapped_file, current_pos, current_end);
          ASSERT(pin > 0, V(current_id));
          hyperedge.push_back(pin - 1);
          while ( !is_line_ending(mapped_file, current_pos) ) {
            pin = read_number(mapped_file, current_pos, current_end);
            ASSERT(pin > 0, V(current_id));
            hyperedge.push_back(pin - 1);
          }
          do_line_ending(mapped_file, current_pos);

          // Detect duplicated pins
          std::sort(hyperedge.begin(), hyperedge.end());
          size_t j = 1;
          for ( size_t i = 1; i < hyperedge.size(); ++i ) {
            if ( hyperedge[j - 1] != hyperedge[i] ) {
              std::swap(hyperedge[i], hyperedge[j++]);
            }
          }
          if ( j < hyperedge.size() ) {
            // Remove duplicated pins
            __atomic_fetch_add(&res.num_hes_with_duplicated_pins, 1, __ATOMIC_RELAXED);
            __atomic_fetch_add(&res.num_duplicated_pins, hyperedge.size() - j, __ATOMIC_RELAXED);
            for ( size_t i = j; i < hyperedge.size(); ++i ) {
              hyperedge.pop_back();
            }
          }

          ASSERT(hyperedge.size() >= 2);
          ++current_id;
        } else {
          goto_next_line(mapped_file, current_pos, current_end);
        }
      }
    });
    return res;
  }

  void readHypernodeWeights(char* mapped_file,
                            size_t& pos,
                            const size_t length,
                            const HypernodeID num_hypernodes,
                            const mt_kahypar::Type type,
                            vec<HypernodeWeight>& hypernodes_weight) {
    bool has_hypernode_weights = type == mt_kahypar::Type::NodeWeights ||
                                 type == mt_kahypar::Type::EdgeAndNodeWeights ?
                                 true : false;
    if ( has_hypernode_weights ) {
      hypernodes_weight.resize(num_hypernodes);
      for ( HypernodeID hn = 0; hn < num_hypernodes; ++hn ) {
        ASSERT(pos > 0 && pos < length);
        ASSERT(mapped_file[pos - 1] == '\n');
        hypernodes_weight[hn] = read_number(mapped_file, pos, length);
        do_line_ending(mapped_file, pos);
      }
    }
  }


  void readHypergraphFile(const std::string& filename,
                          HyperedgeID& num_hyperedges,
                          HypernodeID& num_hypernodes,
                          HyperedgeID& num_removed_single_pin_hyperedges,
                          HyperedgeVector& hyperedges,
                          vec<HyperedgeWeight>& hyperedges_weight,
                          vec<HypernodeWeight>& hypernodes_weight,
                          const bool remove_single_pin_hes) {
    ASSERT(!filename.empty(), "No filename for hypergraph file specified");
    FileHandle handle = mmap_file(filename);
    size_t pos = 0;

    // Read Hypergraph Header
    mt_kahypar::Type type = mt_kahypar::Type::Unweighted;
    readHGRHeader(handle.mapped_file, pos, handle.length, num_hyperedges, num_hypernodes, type);

    // Read Hyperedges
    HyperedgeReadResult res =
            readHyperedges(handle.mapped_file, pos, handle.length, num_hyperedges,
              type, hyperedges, hyperedges_weight, remove_single_pin_hes);
    num_hyperedges -= res.num_removed_single_pin_hyperedges;
    num_removed_single_pin_hyperedges = res.num_removed_single_pin_hyperedges;

    if ( res.num_hes_with_duplicated_pins > 0 ) {
      WARNING("Removed" << res.num_duplicated_pins << "duplicated pins in"
        << res.num_hes_with_duplicated_pins << "hyperedges!");
    }

    // Read Hypernode Weights
    readHypernodeWeights(handle.mapped_file, pos, handle.length, num_hypernodes, type, hypernodes_weight);
    ASSERT(pos == handle.length);

    munmap_file(handle);
  }

  void readMetisHeader(char* mapped_file,
                       size_t& pos,
                       const size_t length,
                       HyperedgeID& num_edges,
                       HypernodeID& num_vertices,
                       bool& has_edge_weights,
                       bool& has_vertex_weights) {
    // Skip comments
    while ( mapped_file[pos] == '%' ) {
      goto_next_line(mapped_file, pos, length);
    }

    num_vertices = read_number(mapped_file, pos, length);
    num_edges = read_number(mapped_file, pos, length);

    if (!is_line_ending(mapped_file, pos)) {
      // read the (up to) three 0/1 format digits
      uint32_t format_num = read_number(mapped_file, pos, length);
      ASSERT(format_num < 100, "Vertex sizes in input file are not supported.");
      ASSERT(format_num / 10 == 0 || format_num / 10 == 1);
      has_vertex_weights = (format_num / 10 == 1);
      ASSERT(format_num % 10 == 0 || format_num % 10 == 1);
      has_edge_weights = (format_num % 10 == 1);
    }
    do_line_ending(mapped_file, pos);
  }

  struct VertexRange {
    const size_t start;
    const size_t end;
    const HypernodeID vertex_start_id;
    const HypernodeID num_vertices;
    const HyperedgeID edge_start_id;
  };

  void readVertices(char* mapped_file,
                    size_t& pos,
                    const size_t length,
                    const HyperedgeID num_edges,
                    const HypernodeID num_vertices,
                    const bool has_edge_weights,
                    const bool has_vertex_weights,
                    HyperedgeVector& edges,
                    vec<HyperedgeWeight>& edges_weight,
                    vec<HypernodeWeight>& vertices_weight) {
    vec<VertexRange> vertex_ranges;
    tbb::parallel_invoke([&] {
      // Sequential pass over all vertices to determine ranges in the
      // input file that are read in parallel.
      // Additionally, we need to sum the vertex degrees to determine edge indices.
      size_t current_range_start = pos;
      HypernodeID current_range_vertex_id = 0;
      HypernodeID current_range_num_vertices = 0;
      HyperedgeID current_range_edge_id = 0;
      HyperedgeID current_range_num_edges = 0;
      const HypernodeID num_vertices_per_range = std::max(
              (num_vertices / ( 2 * std::thread::hardware_concurrency())), ID(1));
      while ( current_range_vertex_id + current_range_num_vertices < num_vertices ) {
        // Skip Comments
        ASSERT(pos < length);
        while ( mapped_file[pos] == '%' ) {
          goto_next_line(mapped_file, pos, length);
          ASSERT(pos < length);
        }

        ASSERT(mapped_file[pos - 1] == '\n');
        ++current_range_num_vertices;

        // Count the forward edges, ignore backward edges.
        // This is necessary because we can only calculate unique edge ids
        // efficiently if the edges are deduplicated.
        if ( has_vertex_weights ) {
          read_number(mapped_file, pos, length);
        }
        HyperedgeID vertex_degree = 0;
        while (!is_line_ending(mapped_file, pos) && pos < length) {
          const HypernodeID source = current_range_vertex_id + current_range_num_vertices;
          const HypernodeID target = read_number(mapped_file, pos, length);
          ASSERT(source != target);
          if ( source < target ) {
            ++vertex_degree;
          }
          if ( has_edge_weights ) {
            read_number(mapped_file, pos, length);
          }
        }
        do_line_ending(mapped_file, pos);
        current_range_num_edges += vertex_degree;

        // If there are enough vertices in the current scanned range
        // we store that range, which will be processed in parallel later
        if ( current_range_num_vertices == num_vertices_per_range ) {
          vertex_ranges.push_back(VertexRange {
                  current_range_start, pos, current_range_vertex_id, current_range_num_vertices, current_range_edge_id});
          current_range_start = pos;
          current_range_vertex_id += current_range_num_vertices;
          current_range_num_vertices = 0;
          current_range_edge_id += current_range_num_edges;
          current_range_num_edges = 0;
        }
      }
      if ( current_range_num_vertices > 0 ) {
        vertex_ranges.push_back(VertexRange {
                current_range_start, pos, current_range_vertex_id, current_range_num_vertices, current_range_edge_id});
        current_range_vertex_id += current_range_num_vertices;
        current_range_edge_id += current_range_num_edges;
      }
      ASSERT(current_range_vertex_id == num_vertices);
      ASSERT(current_range_edge_id == num_edges);
    }, [&] {
      edges.resize(num_edges);
    }, [&] {
      if ( has_edge_weights ) {
        edges_weight.resize(num_edges);
      }
    }, [&] {
      if ( has_vertex_weights ) {
        vertices_weight.resize(num_vertices);
      }
    });

    ASSERT([&]() {
        HyperedgeID last_end = 0;
        for(const auto& range: vertex_ranges) {
          if (last_end > range.start) {
            return false;
          }
          last_end = range.end;
        }
        return true;
      }()
    );

    // Process all ranges in parallel, build edge vector and assign weights
    tbb::parallel_for(UL(0), vertex_ranges.size(), [&](const size_t i) {
      const VertexRange& range = vertex_ranges[i];
      size_t current_pos = range.start;
      const size_t current_end = range.end;
      HypernodeID current_vertex_id = range.vertex_start_id;
      const HypernodeID last_vertex_id = current_vertex_id + range.num_vertices;
      HyperedgeID current_edge_id = range.edge_start_id;

      while ( current_vertex_id < last_vertex_id ) {
        // Skip Comments
        ASSERT(current_pos < current_end);
        while ( mapped_file[pos] == '%' ) {
          goto_next_line(mapped_file, current_pos, current_end);
          ASSERT(current_pos < current_end);
        }

        if ( has_vertex_weights ) {
          ASSERT(current_vertex_id < vertices_weight.size());
          vertices_weight[current_vertex_id] = read_number(mapped_file, current_pos, current_end);
        }

        while ( !is_line_ending(mapped_file, current_pos) ) {
          const HypernodeID target = read_number(mapped_file, current_pos, current_end);
          ASSERT(target > 0 && (target - 1) < num_vertices, V(target));

          // process forward edges, ignore backward edges
          if ( current_vertex_id < (target - 1) ) {
            ASSERT(current_edge_id < edges.size());
            // At this point, some magic is involved:
            // In case of the graph partitioner, the right handed expression is considered a pair.
            // In case of the hypergraph partitioner, the right handed expression is considered  a vector.
            edges[current_edge_id] = {current_vertex_id, target - 1};

            if ( has_edge_weights ) {
              edges_weight[current_edge_id] = read_number(mapped_file, current_pos, current_end);
            }
            ++current_edge_id;
          } else if ( has_edge_weights ) {
            read_number(mapped_file, current_pos, current_end);
          }
        }
        do_line_ending(mapped_file, current_pos);
        ++current_vertex_id;
      }
    });
  }

  void readGraphFile(const std::string& filename,
                     HyperedgeID& num_edges,
                     HypernodeID& num_vertices,
                     HyperedgeVector& edges,
                     vec<HyperedgeWeight>& edges_weight,
                     vec<HypernodeWeight>& vertices_weight) {
    ASSERT(!filename.empty(), "No filename for metis file specified");
    FileHandle handle = mmap_file(filename);
    size_t pos = 0;

    // Read Metis Header
    bool has_edge_weights = false;
    bool has_vertex_weights = false;
    readMetisHeader(handle.mapped_file, pos, handle.length, num_edges,
      num_vertices, has_edge_weights, has_vertex_weights);

    // Read Vertices
    readVertices(handle.mapped_file, pos, handle.length, num_edges, num_vertices,
      has_edge_weights, has_vertex_weights, edges, edges_weight, vertices_weight);
    ASSERT(pos == handle.length);

    munmap_file(handle);
  }

  void readPartitionFile(const std::string& filename, std::vector<PartitionID>& partition) {
    ASSERT(!filename.empty(), "No filename for partition file specified");
    ASSERT(partition.empty(), "Partition vector is not empty");
    std::ifstream file(filename);
    if (file) {
      int part;
      while (file >> part) {
        partition.push_back(part);
      }
      file.close();
    } else {
      std::cerr << "Error: File not found: " << std::endl;
    }
  }

  void readPartitionFile(const std::string& filename, PartitionID* partition) {
    ASSERT(!filename.empty(), "No filename for partition file specified");
    std::ifstream file(filename);
    if (file) {
      int part;
      HypernodeID hn = 0;
      while (file >> part) {
        partition[hn++] = part;
      }
      file.close();
    } else {
      std::cerr << "Error: File not found: " << std::endl;
    }
  }

  template<typename PartitionedHypergraph>
  void writePartitionFile(const PartitionedHypergraph& phg, const std::string& filename) {
    if (filename.empty()) {
      LOG << "No filename for partition file specified";
    } else {
      std::ofstream out_stream(filename.c_str());
      std::vector<PartitionID> partition(phg.initialNumNodes(), -1);
      for (const HypernodeID& hn : phg.nodes()) {
        ASSERT(hn < partition.size());
        partition[hn] = phg.partID(hn);
      }
      for (const PartitionID& part : partition) {
        out_stream << part << std::endl;
      }
      out_stream.close();
    }
  }

  namespace {
  #define WRITE_PARTITION_FILE(X) void writePartitionFile(const X& phg, const std::string& filename)
  }

  INSTANTIATE_FUNC_WITH_PARTITIONED_HG(WRITE_PARTITION_FILE)

} // namespace
