/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
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

#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "tbb/parallel_sort.h"
#include "tbb/enumerable_thread_specific.h"
#include "tbb/parallel_reduce.h"

#include "mt-kahypar/macros.h"
#include "mt-kahypar/datastructures/static_hypergraph.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/io/hypergraph_io.h"
#include "mt-kahypar/utils/cast.h"
#include "mt-kahypar/utils/delete.h"
#include "mt-kahypar/utils/hypergraph_statistics.h"

#include "kahypar-resources/utils/math.h"

using namespace mt_kahypar;
namespace po = boost::program_options;

using Hypergraph = ds::StaticHypergraph;

struct Statistic {
  uint64_t min = 0;
  uint64_t q1 = 0;
  uint64_t med = 0;
  uint64_t q3 = 0;
  uint64_t top90 = 0;
  uint64_t max = 0;
  double avg = 0.0;
  double sd = 0.0;
};

template <typename T>
Statistic createStats(const std::vector<T>& vec, const double avg, const double stdev) {
  Statistic stats;
  if (!vec.empty()) {
    const auto quartiles = kahypar::math::firstAndThirdQuartile(vec);
    stats.min = vec.front();
    stats.q1 = quartiles.first;
    stats.med = kahypar::math::median(vec);
    stats.q3 = quartiles.second;
    stats.top90 = vec[ceil(90.0 / 100 * (vec.size() - 1))];
    stats.max = vec.back();
    stats.avg = avg;
    stats.sd = stdev;
  }
  return stats;
}

int main(int argc, char* argv[]) {
  Context context;

  po::options_description options("Options");
  options.add_options()
          ("hypergraph,h",
           po::value<std::string>(&context.partition.graph_filename)->value_name("<string>")->required(),
           "Hypergraph Filename")
          ("input-file-format",
            po::value<std::string>()->value_name("<string>")->notifier([&](const std::string& s) {
              if (s == "hmetis") {
                context.partition.file_format = FileFormat::hMetis;
              } else if (s == "metis") {
                context.partition.file_format = FileFormat::Metis;
              }
            }),
            "Input file format: \n"
            " - hmetis : hMETIS hypergraph file format \n"
            " - metis : METIS graph file format");

  po::variables_map cmd_vm;
  po::store(po::parse_command_line(argc, argv, options), cmd_vm);
  po::notify(cmd_vm);

  // Read Hypergraph
  mt_kahypar_hypergraph_t hypergraph =
    mt_kahypar::io::readInputFile(
      context.partition.graph_filename, PresetType::default_preset,
      InstanceType::hypergraph, context.partition.file_format, true);
  Hypergraph& hg = utils::cast<Hypergraph>(hypergraph);

  std::vector<HypernodeID> he_sizes;
  std::vector<HyperedgeWeight> he_weights;
  std::vector<HyperedgeID> hn_degrees;
  std::vector<HypernodeWeight> hn_weights;

  tbb::parallel_invoke([&] {
    he_sizes.resize(hg.initialNumEdges());
  }, [&] {
    he_weights.resize(hg.initialNumEdges());
  }, [&] {
    hn_degrees.resize(hg.initialNumNodes());
  }, [&] {
    hn_weights.resize(hg.initialNumNodes());
  });

  HypernodeID num_hypernodes = hg.initialNumNodes();
  const double avg_hn_degree = utils::avgHypernodeDegree(hg);
  hg.doParallelForAllNodes([&](const HypernodeID& hn) {
    hn_degrees[hn] = hg.nodeDegree(hn);
    hn_weights[hn] = hg.nodeWeight(hn);
  });
  const double avg_hn_weight = utils::parallel_avg(hn_weights, num_hypernodes);
  const double stdev_hn_degree = utils::parallel_stdev(hn_degrees, avg_hn_degree, num_hypernodes);
  const double stdev_hn_weight = utils::parallel_stdev(hn_weights, avg_hn_weight, num_hypernodes);

  HyperedgeID num_hyperedges = hg.initialNumEdges();
  const double avg_he_size = utils::avgHyperedgeDegree(hg);
  tbb::enumerable_thread_specific<size_t> single_pin_hes(0);
  hg.doParallelForAllEdges([&](const HyperedgeID& he) {
    he_sizes[he] = hg.edgeSize(he);
    he_weights[he] = hg.edgeWeight(he);
    if ( hg.edgeSize(he) == 1 ) {
      ++single_pin_hes.local();
    }
  });
  const double avg_he_weight = utils::parallel_avg(he_weights, num_hyperedges);
  const double stdev_he_size = utils::parallel_stdev(he_sizes, avg_he_size, num_hyperedges);
  const double stdev_he_weight = utils::parallel_stdev(he_weights, avg_he_weight, num_hyperedges);

  tbb::enumerable_thread_specific<size_t> graph_edge_count(0);
  hg.doParallelForAllEdges([&](const HyperedgeID& he) {
    if (hg.edgeSize(he) == 2) {
      graph_edge_count.local() += 1;
    }
  });

  HyperedgeWeight total_he_weight = 0;
  tbb::parallel_invoke([&] {
    tbb::parallel_sort(he_sizes.begin(), he_sizes.end());
  }, [&] {
    tbb::parallel_sort(he_weights.begin(), he_weights.end());
  }, [&] {
    tbb::parallel_sort(hn_degrees.begin(), hn_degrees.end());
  }, [&] {
    tbb::parallel_sort(hn_weights.begin(), hn_weights.end());
  }, [&] {
    total_he_weight = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, he_weights.size()), 0,
      [&](tbb::blocked_range<size_t> r, HyperedgeWeight running_total) {
          for (size_t i = r.begin(); i < r.end(); ++i) {
              running_total += he_weights[i];
          }
          return running_total;
      }, std::plus<HyperedgeWeight>() );
  });

  Statistic he_size_stats = createStats(he_sizes, avg_he_size, stdev_he_size);
  Statistic he_weight_stats = createStats(he_weights, avg_he_weight, stdev_he_weight);
  Statistic hn_degree_stats = createStats(hn_degrees, avg_hn_degree, stdev_hn_degree);
  Statistic hn_weight_stats = createStats(hn_weights, avg_hn_weight, stdev_hn_weight);

  std::string graph_name = context.partition.graph_filename.substr(
    context.partition.graph_filename.find_last_of("/") + 1);
  std::cout  << "RESULT graph=" << graph_name
             << " HNs=" << hg.initialNumNodes()
             << " HEs=" << hg.initialNumEdges()
             << " pins=" << hg.initialNumPins()
             << " numSingleNodeHEs=" << single_pin_hes.combine(std::plus<size_t>())
             << " avgHEsize=" << he_size_stats.avg
             << " sdHEsize=" << he_size_stats.sd
             << " minHEsize=" << he_size_stats.min
             << " heSize90thPercentile=" << he_size_stats.top90
             << " Q1HEsize=" << he_size_stats.q1
             << " medHEsize=" << he_size_stats.med
             << " Q3HEsize=" << he_size_stats.q3
             << " maxHEsize=" << he_size_stats.max
             << " totalHEweight=" << total_he_weight
             << " avgHEweight=" << he_weight_stats.avg
             << " sdHEweight=" << he_weight_stats.sd
             << " minHEweight=" << he_weight_stats.min
             << " Q1HEweight=" << he_weight_stats.q1
             << " medHEweight=" << he_weight_stats.med
             << " Q3HEweight=" << he_weight_stats.q3
             << " maxHEweight=" << he_weight_stats.max
             << " avgHNdegree=" << hn_degree_stats.avg
             << " sdHNdegree=" << hn_degree_stats.sd
             << " minHnDegree=" << hn_degree_stats.min
             << " hnDegree90thPercentile=" << hn_degree_stats.top90
             << " maxHnDegree=" << hn_degree_stats.max
             << " Q1HNdegree=" << hn_degree_stats.q1
             << " medHNdegree=" << hn_degree_stats.med
             << " Q3HNdegree=" << hn_degree_stats.q3
             << " totalHNweight=" << hg.totalWeight()
             << " avgHNweight=" << hn_weight_stats.avg
             << " sdHNweight=" << hn_weight_stats.sd
             << " minHNweight=" << hn_weight_stats.min
             << " Q1HNweight=" << hn_weight_stats.q1
             << " medHNweight=" << hn_weight_stats.med
             << " Q3HNweight=" << hn_weight_stats.q3
             << " maxHNweight=" << hn_weight_stats.max
             << " density=" << static_cast<double>(hg.initialNumEdges()) / hg.initialNumNodes()
             << std::endl;

  utils::delete_hypergraph(hypergraph);

  return 0;
}
