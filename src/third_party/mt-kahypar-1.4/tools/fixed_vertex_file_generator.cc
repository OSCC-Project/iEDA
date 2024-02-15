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
#include <string>

#include "mt-kahypar/macros.h"
#include "mt-kahypar/datastructures/hypergraph_common.h"
#include "mt-kahypar/io/hypergraph_io.h"
#include "mt-kahypar/utils/randomize.h"

namespace po = boost::program_options;
using namespace mt_kahypar;

int main(int argc, char* argv[]) {
  std::string partition_file;
  double percentage = 0;
  PartitionID k;

  po::options_description options("Options");
  options.add_options()
    ("partition-file,p",
    po::value<std::string>(&partition_file)->value_name("<string>")->required(),
    "Partition file")
    ("num-blocks,k",
    po::value<PartitionID>(&k)->value_name("<int32_t>")->required(),
    "Number of blocks")
    ("fixed-vertex-percentage",
    po::value<double>(&percentage)->value_name("<double>")->required(),
    "Percentage of Fixed Vertices");

  po::variables_map cmd_vm;
  po::store(po::parse_command_line(argc, argv, options), cmd_vm);
  po::notify(cmd_vm);

  std::vector<PartitionID> partition;
  io::readPartitionFile(partition_file, partition);

  int threshold = percentage * 1000;
  std::string fixed_vertex_file = partition_file;
  fixed_vertex_file.erase(fixed_vertex_file.find_first_of("."), std::string::npos);
  fixed_vertex_file = fixed_vertex_file + ".k" + std::to_string(k) + + ".p" + std::to_string(threshold / 10) + ".fix";
  std::ofstream out_stream(fixed_vertex_file.c_str());
  utils::Randomize rand = utils::Randomize::instance();
  for ( size_t i = 0; i < partition.size(); ++i ) {
    int num = rand.getRandomInt(0, 1000, THREAD_ID);
    if ( num < threshold ) {
      out_stream << partition[i] << std::endl;
    } else {
      out_stream << kInvalidPartition << std::endl;
    }
  }
  out_stream.close();

  return 0;
}
