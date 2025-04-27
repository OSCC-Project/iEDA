// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************

// #include <gperftools/heap-profiler.h>

#include "gtest/gtest.h"
#include "iir-rust/IRRustC.hh"
#include "log/Log.hh"
#include "string/Str.hh"

using ieda::Log;

using namespace iir;

namespace {

class BuildMatrixTest : public testing::Test {
  void SetUp() {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(BuildMatrixTest, build_equcation) {
  const char* spef_file_path =
      "/home/taosimin/T28/spef/asic_top.spef_vdd_vss_1212.rcworst.0c.spef";
  const char* instance_power_path =
      "/home/shaozheqing/iEDA/bin/report_instance.csv";

  BuildMatrixFromRawData(instance_power_path, spef_file_path);
}

TEST_F(BuildMatrixTest, build_matrix) {
  const char* spef_file_path =
      "/home/taosimin/T28/spef/asic_top.spef_vdd_vss_1212.rcworst.0c.spef";

  auto* rc_data = read_spef(spef_file_path);
  auto one_net_matrix_data =
      build_one_net_conductance_matrix_data(rc_data, "VDD");

  RustMatrix* one_data;
  FOREACH_VEC_ELEM(&one_net_matrix_data.g_matrix_vec, RustMatrix, one_data) {
    LOG_INFO << "row " << one_data->row << " column " << one_data->col
             << " data " << one_data->data;
  }
}

}  // namespace
