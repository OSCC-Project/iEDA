// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************

// #include <gperftools/heap-profiler.h>

#include "gtest/gtest.h"
#include "liberty/Liberty.hh"
#include "log/Log.hh"
#include "string/Str.hh"

using ieda::Log;
using ista::Liberty;

using namespace ista;

namespace {

class LibertyTest : public testing::Test {
  void SetUp() {
    char config[] = "test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(LibertyTest, read) {
  LOG_INFO << "lib test";
  Liberty lib;
  auto library = lib.loadLiberty(
      "/home/smtao/peda/iEDA/src/iSTA/examples/example1_fast.lib");
  LOG_INFO << "build lib test";

  EXPECT_TRUE(library);
}

TEST_F(LibertyTest, ccs) {
  LOG_INFO << "lib ccs test";
  const char* lib_path =
      "/home/smtao/opensource/dctk/test/"
      "NangateOpenCellLibrary_typical_ccs.adjusted.lib";
  /// home/smtao/ccs/scc011ums_hd_rvt_ss_v1p08_125c_ccs.lib
  Liberty lib;
  auto library = lib.loadLiberty(lib_path);

  LOG_INFO << "build lib test";

  EXPECT_TRUE(library);
}

TEST_F(LibertyTest, bus) {
  LOG_INFO << "lib bus test";
  const char* lib_path =
      "/home/smtao/nutshell/"
      "S013PLLFN_v1.5.1_typ.lib";

  Liberty lib;
  auto library = lib.loadLiberty(lib_path);

  LOG_INFO << "build lib test";
  EXPECT_TRUE(library);
}

TEST_F(LibertyTest, large) {
  LOG_INFO << "lib large test";
  const char* lib_path =
      "/home/taosimin/nutshell/"
      "scc011ums_hd_hvt_ss_v1p08_125c_ccs.lib";

  std::unique_ptr<LibertyLibrary> library;

  {
    Liberty lib;
    library = lib.loadLiberty(lib_path);

    // HeapProfilerStart("libertymem");
    LOG_INFO << "build large lib test";
    EXPECT_TRUE(library);

    // HeapProfilerDump("liberty parser");
    // LOG_INFO << GetHeapProfile();
    // HeapProfilerStop();
  }
}

TEST_F(LibertyTest, wire_load) {
  LOG_INFO << "lib wire load test";
  const char* lib_path =
      "/home/taosimin/iEDA-main/iEDA/src/iSTA/examples/example1_fast.lib";

  std::unique_ptr<LibertyLibrary> library;

  {
    Liberty lib;
    library = lib.loadLiberty(lib_path);
    EXPECT_TRUE(library);

    for (auto& wire_load : library->get_wire_loads()) {
      LOG_INFO << wire_load->get_wire_load_name();
    }

    LOG_INFO << "default wire load:" << library->get_default_wire_load();
  }
}

}  // namespace