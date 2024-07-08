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
/*
 * @Author: sjchanson 13560469332@163.com
 * @Date: 2022-10-31 14:27:37
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-10 20:55:21
 * @FilePath: /irefactor/src/operation/iPL/test/APITest.cc
 * @Description:
 */

#include <string>
 // #include <gperftools/profiler.h>

#include "PLAPI.hh"
#include "PlacerDB.hh"
#include "TimingEval.hpp"
#include "gtest/gtest.h"
#include "idm.h"

namespace ipl {
  class APITestInterface : public testing::Test
  {
    void SetUp()
    {
      // Read Def, Lef
      std::string idb_json_file = "/home/chenshijian/result/iccad15/db_default_config.json";
      dmInst->init(idb_json_file);
    }
    void TearDown() final {}
  };

  TEST_F(APITestInterface, run_flow)
  {
    std::string pl_json_file = "/home/chenshijian/result/iccad15/pl_default_config.json";
    auto* idb_builder = dmInst->get_idb_builder();

    iPLAPIInst.initAPI(pl_json_file, idb_builder);
    iPLAPIInst.runFlow();

    iPLAPIInst.destoryInst();

    idb_builder->saveDef("/home/chenshijian/result/iccad15/iPL_result.def");
  }

  TEST_F(APITestInterface, run_itdp) {
    std::string pl_json_file = "/home/chenshijian/result/iccad15/pl_default_config.json";
    auto* idb_builder = dmInst->get_idb_builder();

    iPLAPIInst.initAPI(pl_json_file, idb_builder);
    iPLAPIInst.runLG();
    iPLAPIInst.runPostGP();

    iPLAPIInst.destoryInst();

    idb_builder->saveDef("/home/chenshijian/result/iccad15/iPL_itdp_result.def");
  }

  TEST_F(APITestInterface, run_gp)
  {
    std::string pl_json_file = "<local_path>/pl_default_config.json";
    auto* idb_builder = dmInst->get_idb_builder();

    iPLAPIInst.initAPI(pl_json_file, idb_builder);
    // ProfilerStart("ipl_gp.prof"); // start
    iPLAPIInst.runGP();
    // ProfilerStop(); // stop
    iPLAPIInst.writeBackSourceDataBase();
    iPLAPIInst.destoryInst();

    idb_builder->saveDef("<local_path>/iPL_gp.def");
  }

  TEST_F(APITestInterface, run_lg)
  {
    std::string pl_json_file = "<local_path>/pl_default_config.json";
    auto* idb_builder = dmInst->get_idb_builder();

    iPLAPIInst.initAPI(pl_json_file, idb_builder);

    // ProfilerStart("ipl_lg.prof"); // start
    iPLAPIInst.runLG();
    // ProfilerStop(); // stop
    iPLAPIInst.reportPLInfo();
    iPLAPIInst.writeBackSourceDataBase();
    iPLAPIInst.destoryInst();

    idb_builder->saveDef("<local_path>/iPL_lg.def");
  }

  TEST_F(APITestInterface, run_dp)
  {
    std::string pl_json_file = "<local_path>/pl_default_config.json";
    auto* idb_builder = dmInst->get_idb_builder();

    iPLAPIInst.initAPI(pl_json_file, idb_builder);
    iPLAPIInst.runLG();
    iPLAPIInst.runDP();
    iPLAPIInst.reportPLInfo();
    iPLAPIInst.writeBackSourceDataBase();
    iPLAPIInst.destoryInst();

    idb_builder->saveDef("<local_path>/iPL_dp.def");
  }

  TEST_F(APITestInterface, run_buffer)
  {
    std::string pl_json_file = "<local_path>/pl_default_config.json";
    auto* idb_builder = dmInst->get_idb_builder();

    iPLAPIInst.initAPI(pl_json_file, idb_builder);
    iPLAPIInst.runBufferInsertion();
    // iPLAPIInst.runLG();
    // iPLAPIInst.reportViolationInfo();
    iPLAPIInst.reportPLInfo();
    iPLAPIInst.writeBackSourceDataBase();
    iPLAPIInst.destoryInst();

    idb_builder->saveDef("<local_path>/iPL_buffer_result.def");
  }

  // TEST_F(APITestInterface, run_mp)
  // {
  //   std::string pl_json_file = "<local_path>/pl_default_config.json";
  //   auto* idb_builder = dmInst->get_idb_builder();

  //   iPLAPIInst.initAPI(pl_json_file, idb_builder);
  //   iPLAPIInst.runMP();
  //   iPLAPIInst.destoryInst();
  // }

  TEST_F(APITestInterface, run_incremental)
  {
    std::string pl_json_file = "<local_path>/pl_default_config.json";
    auto* idb_builder = dmInst->get_idb_builder();

    iPLAPIInst.initAPI(pl_json_file, idb_builder);
    iPLAPIInst.runIncrementalFlow();

    iPLAPIInst.destoryInst();

    idb_builder->saveDef("<local_path>/iPL_incremental.def");
  }

  TEST_F(APITestInterface, run_filler)
  {
    std::string pl_json_file = "<local_path>/pl_default_config.json";
    auto* idb_builder = dmInst->get_idb_builder();

    iPLAPIInst.initAPI(pl_json_file, idb_builder);
    iPLAPIInst.insertLayoutFiller();
    iPLAPIInst.destoryInst();

    idb_builder->saveDef("<local_path>/iPL_filler.def");
  }

  TEST_F(APITestInterface, check_legality)
  {
    std::string pl_json_file = "<local_path>/pl_default_config.json";
    auto* idb_builder = dmInst->get_idb_builder();

    iPLAPIInst.initAPI(pl_json_file, idb_builder);

    iPLAPIInst.checkLegality();
  }

  TEST_F(APITestInterface, long_net)
  {
    std::string pl_json_file = "<local_path>/pl_default_config.json";
    auto* idb_builder = dmInst->get_idb_builder();
    iPLAPIInst.initAPI(pl_json_file, idb_builder);

    iPLAPIInst.reportPLInfo();
  }

  TEST_F(APITestInterface, run_timing)
  {
    std::string pl_json_file = "<local_path>/pl_default_config.json";
    auto* idb_builder = dmInst->get_idb_builder();
    iPLAPIInst.initAPI(pl_json_file, idb_builder);
    iPLAPIInst.initTimingEval();
    iPLAPIInst.updateTiming(PlacerDBInst.get_topo_manager());

    for (std::string clock_name : iPLAPIInst.obtainClockNameList()) {
      double wns = iPLAPIInst.obtainWNS(clock_name.c_str(), ista::AnalysisMode::kMax);
      double tns = iPLAPIInst.obtainTNS(clock_name.c_str(), ista::AnalysisMode::kMax);
      LOG_INFO << "Clock : " << clock_name << " WNS : " << wns << ";"
        << " TNS : " << tns;
    }
  }

  TEST_F(APITestInterface, plot_module)
  {
    std::string pl_json_file = "<local_path>/pl_default_config.json";
    auto* idb_builder = dmInst->get_idb_builder();
    iPLAPIInst.initAPI(pl_json_file, idb_builder);
    iPLAPIInst.initSTA();
    iPLAPIInst.updateSTATiming();

    std::vector<std::string> module_list{ "", "" };

    iPLAPIInst.plotModuleListForDebug(module_list, iPLAPIInst.obtainTargetDir() + "/pl/report/MouduleConnection.gds");
  }

  TEST_F(APITestInterface, print_wl)
  {
    std::string pl_json_file = "<local_path>/pl_default_config.json";
    auto* idb_builder = dmInst->get_idb_builder();
    iPLAPIInst.initAPI(pl_json_file, idb_builder);
    // iPLAPIInst.initSTA();
    // iPLAPIInst.updateSTATiming();

    iPLAPIInst.printHPWLInfo();
  }

  TEST_F(APITestInterface, print_peak_density)
  {
    std::string pl_json_file = "<local_path>/pl_default_config.json";
    auto* idb_builder = dmInst->get_idb_builder();
    iPLAPIInst.initAPI(pl_json_file, idb_builder);
    iPLAPIInst.reportPLInfo();
  }

  TEST_F(APITestInterface, incremental_legal)
  {
    std::string pl_json_file = "<local_path>/pl_default_config.json";
    auto* idb_builder = dmInst->get_idb_builder();
    iPLAPIInst.initAPI(pl_json_file, idb_builder);
    iPLAPIInst.runLG();

    std::vector<std::string> incremental_insts;
    std::vector<Instance*> inst_list;

    std::string inst1_name = "PLInstance1";
    incremental_insts.push_back(inst1_name);
    Instance* inst1 = new Instance(inst1_name);
    Cell* inst1_cell = PlacerDBInst.get_layout()->find_cell("sky130_fd_sc_hs__nor2_4");
    inst1->set_cell_master(inst1_cell);
    inst1->set_shape(126240, 174825, 126240 + inst1_cell->get_width(), 174825 + inst1_cell->get_height());
    inst1->set_instance_type(INSTANCE_TYPE::kNormal);
    inst1->set_instance_state(INSTANCE_STATE::KUnPlaced);
    inst1->set_orient(Orient::kN_R0);
    inst_list.push_back(inst1);
    PlacerDBInst.updateInstancesForDebug(inst_list);

    iPLAPIInst.plotModuleStateForDebug(incremental_insts, iPLAPIInst.obtainTargetDir() + "/pl/report/InstanceState_before.gds");

    iPLAPIInst.runIncrLG(incremental_insts);
    iPLAPIInst.reportPLInfo();

    iPLAPIInst.plotModuleStateForDebug(incremental_insts, iPLAPIInst.obtainTargetDir() + "/pl/report/InstanceState_after.gds");
  }

}  // namespace ipl