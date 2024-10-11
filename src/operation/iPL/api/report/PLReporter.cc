#include "PLReporter.hh"

#include "module/checker/layout_checker/LayoutChecker.hh"
#include "module/evaluator/density/Density.hh"
#include "module/evaluator/wirelength/HPWirelength.hh"
#include "module/evaluator/wirelength/SteinerWirelength.hh"
#include "module/logger/Log.hh"
#include "time/Time.hh"
#include "usage/usage.hh"

#include <fstream>
#include "report/ReportTable.hh"
#include <set>
#include "netlist/Net.hh"
#include "congestion_db.h"


namespace ipl {

  PLReporter::PLReporter(ExternalAPI* external_api)
  {
    _external_api = external_api;
  }

  PLReporter::~PLReporter()
  {
    //
  }

  void PLReporter::reportPLInfo(std::string target_dir)
  {
    LOG_INFO << "-----------------Start iPL Report Generation-----------------";

    ieda::Stats report_status;

    // std::string design_name = PlacerDBInst.get_design()->get_design_name();
    // std::string output_dir = "./evaluation_task/benchmark/" + design_name + "/pl_reports/";
    std::string output_dir = target_dir;

    std::string summary_file = "summary_report.txt";
    std::ofstream summary_stream;
    summary_stream.open(output_dir + "/" + summary_file);
    if (!summary_stream.good()) {
      LOG_WARNING << "Cannot open file for summary report !";
    }
    summary_stream << "Generate the report at " << ieda::Time::getNowWallTime() << std::endl;

    // report base info
    reportPLBaseInfo(summary_stream);

    // report violation info
    reportViolationInfo(summary_stream, target_dir);

    // report wirelength info
    reportWLInfo(summary_stream, target_dir);

    // report density info
    reportBinDensity(summary_stream);

    // report timing info
    if (PlacerDBInst.get_placer_config()->isTimingEffort()) {
      reportTimingInfo(summary_stream);
    }

    // report congestion
    // reportCongestionInfo(summary_stream);

    summary_stream.close();

    double time_delta = report_status.elapsedRunTime();

    LOG_INFO << "Report Generation Total Time Elapsed: " << time_delta << "s";
    LOG_INFO << "-----------------Finish Report Generation-----------------";
  }

  void PLReporter::reportTopoInfo() {
    //
  }

  void PLReporter::reportViolationInfo(std::ofstream& feed, std::string target_dir)
  {
    std::string output_dir = target_dir;
    std::string violation_detail_file = "violation_detail_report.txt";
    std::ofstream violation_detail_stream;
    violation_detail_stream.open(output_dir + "/" + violation_detail_file);
    if (!violation_detail_stream.good()) {
      LOG_WARNING << "Cannot open file for violation detail report !";
    }
    violation_detail_stream << "Generate the report at " << ieda::Time::getNowWallTime() << std::endl;

    auto report_tbl = _external_api->generateTable("table");
    (*report_tbl) << TABLE_HEAD;
    (*report_tbl)[0][0] = "Violation Info";
    (*report_tbl)[0][1] = "Value";

    // violation info
    int32_t core_violated_cnt = 0;
    int32_t rowsite_violated_cnt = 0;
    int32_t power_violated_cnt = 0;
    int32_t overlap_violated_cnt = 0;

    LayoutChecker* checker = new LayoutChecker(&PlacerDBInst);
    LOG_INFO << "Detect Core outside Instances...";
    std::vector<Instance*> illegal_outside_inst_list = checker->obtainIllegalInstInsideCore();
    if (static_cast<int32_t>(illegal_outside_inst_list.size()) != 0) {
      core_violated_cnt = static_cast<int32_t>(illegal_outside_inst_list.size());
      LOG_ERROR << "Illegal Outside Instances Count : " << core_violated_cnt;
      violation_detail_stream << "Illegal Outside Instances Count : " << illegal_outside_inst_list.size() << std::endl;
      for (auto inst : illegal_outside_inst_list) {
        violation_detail_stream << "Illegal Location Instance " << inst->get_name() << " Location : " << inst->get_shape().get_ll_x() << ","
          << inst->get_shape().get_ll_y() << " " << inst->get_shape().get_ur_x() << "," << inst->get_shape().get_ur_y()
          << std::endl;
      }
      violation_detail_stream << std::endl;
    }

    LOG_INFO << "Detect Instances' Alignment...";
    std::vector<Instance*> illegal_loc_inst_list = checker->obtainIllegalInstAlignRowSite();
    if (static_cast<int32_t>(illegal_loc_inst_list.size()) != 0) {
      rowsite_violated_cnt = static_cast<int32_t>(illegal_loc_inst_list.size());
      LOG_ERROR << "Illegal Alignment Instances Count : " << rowsite_violated_cnt;
      violation_detail_stream << "Illegal Alignment Instances Count : " << illegal_loc_inst_list.size() << std::endl;
      for (auto inst : illegal_loc_inst_list) {
        violation_detail_stream << "Illegal Location Instance " << inst->get_name() << " Location : " << inst->get_shape().get_ll_x() << ","
          << inst->get_shape().get_ll_y() << " " << inst->get_shape().get_ur_x() << "," << inst->get_shape().get_ur_y()
          << std::endl;
      }
      violation_detail_stream << std::endl;
    }

    LOG_INFO << "Detect Power Alignment...";
    std::vector<Instance*> illegal_power_inst_list = checker->obtainIllegalInstAlignPower();
    if (static_cast<int32_t>(illegal_power_inst_list.size()) != 0) {
      power_violated_cnt = static_cast<int32_t>(illegal_power_inst_list.size());
      LOG_ERROR << "Illegal Power Orient Instances Count : " << power_violated_cnt;
      violation_detail_stream << "Illegal Power Orient Instances Count : " << illegal_power_inst_list.size() << std::endl;
      for (auto inst : illegal_power_inst_list) {
        violation_detail_stream << "Illegal Power Orient Instance " << inst->get_name() << " Location : " << inst->get_shape().get_ll_x()
          << "," << inst->get_shape().get_ll_y() << " " << inst->get_shape().get_ur_x() << ","
          << inst->get_shape().get_ur_y() << std::endl;
      }
      violation_detail_stream << std::endl;
    }

    LOG_INFO << "Detect Overlap Between Instances...";
    if (!checker->isNoOverlapAmongInsts()) {
      overlap_violated_cnt = reportOverlapInfo(violation_detail_stream);
      LOG_ERROR << "Overlap Exist";
    }

    delete checker;

    (*report_tbl)[1][0] = "Core Range Violated Count";
    (*report_tbl)[1][1] = std::to_string(core_violated_cnt);
    (*report_tbl)[2][0] = "Row/Site Alignment Violated Count";
    (*report_tbl)[2][1] = std::to_string(rowsite_violated_cnt);
    (*report_tbl)[3][0] = "Power Alignment Violated Count";
    (*report_tbl)[3][1] = std::to_string(power_violated_cnt);
    (*report_tbl)[4][0] = "Overlap Violated Count";
    (*report_tbl)[4][1] = std::to_string(overlap_violated_cnt);
    (*report_tbl) << TABLE_ENDLINE;
    feed << (*report_tbl).to_string() << std::endl;

    violation_detail_stream.close();
    LOG_INFO << "Detail Violations Info Writed to "
      << "'" << output_dir << "'";
  }

  void PLReporter::reportLayoutWhiteInfo(std::string target_dir)
  {
    LayoutChecker* checker = new LayoutChecker(&PlacerDBInst);

    auto white_site_list = checker->obtainWhiteSiteList();

    if (white_site_list.size()) {
      LOG_INFO << "Detect Sites haven't been filled Count : " << white_site_list.size();
      // int32_t dbu = PlacerDBInst.get_layout()->get_database_unit();

      std::ofstream file_stream;
      file_stream.open(target_dir + "/WhiteSites.txt");
      if (!file_stream.good()) {
        LOG_WARNING << "Cannot open file for white sites !";
      }

      int32_t idx = 1;
      for (auto rect : white_site_list) {
        // file_stream << idx++ << " (ll_x,ll_y ur_x,ur_y) : " << static_cast<float>(rect.get_ll_x()) / dbu  << "," <<
        // static_cast<float>(rect.get_ll_y()) / dbu << " " << static_cast<float>(rect.get_ur_x()) / dbu << "," <<
        // static_cast<float>(rect.get_ur_y()) / dbu << std::endl;
        file_stream << idx++ << " (ll_x,ll_y ur_x,ur_y) : " << rect.get_ll_x() << "," << rect.get_ll_y() << " " << rect.get_ur_x() << ","
          << rect.get_ur_y() << std::endl;
      }
      file_stream.close();
      LOG_INFO << "White Sites has writen to file ./result/pl/WhiteSites.txt";
    }

    delete checker;
  }

  void PLReporter::reportBinDensity(std::ofstream& feed)
  {
    auto report_tbl = _external_api->generateTable("table");
    (*report_tbl) << TABLE_HEAD;
    (*report_tbl)[0][0] = "Bin Density Info";
    (*report_tbl)[0][1] = "Value";

    auto core_shape = PlacerDBInst.get_layout()->get_core_shape();
    int32_t bin_cnt_x = PlacerDBInst.get_placer_config()->get_nes_config().get_bin_cnt_x();
    int32_t bin_cnt_y = PlacerDBInst.get_placer_config()->get_nes_config().get_bin_cnt_y();
    float target_density = PlacerDBInst.get_placer_config()->get_nes_config().get_target_density();

    GridManager grid_manager(core_shape, bin_cnt_x, bin_cnt_y, target_density, 1);

    // add inst
    for (auto* inst : PlacerDBInst.get_design()->get_instance_list()) {
      if (inst->isOutsideInstance()) {
        continue;
      }
      if (inst->get_coordi().isUnLegal()) {
        continue;
      }
      auto inst_shape = std::move(inst->get_shape());
      std::vector<Grid*> overlap_grid_list;
      grid_manager.obtainOverlapGridList(overlap_grid_list, inst_shape);
      for (auto* grid : overlap_grid_list) {
        int64_t overlap_area = grid_manager.obtainOverlapArea(grid, inst_shape);
        grid->occupied_area += overlap_area;
        // grid->add_area(overlap_area);
      }
    }

    Density density_eval(&grid_manager);
    float peak_density = density_eval.obtainPeakBinDensity();

    (*report_tbl)[1][0] = "Peak BinDensity";
    (*report_tbl)[1][1] = std::to_string(peak_density);
    (*report_tbl) << TABLE_ENDLINE;
    feed << (*report_tbl).to_string() << std::endl;
  }

  // NOLINTNEXTLINE
  void plotPinName(std::stringstream& feed, std::string net_name, TreeNode* tree_node)
  {
    feed << "TEXT" << std::endl;
    feed << "LAYER 1" << std::endl;
    feed << "TEXTTYPE 0" << std::endl;
    feed << "PRESENTATION 0,2,0" << std::endl;
    feed << "PATHTYPE 1" << std::endl;
    feed << "STRANS 0,0,0" << std::endl;
    feed << "MAG 1875" << std::endl;
    feed << "XY" << std::endl;
    int32_t pin_x = tree_node->get_point().get_x();
    int32_t pin_y = tree_node->get_point().get_y();
    feed << pin_x << " : " << pin_y << std::endl;

    auto* node = tree_node->get_node();
    if (node) {  // the real pin.
      feed << "STRING " + node->get_name() << std::endl;
    }
    else {  // the stainer pin.
      feed << "STRING " + net_name + " : " + "s" << std::endl;
    }
    feed << "ENDEL" << std::endl;
  }

  void PLReporter::plotConnectionForDebug(std::vector<std::string> net_name_list, std::string path)
  {
    std::ofstream file_stream;
    file_stream.open(path);
    if (!file_stream.good()) {
      LOG_WARNING << "Cannot open file for connection info !";
    }

    auto* ipl_layout = PlacerDBInst.get_layout();
    auto* ipl_design = PlacerDBInst.get_design();
    auto* topo_manager = PlacerDBInst.get_topo_manager();

    std::vector<NetWork*> net_list;
    for (std::string net_name : net_name_list) {
      auto* pl_net = ipl_design->find_net(net_name);
      auto* network = topo_manager->findNetworkById(pl_net->get_net_id());
      if (!network) {
        LOG_WARNING << "Net : " << net_name << " Not Found!";
      }
      else {
        net_list.push_back(network);
      }
    }

    SteinerWirelength stwl_eval(PlacerDBInst.get_topo_manager());
    stwl_eval.updatePartOfNetWorkPointPair(net_list);

    std::stringstream feed;
    feed << "HEADER 600" << std::endl;
    feed << "BGNLIB" << std::endl;
    feed << "LIBNAME ITDP_LIB" << std::endl;
    feed << "UNITS 0.001 1e-9" << std::endl;
    feed << "BGNSTR" << std::endl;
    feed << "STRNAME core" << std::endl;
    feed << "BOUNDARY" << std::endl;
    feed << "LAYER 0" << std::endl;
    feed << "DATATYPE 0" << std::endl;
    feed << "XY" << std::endl;

    auto core_shape = ipl_layout->get_core_shape();
    feed << core_shape.get_ll_x() << " : " << core_shape.get_ll_y() << std::endl;
    feed << core_shape.get_ur_x() << " : " << core_shape.get_ll_y() << std::endl;
    feed << core_shape.get_ur_x() << " : " << core_shape.get_ur_y() << std::endl;
    feed << core_shape.get_ll_x() << " : " << core_shape.get_ur_y() << std::endl;
    feed << core_shape.get_ll_x() << " : " << core_shape.get_ll_y() << std::endl;
    feed << "ENDEL" << std::endl;

    // tmp set the wire width.
    int32_t wire_width = 160;

    for (auto* network : net_list) {
      auto* multi_tree = stwl_eval.obtainMultiTree(network);
      std::set<TreeNode*> visited_tree_nodes;
      std::queue<TreeNode*> tree_node_queue;
      tree_node_queue.push(multi_tree->get_root());
      while (!tree_node_queue.empty()) {
        auto* source_tree_node = tree_node_queue.front();
        if (visited_tree_nodes.find(source_tree_node) == visited_tree_nodes.end()) {
          plotPinName(feed, network->get_name(), source_tree_node);
          visited_tree_nodes.emplace(source_tree_node);
        }
        for (auto* sink_tree_node : source_tree_node->get_child_list()) {
          plotPinName(feed, network->get_name(), sink_tree_node);
          visited_tree_nodes.emplace(sink_tree_node);
          tree_node_queue.push(sink_tree_node);

          // plot wire.
          feed << "PATH" << std::endl;
          feed << "LAYER 2" << std::endl;
          feed << "DATATYPE 0" << std::endl;
          feed << "WIDTH " + std::to_string(wire_width) << std::endl;
          feed << "XY" << std::endl;
          feed << source_tree_node->get_point().get_x() << " : " << source_tree_node->get_point().get_y() << std::endl;
          feed << sink_tree_node->get_point().get_x() << " : " << sink_tree_node->get_point().get_y() << std::endl;
          feed << "ENDEL" << std::endl;
        }
        tree_node_queue.pop();
      }
    }
    feed << "ENDSTR" << std::endl;
    feed << "ENDLIB" << std::endl;
    file_stream << feed.str();
    feed.clear();
    file_stream.close();
  }

  void PLReporter::plotModuleListForDebug(std::vector<std::string> module_prefix_list, std::string path)
  {
    std::ofstream file_stream;
    file_stream.open(path);
    if (!file_stream.good()) {
      LOG_WARNING << "Cannot open file for module list !";
    }

    auto* ipl_layout = PlacerDBInst.get_layout();
    auto* ipl_design = PlacerDBInst.get_design();
    auto* topo_manager = PlacerDBInst.get_topo_manager();

    std::map<std::string, std::vector<Instance*>> inst_list_map;
    for (std::string prefix_name : module_prefix_list) {
      inst_list_map.emplace(prefix_name, std::vector<Instance*>{});
    }

    for (auto* inst : ipl_design->get_instance_list()) {
      std::string inst_name = inst->get_name();
      for (auto pair : inst_list_map) {
        std::string prefix_name = pair.first;
        if (inst_name.size() < prefix_name.size()) {
          continue;
        }
        if (inst_name.substr(0, prefix_name.size()) == prefix_name) {
          inst_list_map[prefix_name].push_back(inst);
        }
      }
    }

    SteinerWirelength stwl_eval(PlacerDBInst.get_topo_manager());
    stwl_eval.updateAllNetWorkPointPair();

    std::stringstream feed;
    feed << "HEADER 600" << std::endl;
    feed << "BGNLIB" << std::endl;
    feed << "LIBNAME ITDP_LIB" << std::endl;
    feed << "UNITS 0.001 1e-9" << std::endl;
    feed << "BGNSTR" << std::endl;
    feed << "STRNAME core" << std::endl;
    feed << "BOUNDARY" << std::endl;
    feed << "LAYER 0" << std::endl;
    feed << "DATATYPE 0" << std::endl;
    feed << "XY" << std::endl;

    auto core_shape = ipl_layout->get_core_shape();
    feed << core_shape.get_ll_x() << " : " << core_shape.get_ll_y() << std::endl;
    feed << core_shape.get_ur_x() << " : " << core_shape.get_ll_y() << std::endl;
    feed << core_shape.get_ur_x() << " : " << core_shape.get_ur_y() << std::endl;
    feed << core_shape.get_ll_x() << " : " << core_shape.get_ur_y() << std::endl;
    feed << core_shape.get_ll_x() << " : " << core_shape.get_ll_y() << std::endl;
    feed << "ENDEL" << std::endl;

    // print all instances.
    for (auto* inst : ipl_design->get_instance_list()) {
      feed << "BOUNDARY" << std::endl;
      feed << "LAYER 0" << std::endl;
      feed << "DATATYPE 0" << std::endl;
      feed << "XY" << std::endl;
      feed << inst->get_shape().get_ll_x() << " : " << inst->get_shape().get_ll_y() << std::endl;
      feed << inst->get_shape().get_ur_x() << " : " << inst->get_shape().get_ll_y() << std::endl;
      feed << inst->get_shape().get_ur_x() << " : " << inst->get_shape().get_ur_y() << std::endl;
      feed << inst->get_shape().get_ll_x() << " : " << inst->get_shape().get_ur_y() << std::endl;
      feed << inst->get_shape().get_ll_x() << " : " << inst->get_shape().get_ll_y() << std::endl;
      feed << "ENDEL" << std::endl;
    }

    // print specify instances and store unrelative nets.
    int32_t layer_idx = 1;
    for (auto pair : inst_list_map) {
      auto& inst_list = pair.second;
      std::set<NetWork*> relative_nets;
      std::set<NetWork*> unrelative_nets;
      for (auto* inst : inst_list) {
        auto* group = topo_manager->findGroupById(inst->get_inst_id());
        for (auto* node : group->get_node_list()) {
          auto* network = node->get_network();
          bool skip_flag = false;
          for (auto* network_node : network->get_node_list()) {
            // skip the origin node.
            if (network_node == node) {
              continue;
            }
            // same type instance.
            auto* node_group = network_node->get_group();
            if (node_group) {
              std::string node_group_name = node_group->get_name();
              if ((node_group_name.size() >= pair.first.size()) && (node_group_name.substr(0, pair.first.size()) == pair.first)) {
                relative_nets.emplace(network);
                skip_flag = true;
                continue;
              }
            }
            if (skip_flag == true) {
              break;
            }

            unrelative_nets.emplace(network);
          }
        }

        feed << "BOUNDARY" << std::endl;
        feed << "LAYER " << layer_idx << std::endl;
        feed << "DATATYPE 0" << std::endl;
        feed << "XY" << std::endl;
        feed << inst->get_shape().get_ll_x() << " : " << inst->get_shape().get_ll_y() << std::endl;
        feed << inst->get_shape().get_ur_x() << " : " << inst->get_shape().get_ll_y() << std::endl;
        feed << inst->get_shape().get_ur_x() << " : " << inst->get_shape().get_ur_y() << std::endl;
        feed << inst->get_shape().get_ll_x() << " : " << inst->get_shape().get_ur_y() << std::endl;
        feed << inst->get_shape().get_ll_x() << " : " << inst->get_shape().get_ll_y() << std::endl;
        // feed << "STRING " << pair.first << std::endl;
        feed << "ENDEL" << std::endl;
      }

      // print relative nets.
      layer_idx++;
      int64_t relative_wl = 0;
      for (auto* network : relative_nets) {
        // skip the clock net.
        if (_external_api->isClockNet(network->get_name())) {
          continue;
        }

        relative_wl += stwl_eval.obtainNetWirelength(network->get_network_id());
        auto* multi_tree = stwl_eval.obtainMultiTree(network);
        std::set<TreeNode*> visited_tree_nodes;
        std::queue<TreeNode*> tree_node_queue;
        tree_node_queue.push(multi_tree->get_root());
        while (!tree_node_queue.empty()) {
          auto* source_tree_node = tree_node_queue.front();
          if (visited_tree_nodes.find(source_tree_node) == visited_tree_nodes.end()) {
            // plotPinName(feed, network->get_name(), source_tree_node);
            visited_tree_nodes.emplace(source_tree_node);
          }
          for (auto* sink_tree_node : source_tree_node->get_child_list()) {
            // plotPinName(feed, network->get_name(), sink_tree_node);
            visited_tree_nodes.emplace(sink_tree_node);
            tree_node_queue.push(sink_tree_node);

            // plot wire.
            feed << "PATH" << std::endl;
            feed << "LAYER " << layer_idx << std::endl;
            feed << "DATATYPE 0" << std::endl;
            feed << "WIDTH "
              << "160" << std::endl;
            feed << "XY" << std::endl;
            feed << source_tree_node->get_point().get_x() << " : " << source_tree_node->get_point().get_y() << std::endl;
            feed << sink_tree_node->get_point().get_x() << " : " << sink_tree_node->get_point().get_y() << std::endl;
            feed << "ENDEL" << std::endl;
          }
          tree_node_queue.pop();
        }
      }

      // print unrelative nets.
      layer_idx++;
      int64_t unrelative_wl = 0;
      for (auto* network : unrelative_nets) {
        // skip the clock net.
        if (_external_api->isClockNet(network->get_name())) {
          continue;
        }

        unrelative_wl += stwl_eval.obtainNetWirelength(network->get_network_id());
        auto* multi_tree = stwl_eval.obtainMultiTree(network);
        std::set<TreeNode*> visited_tree_nodes;
        std::queue<TreeNode*> tree_node_queue;
        tree_node_queue.push(multi_tree->get_root());
        while (!tree_node_queue.empty()) {
          auto* source_tree_node = tree_node_queue.front();
          if (visited_tree_nodes.find(source_tree_node) == visited_tree_nodes.end()) {
            // plotPinName(feed, network->get_name(), source_tree_node);
            visited_tree_nodes.emplace(source_tree_node);
          }
          for (auto* sink_tree_node : source_tree_node->get_child_list()) {
            // plotPinName(feed, network->get_name(), sink_tree_node);
            visited_tree_nodes.emplace(sink_tree_node);
            tree_node_queue.push(sink_tree_node);

            // plot wire.
            feed << "PATH" << std::endl;
            feed << "LAYER " << layer_idx << std::endl;
            feed << "DATATYPE 0" << std::endl;
            feed << "WIDTH "
              << "160" << std::endl;
            feed << "XY" << std::endl;
            feed << source_tree_node->get_point().get_x() << " : " << source_tree_node->get_point().get_y() << std::endl;
            feed << sink_tree_node->get_point().get_x() << " : " << sink_tree_node->get_point().get_y() << std::endl;
            feed << "ENDEL" << std::endl;
          }
          tree_node_queue.pop();
        }
      }

      // print relative net wirelength.
      LOG_INFO << "MOUDULE: " << pair.first << " Relative Net WL : " << relative_wl;

      // print unrelative net wirelength.
      LOG_INFO << "MOUDULE: " << pair.first << " Unrelative Net WL : " << unrelative_wl;
      layer_idx++;
    }

    feed << "ENDSTR" << std::endl;
    feed << "ENDLIB" << std::endl;
    file_stream << feed.str();
    feed.clear();
    file_stream.close();
  }

  void PLReporter::plotModuleStateForDebug(std::vector<std::string> special_inst_list, std::string path)
  {
    std::ofstream file_stream;
    file_stream.open(path);
    if (!file_stream.good()) {
      LOG_WARNING << "Cannot open file for module list !";
    }

    auto* ipl_layout = PlacerDBInst.get_layout();
    auto* ipl_design = PlacerDBInst.get_design();

    std::vector<Instance*> special_insts;
    for (std::string name : special_inst_list) {
      auto* inst = ipl_design->find_instance(name);
      special_insts.push_back(inst);
    }

    std::stringstream feed;
    feed << "HEADER 600" << std::endl;
    feed << "BGNLIB" << std::endl;
    feed << "LIBNAME ITDP_LIB" << std::endl;
    feed << "UNITS 0.001 1e-9" << std::endl;
    feed << "BGNSTR" << std::endl;
    feed << "STRNAME core" << std::endl;
    feed << "BOUNDARY" << std::endl;
    feed << "LAYER 0" << std::endl;
    feed << "DATATYPE 0" << std::endl;
    feed << "XY" << std::endl;

    auto core_shape = ipl_layout->get_core_shape();
    feed << core_shape.get_ll_x() << " : " << core_shape.get_ll_y() << std::endl;
    feed << core_shape.get_ur_x() << " : " << core_shape.get_ll_y() << std::endl;
    feed << core_shape.get_ur_x() << " : " << core_shape.get_ur_y() << std::endl;
    feed << core_shape.get_ll_x() << " : " << core_shape.get_ur_y() << std::endl;
    feed << core_shape.get_ll_x() << " : " << core_shape.get_ll_y() << std::endl;
    feed << "ENDEL" << std::endl;

    // print all instances.
    for (auto* inst : ipl_design->get_instance_list()) {
      feed << "BOUNDARY" << std::endl;
      feed << "LAYER 0" << std::endl;
      feed << "DATATYPE 0" << std::endl;
      feed << "XY" << std::endl;
      feed << inst->get_shape().get_ll_x() << " : " << inst->get_shape().get_ll_y() << std::endl;
      feed << inst->get_shape().get_ur_x() << " : " << inst->get_shape().get_ll_y() << std::endl;
      feed << inst->get_shape().get_ur_x() << " : " << inst->get_shape().get_ur_y() << std::endl;
      feed << inst->get_shape().get_ll_x() << " : " << inst->get_shape().get_ur_y() << std::endl;
      feed << inst->get_shape().get_ll_x() << " : " << inst->get_shape().get_ll_y() << std::endl;
      feed << "ENDEL" << std::endl;
    }

    // print special instances.
    for (auto* inst : special_insts) {
      feed << "BOUNDARY" << std::endl;
      feed << "LAYER 1" << std::endl;
      feed << "DATATYPE 0" << std::endl;
      feed << "XY" << std::endl;
      feed << inst->get_shape().get_ll_x() << " : " << inst->get_shape().get_ll_y() << std::endl;
      feed << inst->get_shape().get_ur_x() << " : " << inst->get_shape().get_ll_y() << std::endl;
      feed << inst->get_shape().get_ur_x() << " : " << inst->get_shape().get_ur_y() << std::endl;
      feed << inst->get_shape().get_ll_x() << " : " << inst->get_shape().get_ur_y() << std::endl;
      feed << inst->get_shape().get_ll_x() << " : " << inst->get_shape().get_ll_y() << std::endl;
      feed << "ENDEL" << std::endl;
    }

    feed << "ENDSTR" << std::endl;
    feed << "ENDLIB" << std::endl;
    file_stream << feed.str();
    feed.clear();
    file_stream.close();
  }

  void PLReporter::saveNetPinInfoForDebug(std::string path)
  {
    std::ofstream file_stream;
    file_stream.open(path);
    if (!file_stream.good()) {
      LOG_WARNING << "Cannot open file for net pin list info !";
    }

    std::vector<std::string> pin_name_list;
    for (auto* net : PlacerDBInst.get_design()->get_net_list()) {
      for (auto* pin : net->get_pins()) {
        pin_name_list.push_back(pin->get_name());
      }
    }

    std::sort(pin_name_list.begin(), pin_name_list.end());

    file_stream << "There are pins count : " << pin_name_list.size() + 1 << std::endl;
    file_stream << std::endl;

    for (std::string pin_name : pin_name_list) {
      file_stream << pin_name << std::endl;
    }
    file_stream.close();
  }

  void PLReporter::savePinListInfoForDebug(std::string path)
  {
    std::ofstream file_stream;
    file_stream.open(path);
    if (!file_stream.good()) {
      LOG_WARNING << "Cannot open file for pin list info !";
    }

    std::vector<std::string> pin_name_list;
    for (auto* pin : PlacerDBInst.get_design()->get_pin_list()) {
      pin_name_list.push_back(pin->get_name());
    }

    std::sort(pin_name_list.begin(), pin_name_list.end());

    file_stream << "There are pins count : " << pin_name_list.size() + 1 << std::endl;
    file_stream << std::endl;

    for (std::string pin_name : pin_name_list) {
      file_stream << pin_name << std::endl;
    }
    file_stream.close();
  }

  void PLReporter::reportWLInfo(std::ofstream& feed, std::string target_dir)
  {
    std::string output_dir = target_dir;
    std::string wl_detail_file = "wl_detail_report.txt";
    std::ofstream wl_detail_stream;
    wl_detail_stream.open(output_dir + "/" + wl_detail_file);
    if (!wl_detail_stream.good()) {
      LOG_WARNING << "Cannot open file for wl detail report !";
    }
    wl_detail_stream << "Generate the report at " << ieda::Time::getNowWallTime() << std::endl;

    auto report_tbl = _external_api->generateTable("table");
    (*report_tbl) << TABLE_HEAD;
    (*report_tbl)[0][0] = "Wirelength Info";
    (*report_tbl)[0][1] = "Value";

    // wl info
    int64_t total_hpwl = 0;
    int64_t max_hpwl = 0;
    int64_t total_stwl = 0;
    int64_t max_stwl = 0;
    int32_t constraint_hpwl = PlacerDBInst.get_placer_config()->get_buffer_config().get_max_wirelength_constraint();
    int32_t long_net_cnt = 0;

    auto* topo_manager = PlacerDBInst.get_topo_manager();
    HPWirelength hpwl_eval(topo_manager);
    SteinerWirelength stwl_eval(topo_manager);
    stwl_eval.updateAllNetWorkPointPair();

    for (auto* network : topo_manager->get_network_list()) {
      int64_t hpwl = hpwl_eval.obtainNetWirelength(network->get_network_id());
      int64_t stwl = stwl_eval.obtainNetWirelength(network->get_network_id());

      hpwl > max_hpwl ? max_hpwl = hpwl : max_hpwl;
      stwl > max_stwl ? max_stwl = stwl : max_stwl;
      total_hpwl += hpwl;
      total_stwl += stwl;

      if (hpwl > constraint_hpwl) {
        long_net_cnt++;
      }
    }

    (*report_tbl)[1][0] = "Total HPWL";
    (*report_tbl)[1][1] = std::to_string(total_hpwl);
    (*report_tbl)[2][0] = "Max HPWL";
    (*report_tbl)[2][1] = std::to_string(max_hpwl);
    (*report_tbl)[3][0] = "Total STWL";
    (*report_tbl)[3][1] = std::to_string(total_stwl);
    (*report_tbl)[4][0] = "Max STWL";
    (*report_tbl)[4][1] = std::to_string(max_stwl);
    (*report_tbl)[5][0] = ("LongNet HPWL (Exceed " + std::to_string(constraint_hpwl)) + ") Count";
    (*report_tbl)[5][1] = std::to_string(long_net_cnt);
    (*report_tbl) << TABLE_ENDLINE;
    feed << (*report_tbl).to_string() << std::endl;

    reportLongNetInfo(wl_detail_stream);
    wl_detail_stream.close();
    LOG_INFO << "Detail Long Net Wirelength Info Writed to "
      << "'" << output_dir << "'";
  }

  void PLReporter::reportSTWLInfo(std::ofstream& feed)
  {
    auto* topo_manager = PlacerDBInst.get_topo_manager();
    SteinerWirelength stwl_eval(topo_manager);
    stwl_eval.updateAllNetWorkPointPair();

    int64_t sum_stwl = 0;
    int64_t max_stwl = 0;

    for (auto* network : topo_manager->get_network_list()) {
      int64_t stwl = stwl_eval.obtainNetWirelength(network->get_network_id());
      stwl > max_stwl ? max_stwl = stwl : max_stwl;
      sum_stwl += stwl;
    }

    feed << "Total STWL : " << sum_stwl << std::endl;
    feed << "Max STWL : " << max_stwl << std::endl;
    feed << std::endl;
  }

  void PLReporter::printHPWLInfo()
  {
    auto* topo_manager = PlacerDBInst.get_topo_manager();
    HPWirelength hpwl_eval(topo_manager);
    LOG_INFO << "Current Stage Total HPWL: " << hpwl_eval.obtainTotalWirelength();
  }

  void PLReporter::printTimingInfo()
  {
    for (std::string clock_name : _external_api->obtainClockNameList()) {
      double early_wns = _external_api->obtainWNS(clock_name.c_str(), ista::AnalysisMode::kMin);
      double early_tns = _external_api->obtainTNS(clock_name.c_str(), ista::AnalysisMode::kMin);
      double late_wns = _external_api->obtainWNS(clock_name.c_str(), ista::AnalysisMode::kMax);
      double late_tns = _external_api->obtainTNS(clock_name.c_str(), ista::AnalysisMode::kMax);
      LOG_INFO << clock_name << " early_wns: " << std::to_string(early_wns)
        << " early_tns: " << std::to_string(early_tns)
        << " late_wns: " << std::to_string(late_wns)
        << " late_tns: " << std::to_string(late_tns);
    }

  }

  void PLReporter::reportHPWLInfo(std::ofstream& feed)
  {
    auto* topo_manager = PlacerDBInst.get_topo_manager();
    HPWirelength hpwl_eval(topo_manager);
    feed << "Total HPWL: " << hpwl_eval.obtainTotalWirelength() << std::endl;
    feed << std::endl;
  }

  void PLReporter::reportLongNetInfo(std::ofstream& feed)
  {
    auto* pl_config = PlacerDBInst.get_placer_config();
    int32_t max_wirelength_constraint = pl_config->get_buffer_config().get_max_wirelength_constraint();
    feed << "Report LongNet HPWL Exceed " << max_wirelength_constraint << std::endl;

    auto core_shape = PlacerDBInst.get_layout()->get_core_shape();
    int32_t core_width = core_shape.get_width();
    int32_t core_height = core_shape.get_height();
    int32_t long_width = max_wirelength_constraint;
    int32_t long_height = max_wirelength_constraint;
    auto* topo_manager = PlacerDBInst.get_topo_manager();

    int net_cnt = 0;
    for (auto* network : topo_manager->get_network_list()) {
      if (fabs(network->get_net_weight()) < 1e-7) {
        continue;
      }
      if (PlacerDBInst.get_placer_config()->isTimingEffort()) {
        if (_external_api->isClockNet(network->get_name())) {
          continue;
        }
      }

      auto shape = network->obtainNetWorkShape();
      int32_t network_width = shape.get_width();
      int32_t network_height = shape.get_height();

      if (network_width > long_width && network_height > long_height) {
        feed << "Net : " << network->get_name() << " Width/CoreWidth " << network_width << "/" << core_width << " Height/CoreHeight "
          << network_height << "/" << core_height << std::endl;
        ++net_cnt;
      }
      else if (network_width > long_width) {
        feed << "Net : " << network->get_name() << " Width/CoreWidth " << network_width << "/" << core_width << std::endl;
        ++net_cnt;
      }
      else if (network_height > long_height) {
        feed << "Net : " << network->get_name() << " Height/CoreHeight " << network_height << "/" << core_height << std::endl;
        ++net_cnt;
      }
    }

    feed << std::endl;
    feed << "SUMMARY : "
      << "AcrossLongNets / Total Nets = " << net_cnt << " / " << topo_manager->get_network_list().size() << std::endl;
    feed << std::endl;
  }

  int32_t PLReporter::reportOverlapInfo(std::ofstream& feed)
  {
    LayoutChecker* checker = new LayoutChecker(&PlacerDBInst);

    feed << "Overlap Violation" << std::endl;
    feed << "Regions In the Layout : " << std::endl;
    int32_t idx = 0;
    for (auto* region : PlacerDBInst.get_design()->get_region_list()) {
      feed << "Region " << idx++ << region->get_name() << std::endl;
      for (auto boundary : region->get_boundaries()) {
        feed << "Boundary : " << boundary.get_ll_x() << "," << boundary.get_ll_y() << " " << boundary.get_ur_x() << "," << boundary.get_ur_y()
          << std::endl;
      }
    }

    feed << std::endl;

    std::vector<std::vector<Instance*>> clique_list = checker->obtainOverlapInstClique();
    feed << "Illegal Overlap Instance Cliques Count : " << clique_list.size() << std::endl;
    for (size_t i = 0; i < clique_list.size(); i++) {
      feed << "Overlap Clique : " << i << std::endl;
      for (size_t j = 0; j < clique_list.at(i).size(); j++) {
        auto* inst = clique_list.at(i).at(j);
        feed << "Inst " << inst->get_name() << " Coordinate : " << inst->get_shape().get_ll_x() << "," << inst->get_shape().get_ll_y() << " "
          << inst->get_shape().get_ur_x() << "," << inst->get_shape().get_ur_y() << std::endl;
        if (clique_list.at(i).size() == 1) {
          feed << "Maybe overlap with blockage !" << std::endl;
        }
      }
    }

    delete checker;

    return static_cast<int32_t>(clique_list.size());
  }

  void PLReporter::reportTimingInfo(std::ofstream& feed)
  {
    auto report_tbl = _external_api->generateTable("table");
    (*report_tbl) << TABLE_HEAD;
    (*report_tbl)[0][0] = "Clock Timing Info";
    (*report_tbl)[0][1] = "Early WNS";
    (*report_tbl)[0][2] = "Early TNS";
    (*report_tbl)[0][3] = "Late WNS";
    (*report_tbl)[0][4] = "Late TNS";
    (*report_tbl) << TABLE_ENDLINE;

    for (std::string clock_name : _external_api->obtainClockNameList()) {
      double early_wns = _external_api->obtainWNS(clock_name.c_str(), ista::AnalysisMode::kMin);
      double early_tns = _external_api->obtainTNS(clock_name.c_str(), ista::AnalysisMode::kMin);
      double late_wns = _external_api->obtainWNS(clock_name.c_str(), ista::AnalysisMode::kMax);
      double late_tns = _external_api->obtainTNS(clock_name.c_str(), ista::AnalysisMode::kMax);
      (*report_tbl) << clock_name << std::to_string(early_wns) << std::to_string(early_tns) << std::to_string(late_wns)
        << std::to_string(late_tns) << TABLE_ENDLINE;
    }
    (*report_tbl) << TABLE_ENDLINE;
    feed << (*report_tbl).to_string() << std::endl;

    // _external_api->destroyTimingEval();
  }

  void PLReporter::reportCongestionInfo(std::ofstream& feed)
  {
    ieval::OverflowSummary overflow_summary = _external_api->evalproCongestion();

    auto report_tbl = _external_api->generateTable("table");
    (*report_tbl) << TABLE_HEAD;
    (*report_tbl)[0][0] = "Congestion Info";
    (*report_tbl)[1][0] = "Average Congestion of Edges";
    (*report_tbl)[1][1] = std::to_string(overflow_summary.weighted_average_overflow_union);
    (*report_tbl)[2][0] = "Total Overflow";
    (*report_tbl)[2][1] = std::to_string(overflow_summary.total_overflow_union);
    (*report_tbl)[3][0] = "Maximal Overflow";
    (*report_tbl)[3][1] = std::to_string(overflow_summary.max_overflow_union);
    (*report_tbl) << TABLE_ENDLINE;
    feed << (*report_tbl).to_string() << std::endl;
  }

  void PLReporter::reportPLBaseInfo(std::ofstream& feed)
  {
    auto* pl_layout = PlacerDBInst.get_layout();
    auto* pl_design = PlacerDBInst.get_design();

    auto report_tbl = _external_api->generateTable("table");
    (*report_tbl) << TABLE_HEAD;
    (*report_tbl)[0][0] = "Base Info";
    (*report_tbl)[0][1] = "Value";
    (*report_tbl)[1][0] = "Design";
    (*report_tbl)[1][1] = pl_design->get_design_name();
    (*report_tbl)[2][0] = "Utilization";
    (*report_tbl)[2][1] = std::to_string(PlacerDBInst.obtainUtilization());

    // core site info
    int64_t row_num = pl_layout->get_core_shape().get_height() / pl_layout->get_row_height();
    int64_t site_num = pl_layout->get_core_shape().get_width() / pl_layout->get_site_width();
    (*report_tbl)[3][0] = "Site Num";
    (*report_tbl)[3][1] = std::to_string(row_num) + " * " + std::to_string(site_num);

    // instance info
    int32_t instance_cnt = 0;
    int32_t macro_cnt = 0;
    int32_t stdcell_cnt = 0;
    int32_t flipflop_cnt = 0;
    int32_t clock_buf_cnt = 0;
    int32_t logic_cnt = 0;
    for (auto* inst : pl_design->get_instance_list()) {
      instance_cnt++;
      Cell* cell_master = inst->get_cell_master();
      if (cell_master) {
        if (cell_master->isMacro()) {
          macro_cnt++;
        }
        else {
          stdcell_cnt++;
          if (cell_master->isFlipflop()) {
            flipflop_cnt++;
          }
          else if (cell_master->isClockBuffer()) {
            clock_buf_cnt++;
          }
          else {
            logic_cnt++;
          }
        }
      }
    }
    (*report_tbl)[4][0] = "Instances Count";
    (*report_tbl)[4][1] = std::to_string(instance_cnt);
    (*report_tbl)[5][0] = "- Macro Count";
    (*report_tbl)[5][1] = std::to_string(macro_cnt);
    (*report_tbl)[6][0] = "- StdCell Count";
    (*report_tbl)[6][1] = std::to_string(stdcell_cnt);
    (*report_tbl)[7][0] = "-- FlipFlop Count";
    (*report_tbl)[7][1] = std::to_string(flipflop_cnt);
    (*report_tbl)[8][0] = "-- Clock Buffer Count";
    (*report_tbl)[8][1] = std::to_string(clock_buf_cnt);
    (*report_tbl)[9][0] = "-- Normal Logic Count";
    (*report_tbl)[9][1] = std::to_string(logic_cnt);

    // net info
    int32_t net_cnt = 0;
    int32_t signal_cnt = 0;
    int32_t clock_cnt = 0;
    int32_t reset_cnt = 0;
    int32_t other_cnt = 0;
    for (auto* net : pl_design->get_net_list()) {
      net_cnt++;
      if (net->isClockNet()) {
        clock_cnt++;
      }
      else if (net->isSignalNet()) {
        signal_cnt++;
      }
      else if (net->isResetNet()) {
        reset_cnt++;
      }
      else {
        other_cnt++;
      }
    }
    (*report_tbl)[10][0] = "Nets Count";
    (*report_tbl)[10][1] = std::to_string(net_cnt);
    (*report_tbl)[11][0] = "- Signal Net Count";
    (*report_tbl)[11][1] = std::to_string(signal_cnt);
    (*report_tbl)[12][0] = "- Clock Net Count";
    (*report_tbl)[12][1] = std::to_string(clock_cnt);
    (*report_tbl)[13][0] = "- Reset Net Count";
    (*report_tbl)[13][1] = std::to_string(reset_cnt);
    (*report_tbl)[14][0] = "- Other Net Count";
    (*report_tbl)[14][1] = std::to_string(other_cnt);
    (*report_tbl) << TABLE_ENDLINE;
    feed << (*report_tbl).to_string() << std::endl;
  }

  void PLReporter::reportEDAEvaluation() {
    int dbu = PlacerDBInst.get_layout()->get_database_unit();

    int inst_cnt, fix_inst_cnt, net_cnt, pin_cnt;
    std::string core_area;
    float place_density[3], pin_density[3];
    int64_t bin_number;
    std::string bin_size;
    int overflow_number;
    float overflow;
    float HPWL[3], STWL[3], GRWL[3]; // for gp,lg,dp
    int32_t egr_tof[3];
    int32_t egr_mof[3];
    float egr_ace[3];
    float tns[3], wns[3];
    float suggest_freq[3];
    float total_movement, max_movement;

    auto* pl_design = PlacerDBInst.get_design();
    auto* pl_layout = PlacerDBInst.get_layout();
    auto& gp_config = PlacerDBInst.get_placer_config()->get_nes_config();

    inst_cnt = pl_design->get_instances_range();
    fix_inst_cnt = 0;
    for (auto* inst : pl_design->get_instance_list()) {
      if (inst->isFixed()) {
        fix_inst_cnt++;
      }
    }
    net_cnt = pl_design->get_nets_range();
    pin_cnt = pl_design->get_pins_range();

    std::ostringstream core_area_ss;
    core_area_ss << std::fixed << std::setprecision(3);
    core_area_ss << pl_layout->get_core_shape().get_width() / 1.0 / dbu;
    core_area_ss << " * ";
    core_area_ss << pl_layout->get_core_shape().get_height() / 1.0 / dbu;
    core_area = core_area_ss.str();

    bin_number = gp_config.get_bin_cnt_x() * gp_config.get_bin_cnt_y();

    std::ostringstream bin_size_ss;
    bin_size_ss << std::fixed << std::setprecision(3);
    bin_size_ss << PlacerDBInst.bin_size_x / 1.0 / dbu;
    bin_size_ss << " * ";
    bin_size_ss << PlacerDBInst.bin_size_y / 1.0 / dbu;
    bin_size = bin_size_ss.str();

    overflow_number = PlacerDBInst.gp_overflow_number;
    overflow = PlacerDBInst.gp_overflow;
    for (int i = 0; i < 3; i++) {
      place_density[i] = PlacerDBInst.place_density[i];
      pin_density[i] = PlacerDBInst.pin_density[i];
      HPWL[i] = PlacerDBInst.PL_HPWL[i] / 1.0 / dbu;
      STWL[i] = PlacerDBInst.PL_STWL[i] / 1.0 / dbu;
      GRWL[i] = PlacerDBInst.PL_GRWL[i];
      egr_tof[i] = PlacerDBInst.egr_tof[i];
      egr_mof[i] = PlacerDBInst.egr_mof[i];
      egr_ace[i] = PlacerDBInst.egr_ace[i];
      tns[i] = PlacerDBInst.tns[i];
      wns[i] = PlacerDBInst.wns[i];
      suggest_freq[i] = PlacerDBInst.suggest_freq[i];
    }

    total_movement = PlacerDBInst.lg_total_movement / 1.0 / dbu;
    max_movement = PlacerDBInst.lg_max_movement / 1.0 / dbu;

    // print the target csv file
    std::string output_dir = "./evaluation_task/";
    std::string report_file = "evaluation_report.csv";
    std::ofstream report_stream;
    report_stream.open(output_dir + report_file, std::ios::app);
    if (!report_stream.good()) {
      LOG_WARNING << "Cannot open file for evaluation report ! ";
    }
    report_stream << std::fixed << std::setprecision(3);

    std::string design_name = PlacerDBInst.get_design()->get_design_name();
    double sta_update_runtime = PlacerDBInst.sta_update_time;

    report_stream << design_name << "," << inst_cnt << "," << fix_inst_cnt << "," << net_cnt << "," << pin_cnt << ","
      << core_area << "," << place_density[0] << "," << pin_density[0] << "," << bin_number << ","
      << bin_size << "," << overflow_number << "," << overflow << "," << HPWL[0] << ","
      << STWL[0] << "," << GRWL[0] << "," << egr_tof[0] << "," << egr_mof[0] << "," << egr_ace[0] << "," << tns[0] << "," << wns[0] << ","
      << suggest_freq[0] << "," << total_movement << "," << max_movement << "," << pin_density[1] << ","
      << HPWL[1] << "," << STWL[1] << "," << GRWL[1] << "," << egr_tof[1] << "," << egr_mof[1] << "," << egr_ace[1]  << "," << tns[1] << ","
      << wns[1] << "," << suggest_freq[1] << "," << inst_cnt << "," << fix_inst_cnt << "," << net_cnt << ","
      << pin_cnt << "," << core_area << "," << place_density[2] << "," << pin_density[2] << "," << HPWL[2] << ","
      << STWL[2] << "," << GRWL[2] << "," << egr_tof[2] << "," << egr_mof[2] << "," << egr_ace[2]  << "," << tns[2] << "," << wns[2] << ","
      << suggest_freq[2] << "," << sta_update_runtime << std::endl;

    report_stream.close();
  }

  void PLReporter::reportEDAFillerEvaluation() {
    int fix_inst_cnt, net_cnt, inst_cnt, filler_cnt;

    fix_inst_cnt = 0;
    for (auto* inst : PlacerDBInst.get_design()->get_instance_list()) {
      if (inst->isFixed()) {
        fix_inst_cnt++;
      }
    }
    inst_cnt = PlacerDBInst.get_design()->get_instance_list().size();
    filler_cnt = inst_cnt - PlacerDBInst.init_inst_cnt;

    // print the target csv file
    std::string output_dir = "./evaluation_task/";
    std::string report_file = "evaluation_filler_report.csv";
    std::ofstream report_stream;
    report_stream.open(output_dir + report_file, std::ios::app);
    if (!report_stream.good()) {
      LOG_WARNING << "Cannot open file for evaluation report ! ";
    }

    net_cnt = PlacerDBInst.get_design()->get_net_list().size();

    std::string design_name = PlacerDBInst.get_design()->get_design_name();
    report_stream << design_name << "," << fix_inst_cnt << "," << net_cnt << "," << inst_cnt << "," << filler_cnt << std::endl;

    report_stream.close();
  }

  void PLReporter::reportTDPEvaluation() {
    auto* pl_design = PlacerDBInst.get_design();
    auto* pl_layout = PlacerDBInst.get_layout();
    auto& gp_config = PlacerDBInst.get_placer_config()->get_nes_config();
    int32_t dbu = pl_layout->get_database_unit();

    std::string design_name = pl_design->get_design_name();
    int inst_cnt = pl_design->get_instances_range() + 1;
    int fix_inst_cnt = 0;
    for (auto* inst : pl_design->get_instance_list()) {
      if (inst->isFixed()) {
        fix_inst_cnt++;
      }
    }

    std::string clock_name = _external_api->obtainClockNameList().at(0);
    float preset_period = _external_api->obtainTargetClockPeriodNS(clock_name);
    float target_density = gp_config.get_target_density();
    int64_t bin_num = gp_config.get_bin_cnt_x() * gp_config.get_bin_cnt_y();
    float gp_overflow = PlacerDBInst.gp_overflow;
    float lg_total_movement = PlacerDBInst.lg_total_movement / 1.0 / dbu;
    float lg_max_movement = PlacerDBInst.lg_max_movement / 1.0 / dbu;

    float HPWL[3], STWL[3], pin_density[3], egr_tof[3], egr_mof[3], egr_ace[3], tns[3], wns[3], suggest_freq[3];
    for (int i = 0; i < 3; i++) {
      pin_density[i] = PlacerDBInst.pin_density[i];
      HPWL[i] = PlacerDBInst.PL_HPWL[i] / 1.0 / dbu;
      STWL[i] = PlacerDBInst.PL_STWL[i] / 1.0 / dbu;
      egr_tof[i] = PlacerDBInst.egr_tof[i];
      egr_mof[i] = PlacerDBInst.egr_mof[i];
      egr_ace[i] = PlacerDBInst.egr_ace[i];
      tns[i] = PlacerDBInst.tns[i];
      wns[i] = PlacerDBInst.wns[i];
      suggest_freq[i] = PlacerDBInst.suggest_freq[i];
    }

    // print the target csv file
    std::string output_dir = "./evaluation_task/";
    std::string report_file = "evaluation_non_tdp_report.csv";
    std::ofstream report_stream;
    report_stream.open(output_dir + report_file, std::ios::app);
    if (!report_stream.good()) {
      LOG_WARNING << "Cannot open file for evaluation report ! ";
    }
    report_stream << std::fixed << std::setprecision(3);

    report_stream << design_name << "," << inst_cnt << "," << fix_inst_cnt << "," << preset_period << ","
                  << target_density << "," << bin_num << "," << gp_overflow << "," << HPWL[0] << "," << STWL[0] << ","
                  << pin_density[0] << "," << egr_tof[0] << "," << egr_mof[0] << "," << egr_ace[0] << "," << tns[0] << "," << wns[0] << "," << suggest_freq[0] << ","
                  << lg_total_movement << "," << lg_max_movement << "," << HPWL[1] << "," << STWL[1] << "," << pin_density[1] << ","
                  << egr_tof[0] << "," << egr_mof[0] << "," << egr_ace[0] << "," << tns[1] << "," << wns[1] << "," << suggest_freq[1] << "," << HPWL[2] << "," << STWL[2] << ","
                  << pin_density[2] << "," << egr_tof[0] << "," << egr_mof[0] << "," << egr_ace[0] << "," << tns[2] << "," << wns[2] << "," << suggest_freq[2] << std::endl;
    
    report_stream.close();
  }

}  // namespace ipl