#include "manager.hpp"

#include <iostream>

#include "DBWrapper.hpp"

namespace eval {

Manager& Manager::initInst(Config* config)
{
  if (_mg_instance == nullptr) {
    _mg_instance = new Manager(config);
  }
  return *_mg_instance;
}

Manager::Manager(Config* config)
{
  bool is_db_wrapper = config->get_db_config().enable_wrapper();
  bool is_wirelength_eval = config->get_wl_config().enable_eval();
  bool is_congestion_eval = config->get_cong_config().enable_eval();
  bool is_gds_wrapper_eval = config->get_gds_wrapper_config().enable_eval();

  if (is_db_wrapper == true) {
    DBWrapper* idb_wrapper = new DBWrapper(config);
    EvalDB* eval_database = idb_wrapper->get_eval_db();
    if (is_wirelength_eval == true) {
      _wirelength_eval = new WirelengthEval();
      _wirelength_eval->set_net_list(eval_database->get_design()->get_wl_net_list());
      _wirelength_eval->set_name2net_map(eval_database->get_design()->get_name_net_map());
    }
    if (is_congestion_eval == true) {
      _congestion_eval = new CongestionEval();
      _congestion_eval->set_cong_grid(eval_database->get_layout()->get_cong_grid());
      _congestion_eval->set_cong_inst_list(eval_database->get_design()->get_cong_inst_list());
      _congestion_eval->set_cong_net_list(eval_database->get_design()->get_cong_net_list());
    }
    if (is_gds_wrapper_eval == true) {
      _gds_wrapper = new GDSWrapper();
      _gds_wrapper->set_net_list(eval_database->get_design()->get_gds_net_list());
    }
  }

  else {
    if (is_wirelength_eval == true) {
      _wirelength_eval = new WirelengthEval();
    }
    if (is_congestion_eval == true) {
      _congestion_eval = new CongestionEval();
    }
  }
}

Manager& Manager::getInst()
{
  if (_mg_instance == nullptr) {
    std::cout << "The instance not initialized!";
  }
  return *_mg_instance;
}

void Manager::destroyInst()
{
  if (_mg_instance != nullptr) {
    delete _mg_instance;
    _mg_instance = nullptr;
  }
}

Manager* Manager::_mg_instance = nullptr;

}  // namespace eval
