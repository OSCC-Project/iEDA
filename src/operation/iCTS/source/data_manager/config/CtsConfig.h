#pragma once

#include <string>
#include <vector>

using std::string;
using std::vector;

namespace icts {
class CtsConfig {
 public:
  CtsConfig() {}
  CtsConfig(const CtsConfig &rhs) = default;
  ~CtsConfig() = default;

  int get_micron_dbu() const { return _micron_dbu; }
  double get_skew_bound() const { return _skew_bound; }
  double get_max_buf_tran() const { return _max_buf_tran; }
  double get_max_sink_tran() const { return _max_sink_tran; }
  double get_max_cap() const { return _max_cap; }
  int get_max_fanout() const { return _max_fanout; }
  double get_max_length() const { return _max_length; }

  const string &get_router_type() const { return _router_type; }
  const string &get_delay_type() const { return _delay_type; }
  const string &get_cluster_type() const { return _cluster_type; }
  int get_cluster_size() const { return _cluster_size; }
  const string &get_sta_workspace() const { return _sta_workspace; }
  const string &get_output_def_path() const { return _output_def_path; }
  const string &get_log_file() const { return _log_file; }
  vector<string> get_buffer_types() const { return _buffer_types; }
  vector<int> get_routing_layers() const { return _routing_layers; }
  const string &get_gds_file() const { return _gds_file; }
  const string &get_use_netlist_string() const { return _gds_file; }
  bool is_use_netlist() { return _use_netlist == "ON" ? true : false; }
  const vector<std::pair<string, string>> get_clock_netlist() const {
    return _net_list;
  }
  const vector<std::pair<string, string>> get_external_models() const {
    return _external_models;
  }

  void set_micron_dbu(int micron_dbu) { _micron_dbu = micron_dbu; }
  void set_skew_bound(double skew_bound) { _skew_bound = skew_bound; }
  void set_max_buf_tran(double max_buf_tran) { _max_buf_tran = max_buf_tran; }
  void set_max_sink_tran(double max_sink_tran) {
    _max_sink_tran = max_sink_tran;
  }
  void set_max_cap(double max_cap) { _max_cap = max_cap; }
  void set_max_fanout(int max_fanout) { _max_fanout = max_fanout; }
  void set_max_length(double max_length) { _max_length = max_length; }

  void set_router_type(const string &router_type) {
    _router_type = router_type;
  }
  void set_delay_type(const string &delay_type) { _delay_type = delay_type; }
  void set_cluster_type(const string &cluster_type) {
    _cluster_type = cluster_type;
  }
  void set_cluster_size(int size) { _cluster_size = size; }
  void set_sta_workspace(const string &sta_workspace) {
    _sta_workspace = sta_workspace;
  }
  void set_output_def_path(const string &output_def_path) {
    _output_def_path = output_def_path;
  }
  void set_log_file(const string &file) { _log_file = file; }
  void set_buffer_types(const vector<string> &types) { _buffer_types = types; }
  void set_routing_layers(const vector<int> &routing_layers) {
    _routing_layers = routing_layers;
  }
  void set_gds_file(const string &file) { _gds_file = file; }
  void set_use_netlist(const string &use_netlist) {
    _use_netlist = use_netlist;
  }
  void set_netlist(const vector<std::pair<string, string>> &net_list) {
    _net_list = net_list;
  }
  void set_external_models(
      const vector<std::pair<string, string>> &external_models) {
    _external_models = external_models;
  }

 private:
  // units
  int _micron_dbu = 0;
  // algorithm
  string _router_type = "ZST";
  string _delay_type = "elmore";
  double _skew_bound = 0.04;
  double _max_buf_tran = 1.5;
  double _max_sink_tran = 1.5;
  double _max_cap = 1.5;
  int _max_fanout = 32;
  double _max_length = 300;
  string _cluster_type = "kmeans";
  int _cluster_size = 32;
  // file
  string _sta_workspace = "./result/cts";
  string _output_def_path = "./result/cts";
  string _log_file = "./result/cts/cts.log";
  string _gds_file = "./result/cts/cts.gds";
  vector<string> _buffer_types;
  vector<int> _routing_layers;
  string _use_netlist = "OFF";
  vector<std::pair<string, string>> _net_list;
  vector<std::pair<string, string>> _external_models;
};
}  // namespace icts