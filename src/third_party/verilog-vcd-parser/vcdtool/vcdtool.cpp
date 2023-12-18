
#include "VCDFileParser.hpp"
#include "cxxopts.hpp"
// #include "gitversion.h"
// #include <gperftools/profiler.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>

void print_scope_signals(VCDFile* trace, VCDScope* scope,
                         std::string local_parent,

                         std::FILE* csv_fp = nullptr,
                         std::FILE* log_fp = nullptr) {
  for (VCDSignal* signal : scope->signals) {
    auto ref_name = local_parent + "." + signal->reference;
    std::cout << ref_name << "\n";
  }
}

void traverse_scope(std::string parent, VCDFile* trace, VCDScope* scope,
                    bool instances, bool fullpath,
                    std::deque<std::string>& scope_queue,
                    std::FILE* csv_fp = nullptr, std::FILE* log_fp = nullptr) {
  std::string local_parent = parent;

  if (parent.length()) {
    local_parent += "/";
  }

  local_parent += scope->name;
  if (instances) {
    std::cout << "Scope: " << local_parent << std::endl;
  }
  if (fullpath && scope_queue.empty()) {
    print_scope_signals(trace, scope, local_parent, csv_fp, log_fp);
  }
  bool empty = scope_queue.empty();
  for (auto* child : scope->children) {
    bool go_deep = false;
    if (empty && scope_queue.empty()) {
      go_deep = true;
    } else if (child->name == scope_queue.front()) {
      scope_queue.pop_front();
      go_deep = true;
    }
    if (go_deep) {
      traverse_scope(local_parent, trace, child, instances, fullpath,
                     scope_queue, csv_fp, log_fp);
    }
  }
}
/*!
@brief Standalone test function to allow testing of the VCD file parser.
*/
int main(int argc, char** argv) {
  // ProfilerStart("test_capture.prof");
  cxxopts::Options options("vcdtoggle",
                           "Count number of signal toggles in VCD file");
  options.add_options()("h,help", "show help message")(
      "v,version", "Show version")("i,instances", "Show only instances")(
      "r,header", "Show header")("u,fullpath", "Show full signal path")(
      "begin", "Start time (default to 0)", cxxopts::value<VCDTime>())(
      "end", "End time (default to end of file)", cxxopts::value<VCDTime>())(
      "s,scope", "Select the scope for which statistics are to be collected",
      cxxopts::value<std::string>())(
      "f,file", "filename containing scopes and signal name regex",
      cxxopts::value<std::string>())(
      "positional", "Positional pareameters",
      cxxopts::value<std::vector<std::string>>())(
      "o,output_path", "Output csv path", cxxopts::value<std::string>());
  options.parse_positional({"positional"});
  auto result = options.parse(argc, argv);

  if (result["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
  }

  if (result["version"].as<bool>()) {
    std::cout << "Version:  0.1\nGit HEAD: " << std::endl;
  }

  std::string infile(
      result["positional"].as<std::vector<std::string>>().back());

  VCDFileParser parser;
  auto parse_start = std::chrono::steady_clock::now();
  VCDFile* trace = parser.parse_file(infile);
  auto parse_end = std::chrono::steady_clock::now();
  auto parse_diff =
      std::chrono::duration_cast<std::chrono::seconds>(parse_end - parse_start);
  if (result.count("begin")) {
    // parser.start_time = 0;
    trace->start_time = result["begin"].as<VCDTime>() / trace->time_resolution;
  } else {
    trace->start_time = 0;
  }

  if (result.count("end")) {
    // parser.end_time = result["end"].as<VCDTime>();
    trace->end_time = result["end"].as<VCDTime>() / trace->time_resolution;
  } else {
    trace->end_time = trace->get_timestamps()->back();
  }
  bool instances = result["instances"].as<bool>();
  bool fullpath = result["fullpath"].as<bool>();
  if (trace) {
    if (result["header"].as<bool>()) {
      std::cout << "Version:         " << trace->version << std::endl;
      std::cout << "Comment:         " << trace->comment << std::endl;
      std::cout << "Date:            " << trace->date << std::endl;
      std::cout << "Signal count:    " << trace->get_signals()->size()
                << std::endl;
      std::cout << "Times Recorded:  " << trace->get_timestamps()->size()
                << std::endl;
      std::cout << "Parser Run Time: " << parse_diff.count() << "s"
                << std::endl;
      // if (fullpath) {
      //   std::cout << "Hash\tFull signal path\n";
      // }
    }
    // Print out every signal in every scope.
    std::deque<std::string> scope_queue;
    if (result.count("scope")) {
      auto target_scope = result["scope"].as<std::string>();
      // scope_queue = split_scope(target_scope);
    }
    // init warning.log
    std::string log_file = "./warning.log";
    std::FILE* log_fp = fopen(log_file.c_str(), "w");
    fclose(log_fp);
    log_fp = fopen(log_file.c_str(), "a");
    auto statistic_start = std::chrono::steady_clock::now();
    if (result.count("output_path")) {
      std::string out_file = result["output_path"].as<std::string>();
      std::FILE* csv_fp = fopen(out_file.c_str(), "w");
      fclose(csv_fp);
      csv_fp = fopen(out_file.c_str(), "a");
      traverse_scope(std::string(""), trace, trace->root_scope, instances,
                     fullpath, scope_queue, csv_fp, log_fp);
      std::cout << "Output CSV:    " << out_file << std::endl;
      fclose(csv_fp);
    } else {
      traverse_scope(std::string(""), trace, trace->root_scope, instances,
                     fullpath, scope_queue, nullptr, log_fp);
    }
    fclose(log_fp);
    auto statistic_end = std::chrono::steady_clock::now();
    auto statistic_diff = std::chrono::duration_cast<std::chrono::seconds>(
        statistic_end - statistic_start);
    std::cout << "Statistics Time: " << statistic_diff.count() << "s"
              << std::endl;
    delete trace;

    return 0;
  } else {
    std::cout << "Parse Failed." << std::endl;
    return 1;
  }
  // ProfilerStop();
}
