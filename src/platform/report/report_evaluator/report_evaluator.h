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
#pragma once

#include <future>
#include <thread>

#include "IdbEnum.h"
#include "IdbInstance.h"
#include "ReportTable.hh"
#include "report_basic.h"

namespace iplf {

enum class ReportEvaluatorType
{
  kNone = 0,
  kWireLength,
  kCongestion,
};

class ReportEvaluator : public ReportBase
{
 public:
  explicit ReportEvaluator(const std::string& report_name) : ReportBase(report_name) {}
  std::shared_ptr<ieda::ReportTable> createWireLengthReport();
  std::shared_ptr<ieda::ReportTable> createCongestionReport();

 private:
  template <typename NET, typename FUNC>
  static auto computeWireLength(std::vector<NET*> nets, FUNC fptr, const int threads = 16);
  static auto CongStats(float threshold, float step, vector<float>& data);
};

class EvalWrapper
{
 public:
  EvalWrapper() = delete;
  template <typename TT, typename ST, typename F>
  static void wrapRange(std::vector<TT*>& target, const std::vector<ST*>& src, ssize_t begin, ssize_t end, F wrapper);
  template <typename TT, typename ST, typename F>
  static std::vector<TT*> parallelWrap(const std::vector<ST*>& source, F wrapper, int threads = 16);
};

/////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// T E M P L A T E S /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
template <typename TT, typename ST, typename F>
std::vector<TT*> EvalWrapper::parallelWrap(const std::vector<ST*>& source, F wrapper, const int threads)
{
  size_t vsize = source.size();
  std::vector<TT*> target(vsize, nullptr);
  size_t range = vsize / (threads - 1);
  vector<std::future<void>> results(threads - 1);
  for (int i = 0; i < threads - 1; ++i) {
    results[i]
        // std::launch::deferred
        = std::async(std::launch::async,
                     [i, range, &source, &target, &wrapper]() -> void { wrapRange(target, source, i * range, (i + 1) * range, wrapper); });
  }
  wrapRange(target, source, range * (threads - 1), vsize, wrapper);
  for (auto& result : results) {
    result.wait();
  }
  return target;
}

template <typename TT, typename ST, typename F>
void EvalWrapper::wrapRange(std::vector<TT*>& target, const std::vector<ST*>& src, ssize_t begin, ssize_t end, F wrapper)
{
  for (ssize_t i = begin; i < end; ++i) {
    target[i] = wrapper(src[i]);
  }
}

template <typename NET, typename FUNC>
auto ReportEvaluator::computeWireLength(std::vector<NET*> nets, FUNC fptr, const int threads)
{
  int64_t total_len = 0;
  int64_t max_len = 0;
  auto* max_net = nets[0];

  auto compute = [&nets, &fptr](size_t start, size_t end) {
    int64_t total_len = 0;
    NET* max_net = nets[start];
    int64_t max_len = (max_net->*fptr)();
    for (size_t i = start; i < end; ++i) {
      auto len = (nets[i]->*fptr)();
      total_len += len;
      if (len > max_len) {
        max_len = len;
        max_net = nets[i];
      }
    }
    return std::tuple{max_net, max_len, total_len};
  };

  std::vector<decltype(std::async(compute, 0, 0))> fut_arr;
  size_t nsize = nets.size();
  size_t step = nsize / (threads - 1);

  for (int i = 0; i < threads - 1; ++i) {
    fut_arr.push_back(std::async(compute, i * step, (i + 1) * step));
  }
  if ((threads - 1) * step < nsize) {
    fut_arr.push_back(std::async(compute, (threads - 1) * step, nsize - 1));
  }

  for (auto& fut : fut_arr) {
    auto [net, len, total] = fut.get();
    total_len += total;
    if (len > max_len) {
      max_len = len;
      max_net = net;
    }
  }
  return std::make_tuple(total_len, max_len, max_net);
}

}  // namespace iplf