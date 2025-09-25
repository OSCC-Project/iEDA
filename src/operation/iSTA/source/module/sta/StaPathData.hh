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
/**
 * @file StaPath.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of timing path data.
 * @version 0.1
 * @date 2021-03-14
 */

#pragma once

#include <map>
#include <memory>
#include <vector>

#include "BTreeMap.hh"
#include "StaClock.hh"

namespace ista {

class StaPathEndIterator;
class StaPathGroupIterator;

/**
 * @brief The class of launch and capture clock pair for timing analysis.
 *
 */
class StaClockPair {
 public:
  StaClockPair(int setup_launch_clock_edge, int setup_capture_clock_edge,
               int hold_launch_clock_edge, int hold_capture_clock_edge);
  ~StaClockPair() = default;

  StaClockPair(const StaClockPair& orig) = default;
  StaClockPair& operator=(const StaClockPair& rhs) = default;

  StaClockPair(StaClockPair&& other) noexcept;
  StaClockPair& operator=(StaClockPair&& rhs) noexcept;

  int getSetupDiff() {
    return _setup_capture_clock_edge - _setup_launch_clock_edge;
  }

  int64_t getSetupLaunchClockEdge() const { return _setup_launch_clock_edge; }
  int64_t getSetupCaptureClockEdge() const { return _setup_capture_clock_edge; }

  int64_t getHoldLaunchClockEdge() const { return _hold_launch_clock_edge; }
  int64_t getHoldCaptureClockEdge() const { return _hold_capture_clock_edge; }

 private:
  int64_t _setup_launch_clock_edge;  // unit is ps, below is the same.
  int64_t _setup_capture_clock_edge;
  int64_t _hold_launch_clock_edge;
  int64_t _hold_capture_clock_edge;
};

/**
 * @brief The timing path data base class.
 *
 */
class StaPathData {
 public:
  StaPathData(StaPathDelayData* delay_data, StaClockData* launch_clock_data,
              StaClockData* capture_clock_data);
  virtual ~StaPathData() = default;
  virtual int64_t getArriveTime() = 0;
  virtual int64_t getRequireTime() = 0;
  virtual double getArriveTimeNs() = 0;
  virtual int getSlack() = 0;
  virtual int getSkew() = 0;
  virtual double getSlackNs() = 0;
  virtual std::stack<StaPathDelayData*> getPathDelayData() = 0;
  virtual unsigned reportPath(const char* rpt_file_name) = 0;
  AnalysisMode getDelayType() { return _delay_data->get_delay_type(); }

  StaPathDelayData* get_delay_data() { return _delay_data; }
  StaVertex* getEndVertex() { return _delay_data->get_own_vertex(); }
  StaClockData* get_launch_clock_data() { return _launch_clock_data; }
  StaClockData* get_capture_clock_data() { return _capture_clock_data; }

 private:
  StaPathDelayData* _delay_data;      //!< The data path data.
  StaClockData* _launch_clock_data;   //!< The launch clock path data.
  StaClockData* _capture_clock_data;  //!< The capture clock path data.

  FORBIDDEN_COPY(StaPathData);
};

/**
 * @brief The sequential timing path data.
 *
 */
class StaSeqPathData : public StaPathData {
 public:
  StaSeqPathData(StaPathDelayData* delay_data, StaClockData* launch_clock_data,
                 StaClockData* capture_clock_data, StaClockPair&& clock_pair,
                 std::optional<int> cppr, int constrain_value);
  ~StaSeqPathData() override = default;

  virtual unsigned isStaClockGatePathData() { return 0; }
  StaClock* get_capture_clock() { return _capture_clock; }
  int64_t getArriveTime() override;
  std::pair<int64_t, int64_t> getCellAndNetDelayOfArriveTime();
  int64_t getRequireTime() override;
  double getArriveTimeNs() override { return FS_TO_NS(getArriveTime()); }
  [[nodiscard]] auto get_cppr() const { return _cppr; }
  [[nodiscard]] int get_constrain_value() const { return _constrain_value; }
  void set_check_arc(StaArc* check_arc) { _check_arc = check_arc; }
  [[nodiscard]] auto* get_check_arc() const { return _check_arc; }

  void set_uncertainty(int uncertainty) { _uncertainty = uncertainty; }
  [[nodiscard]] auto get_uncertainty() const { return _uncertainty; }

  StaClockPair& get_clock_pair() { return _clock_pair; }
  int64_t getLaunchEdge();
  int64_t getCaptureEdge();
  int getSlack() override;
  int getSkew() override;
  double getSlackNs() override { return FS_TO_NS(getSlack()); }
  std::stack<StaPathDelayData*> getPathDelayData();
  unsigned reportPath(const char* rpt_file_name) override;

 private:
  StaClock*
      _capture_clock;  //!< The capture clock path data, we will group the
                       //!< sequential timing path data accord capture clock.
  StaClockPair _clock_pair;
  std::optional<int>
      _cppr;             //!< The common path pessimism recoverge, unit is fs.
  int _constrain_value;  //!< The setup/hold constrain value.

  StaArc* _check_arc = nullptr;  //!< The constraint arc.
  std::optional<int>
      _uncertainty;  //!< The clock uncertainty for analyze margin, unit is fs.

  FORBIDDEN_COPY(StaSeqPathData);
};

/**
 * @brief The clock gate path data.
 *
 */
class StaClockGatePathData : public StaSeqPathData {
 public:
  StaClockGatePathData(StaPathDelayData* delay_data,
                       StaClockData* launch_clock_data,
                       StaClockData* capture_clock_data,
                       StaClockPair&& clock_pair, std::optional<int> cppr,
                       int constrain_value);
  ~StaClockGatePathData() override = default;
  unsigned isStaClockGatePathData() override { return 1; }

 private:
  FORBIDDEN_COPY(StaClockGatePathData);
};

/**
 * @brief The output port set_output_delay timing analyze.
 *
 */
class StaPortSeqPathData : public StaSeqPathData {
 public:
  StaPortSeqPathData(StaPathDelayData* delay_data,
                     StaClockData* launch_clock_data,
                     StaClockData* capture_clock_data,
                     StaClockPair&& clock_pair, std::optional<int> cppr,
                     int constrain_value, Port* output_port);
  ~StaPortSeqPathData() override = default;

 private:
  Port* _output_port;
};

/**
 * @brief The timing path end, we will accord the end vertex to store the path.
 *
 */
class StaPathEnd {
 public:
  friend StaPathEndIterator;
  explicit StaPathEnd(StaVertex* end_vertex);
  ~StaPathEnd() = default;

  StaVertex* get_end_vertex() const { return _end_vertex; }

  StaPathEnd(StaPathEnd&& other) noexcept;
  StaPathEnd& operator=(StaPathEnd&& rhs) noexcept;

  unsigned insertPathData(StaPathData* seq_data);

  StaPathData* findPathData(StaPathDelayData* delay_data);

 private:
  StaVertex* _end_vertex;  //!< The timing path end vertex.
  std::vector<std::unique_ptr<StaPathData>>
      _max_timing_data;  //!< The max timing data such as setup analysis from
                         //!< the launch clock.

  std::vector<std::unique_ptr<StaPathData>>
      _min_timing_data;  //!< The min timing data such as hold analysis from the
                         //!< launch clock.

  std::map<StaPathDelayData*, StaPathData*>
      _delay_data_to_path_data;  //!< for speed up the delay data find the path
                                 //!< data.

  FORBIDDEN_COPY(StaPathEnd);
};

/**
 * @brief The path end data iterator.
 *
 */
class StaPathEndIterator {
 public:
  StaPathEndIterator(StaPathEnd* path_end, AnalysisMode analysis_mode);
  ~StaPathEndIterator() = default;
  bool hasNext();
  StaPathData* next();

 private:
  std::vector<std::unique_ptr<StaPathData>>::iterator _iter;
  std::vector<std::unique_ptr<StaPathData>>::iterator _end;
};

/**
 * @brief The timing path group.
 *
 */
class StaPathGroup {
 public:
  friend StaPathGroupIterator;
  StaPathGroup() = default;
  virtual ~StaPathGroup() = default;
  StaPathGroup(StaPathGroup&& other) noexcept;
  StaPathGroup& operator=(StaPathGroup&& rhs) noexcept;

  unsigned insertPathData(StaVertex* end_vertex, StaPathData* seq_data);
  StaPathEnd* findPathEndData(StaVertex* end_vertex) {
    if (auto it = _end_data.find(end_vertex); it != _end_data.end()) {
      return it->second.get();
    }
    return nullptr;
  }

 private:
  std::unordered_map<StaVertex*,
                     std::unique_ptr<StaPathEnd>>
      _end_data;  //!< The path data.

  FORBIDDEN_COPY(StaPathGroup);
};

/**
 * @brief The path group iterator.
 *
 */
class StaPathGroupIterator {
 public:
  explicit StaPathGroupIterator(StaPathGroup* path_group);
  ~StaPathGroupIterator() = default;
  bool hasNext();
  StaPathEnd* next();

 private:
  std::unordered_map<StaVertex*, std::unique_ptr<StaPathEnd>>::iterator _iter;
  std::unordered_map<StaVertex*, std::unique_ptr<StaPathEnd>>::iterator _end;
};

/**
 * @brief The sequential path group.
 *
 */
class StaSeqPathGroup : public StaPathGroup {
 public:
  StaSeqPathGroup() = default;
  explicit StaSeqPathGroup(StaClock* capture_clock);
  ~StaSeqPathGroup() override = default;
  StaSeqPathGroup(StaSeqPathGroup&& other) noexcept;
  StaSeqPathGroup& operator=(StaSeqPathGroup&& rhs) noexcept;
  virtual unsigned isStaClockGatePathGroup() { return 0; }
  StaClock* get_capture_clock() { return _capture_clock; }

 private:
  StaClock* _capture_clock;  //!< The capture clock group.

  FORBIDDEN_COPY(StaSeqPathGroup);
};

/**
 * The clock gate path group.
 */
class StaClockGatePathGroup : public StaSeqPathGroup {
 public:
  explicit StaClockGatePathGroup(const char* clock_group);
  ~StaClockGatePathGroup() override = default;
  StaClockGatePathGroup(StaClockGatePathGroup&& other) noexcept;
  StaClockGatePathGroup& operator=(StaClockGatePathGroup&& rhs) noexcept;

  unsigned isStaClockGatePathGroup() { return 1; }
  const char* get_clock_group() { return _clock_group.c_str(); }

 private:
  std::string _clock_group;
  FORBIDDEN_COPY(StaClockGatePathGroup);
};

/**
 * @brief The macro for iterator the path group, usage:
 * StaPathGroup* path_group;
 * StaPathEnd* path_end;
 * StaPathData* path_data;
 * AnalysisMode analysis_type;
 * FOREACH_PATH_GROUP_END(path_group, path_end)
 *  FOREACH_PATH_END_DATA(path_end, analysis_type, path_data)
 *  {
 *      do_something_for_path_data(path_data);
 *  }
 */
#define FOREACH_PATH_GROUP_END(path_group, path_end) \
  for (StaPathGroupIterator p(path_group);           \
       p.hasNext() ? path_end = p.next(), true : false;)

#define FOREACH_PATH_END_DATA(path_end, analysis_type, path_data) \
  for (StaPathEndIterator q(path_end, analysis_type);             \
       q.hasNext() ? path_data = q.next(), true : false;)

/**
 * @brief The unconstrained path group, which is input port to FF, FF to
 * output port, input port to output port.
 *
 */
class StaUnconstrainPathGroup : public StaPathGroup {};

}  // namespace ista
