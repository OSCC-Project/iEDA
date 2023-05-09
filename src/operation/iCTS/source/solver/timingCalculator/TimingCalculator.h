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
/**
 * @file TimingCalculator.h
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include <optional>
#include <vector>

#include "CtsCellLib.h"
#include "CtsConfig.h"
#include "CtsInstance.h"
#include "CtsPolygon.h"
#include "pgl.h"

namespace icts {
#define SCALE_FACTOR 1000

namespace bg = boost::geometry;
using bg_Point = boost::geometry::model::d2::point_xy<double>;
using bg_Segment = boost::geometry::model::linestring<bg_Point>;
using bg_Polygon = boost::geometry::model::polygon<bg_Point>;

enum class TimingNodeType
{
  kBuffer,
  kSteiner,
  kSink
};

enum class InsertType
{
  kNone,
  kMake,
  kMultiple,
  kTopInsert,
  kBreakWire
};

class TimingNode
{
 public:
  TimingNode() = default;
  ~TimingNode() = default;
  explicit TimingNode(CtsInstance* inst) : _inst(inst)
  {
    auto location = inst->get_location();
    _join_segment = Segment(location, location);
    _merge_region = Polygon({location});
    _id = CTSAPIInst.genId();
  }

  TimingNode(TimingNode* left, TimingNode* right) : _left(left), _right(right)
  {
    if (left) {
      left->set_parent(this);
    }
    if (right) {
      right->set_parent(this);
    }
    _inst = new CtsInstance("", "", CtsInstanceType::kSteinerPoint, Point(-1, -1));
    _type = TimingNodeType::kSteiner;
    _id = CTSAPIInst.genId();
  }

  TimingNode* copy(const bool& new_inst = false)
  {
    TimingNode* node = new TimingNode();
    node->_join_segment = _join_segment;
    node->_merge_region = _merge_region;
    node->_inst = _inst;
    if (new_inst) {
      _inst = new CtsInstance("", "", CtsInstanceType::kSteinerPoint, _inst->get_location());
    }
    node->_delay_min = _delay_min;
    node->_delay_max = _delay_max;
    node->_insertion_delay = _insertion_delay;
    node->_slew_in = _slew_in;
    node->_cap_out = _cap_out;
    node->_slew_constraint = _slew_constraint;
    node->_need_snake = _need_snake;
    node->_fanout = _fanout;
    node->_level = _level;
    node->_type = _type;
    node->_parent = _parent;
    node->_left = _left;
    node->_right = _right;
    node->_merged = _merged;
    node->_insert_type = _insert_type;
    node->_id = _id;
    _insert_type = InsertType::kNone;
    _id = CTSAPIInst.genId();
    return node;
  }

  // bool
  bool is_buffer() const { return _type == TimingNodeType::kBuffer; }
  bool is_steiner() const { return _type == TimingNodeType::kSteiner; }
  bool is_sink() const { return _type == TimingNodeType::kSink; }
  bool is_merged() const { return _merged; }
  // getter
  std::string get_name() const { return _inst->get_name(); }
  Point get_location() const { return _inst->get_location(); }
  Segment get_join_segment() const { return _join_segment; }
  Polygon get_merge_region() const { return _merge_region; }
  CtsInstance* get_inst() const { return _inst; }
  double get_delay_min() const { return _delay_min; }
  double get_delay_max() const { return _delay_max; }
  double get_insertion_delay() const { return _insertion_delay; }
  double get_slew_in() const { return _slew_in; }
  double get_cap_out() const { return _cap_out; }
  double get_slew_constraint() const { return _slew_constraint; }
  double get_need_snake() const { return _need_snake; }
  int getFanout() const { return _fanout; }
  int get_level() const { return _level; }
  TimingNodeType get_type() const { return _type; }
  TimingNode* get_parent() const { return _parent; }
  TimingNode* get_left() const { return _left; }
  TimingNode* get_right() const { return _right; }
  std::string get_cell_master() const { return _inst->get_cell_master(); }
  InsertType get_insert_type() const { return _insert_type; }
  int get_id() const { return _id; }
  double get_net_length() const { return _net_length; }
  // setter
  void set_name(const std::string& name)
  {
    if (_inst->get_name().empty()) {
      _inst->set_name(name);
    }
  }
  void set_location(const Point& location) { _inst->set_location(location); }
  void set_location(const int& x, const int& y) { _inst->set_location(Point(x, y)); }
  void set_join_segment(const Segment& join_segment) { _join_segment = join_segment; }
  void set_merge_region(const Polygon& merge_region) { _merge_region = merge_region; }
  void set_merge_region(const Segment& merge_region) { _merge_region = Polygon({merge_region.low(), merge_region.high()}); }
  void set_inst(CtsInstance* inst) { _inst = inst; }
  void set_delay_min(const double& delay_min) { _delay_min = delay_min; }
  void set_delay_max(const double& delay_max) { _delay_max = delay_max; }
  void set_insertion_delay(const double& insertion_delay) { _insertion_delay = insertion_delay; }
  void set_slew_in(const double& slew_in) { _slew_in = slew_in; }
  void set_cap_out(const double& cap_out) { _cap_out = cap_out; }
  void set_slew_constraint(const double& slew_constraint) { _slew_constraint = slew_constraint; }
  void set_need_snake(const double& need_snake) { _need_snake = need_snake; }
  void setFanout(const int& fanout) { _fanout = fanout; }
  void set_level(const int& level) { _level = level; }
  void set_type(const TimingNodeType& type)
  {
    _type = type;
    if (_type == TimingNodeType::kBuffer) {
      _inst->set_type(CtsInstanceType::kBuffer);
    } else if (_type == TimingNodeType::kSteiner) {
      _inst->set_type(CtsInstanceType::kSteinerPoint);
    } else if (_type == TimingNodeType::kSink) {
      _inst->set_type(CtsInstanceType::kSink);
    }
  }
  void set_parent(TimingNode* parent) { _parent = parent; }
  void set_left(TimingNode* left) { _left = left; }
  void set_right(TimingNode* right) { _right = right; }
  void set_cell_master(const std::string& cell_master) { _inst->set_cell_master(cell_master); }
  void set_merged() { _merged = true; }
  void reset_merged() { _merged = false; }
  void set_insert_type(const InsertType& insert_type)
  {
    LOG_FATAL_IF(_insert_type != InsertType::kNone) << "insert type is not none";
    _insert_type = insert_type;
  }
  void set_id(const int& id) { _id = id; }
  void set_net_length(const double& sub_length) { _net_length = sub_length; }

 private:
  Segment _join_segment;
  Polygon _merge_region;
  CtsInstance* _inst = nullptr;
  double _delay_min = 0;
  double _delay_max = 0;
  double _insertion_delay = 0;
  double _slew_in = 0;
  double _cap_out = 0;
  double _slew_constraint = 0;
  double _need_snake = 0;
  int _fanout = 1;
  int _level = 1;
  TimingNodeType _type = TimingNodeType::kSteiner;
  TimingNode* _parent = nullptr;
  TimingNode* _left = nullptr;
  TimingNode* _right = nullptr;
  bool _merged = false;
  InsertType _insert_type = InsertType::kNone;
  int _id = -1;
  double _net_length = 0.0;
};

class TimingCalculator
{
 public:
  TimingCalculator();
  ~TimingCalculator() = default;

  void set_skew_bound(const double& skew_bound) { _skew_bound = skew_bound; }
#if (defined PY_MODEL) && (defined USE_EXTERNAL_MODEL)
  void set_external_model(ModelBase* external_model) { _external_model = external_model; }
#endif

  // basic calc
  bool skewFeasible(TimingNode* s) const
  {
    return (s->get_delay_max() - s->get_delay_min()) < _skew_bound
           || std::fabs(s->get_delay_max() - s->get_delay_min() - _skew_bound) < std::numeric_limits<double>::epsilon();
  }

  double calcShortestLength(TimingNode* i, TimingNode* j) const;

  double calcFarthestLength(TimingNode* i, TimingNode* j) const;

  double calcCapLoad(TimingNode* k) const;

  double minSubSlewConstraint(TimingNode* k) const;

  double calcElmoreDelay(TimingNode* s, TimingNode* t) const;

  double calcTempElmoreDelay(TimingNode* s, TimingNode* t) const;

  double calcIdealSlew(TimingNode* s, TimingNode* t) const;

  double calcMaxIdealSlew(TimingNode* i, TimingNode* j) const;

  double endPointByZeroSkew(TimingNode* i, TimingNode* j, const std::optional<double>& init_delay_i = std::nullopt,
                            const std::optional<double>& init_delay_j = std::nullopt) const;

  std::tuple<int, double> calcEvenlyInsertNum(CtsCellLib* buf_lib, const double& length, const double& target_low_delay) const;

  std::vector<TimingNode*> screenNodes(const std::vector<TimingNode*>& nodes) const;

  Point guideCenter(const std::vector<TimingNode*>& nodes) const;

  int guideDist(const std::vector<TimingNode*>& nodes) const;

  std::pair<double, double> calcEndpointLoc(TimingNode* i, TimingNode* j, const double& skew_bound) const;

  std::pair<double, double> calcTargetSkewRange(TimingNode* k) const;

  // merge calc
  double cardanoMaxRealRoot(const double& a, const double& b, const double& c, const double& d) const;

  double calcMaxRealRoot(const std::vector<double>& coeffs) const;

  double calcFeasibleWirelengthBySlew(TimingNode* k, CtsCellLib* lib) const;

  int calcSlewDivenDist(TimingNode* k, const int& dist_limit) const;

  double calcBestSlewEP(TimingNode* i, TimingNode* j) const;

  double calcMergeWireSlew(TimingNode* i, TimingNode* j) const;

  double calcMergeCost(TimingNode* i, TimingNode* j) const;

  TimingNode* calcMergeNode(TimingNode* i, TimingNode* j) const;

  void mergeNode(TimingNode* k) const;

  void calcMergeRegion(Polygon& merge_region, TimingNode* i, TimingNode* j) const;

  void calcBestSlewMergeRegion(Polygon& merge_region, TimingNode* i, TimingNode* j) const;

  bool balanceTiming(TimingNode* k) const;

  void fixTiming(TimingNode* k) const;

  void connect(TimingNode* parent, TimingNode* left, TimingNode* right) const;

  void insertConnect(TimingNode* i, TimingNode* insert, TimingNode* j) const;

  TimingNode* genSteinerNode() const;

  TimingNode* genBufferNode(const std::string& cell_master = "") const;

  // timing update
  void updateTiming(TimingNode* k, const bool& update_cap = true, const bool& update_delay = true,
                    const bool& update_slew_constraint = true) const;

  void updateCap(TimingNode* k) const;

  void updateDelay(TimingNode* k) const;

  void timingPropagate(TimingNode* k, const bool& propagate_head = true) const;

  void wireSnaking(TimingNode* s, TimingNode* i, const double& incre_delay) const;

  void makeBuffer(TimingNode* k) const;

  void insertBuffer(TimingNode* k, std::optional<Point> guide_point = std::nullopt, const int& guide_dist = 0) const;

  void insertBuffer(TimingNode* s, TimingNode* t) const;

  void insertBuffer(TimingNode* s, TimingNode* t, const double& incre_delay) const;

  CtsCellLib* findLib(TimingNode* k = nullptr) const;

  void setMinCostCellLib(TimingNode* k) const;

  CtsCellLib* findFeasibleLib(TimingNode* k) const;

  bool checkSlew(TimingNode* k, CtsCellLib* lib) const;

  std::tuple<CtsCellLib*, int> findMinCostEvenlyCellLib(const double& length, const double& target_low_delay,
                                                        const double& target_high_delay) const;

  void breakLongWire(TimingNode* s, TimingNode* t) const;

  double predictSlewIn(TimingNode* k) const;

  double predictInsertDelay(TimingNode* k, CtsCellLib* lib) const;

  // polygon calc
  Point bgToPglPoint(const bg_Point& p) const;

  Segment bgToPglSegment(const bg_Segment& s) const;

  Polygon bgToPglPolygon(const bg_Polygon& p) const;

  bg_Point pglToBgPoint(const Point& p) const;

  bg_Segment pglToBgSegment(const Segment& s) const;

  bg_Polygon pglToBgPolygon(const Polygon& p) const;

  std::vector<Point> intersectionPointByBg(const Polygon& poly_a, const Polygon& poly_b) const;

  std::vector<Point> intersectionPointByBg(const Polygon& poly, const Segment& seg) const;

  Point intersectionPointByBg(const Segment& seg_a, const Segment& seg_b) const;

  Polygon intersectionByBg(const Polygon& poly_a, const Polygon& poly_b) const;

  Polygon intersectionByBg(const Polygon& poly, const Segment& seg) const;

  void lineJoinSegment(TimingNode* i, TimingNode* j) const;

  void joinSegment(TimingNode* i, TimingNode* j) const;

  Segment intersectJS(const Segment& js_i, const Segment& js_j, const int& radius_by_j) const;

  void calcSDR(Polygon& sdr, const Segment& seg_i, const Segment& seg_j) const;

 private:
  double _unit_cap = 0.0;
  double _unit_res = 0.0;
  int _sink_num = 0;
  double _skew_bound = 0.0;
  int _db_unit = 0;
  double _max_buf_tran = 0.0;
  double _max_sink_tran = 0.0;
  double _max_cap = 0.0;
  int _max_fanout = 0;
  double _max_length = 0;
  double _min_insert_delay = 0.0;
  std::vector<icts::CtsCellLib*> _delay_libs;
#if (defined PY_MODEL) && (defined USE_EXTERNAL_MODEL)
  ModelBase* _external_model = nullptr;
#endif
};

}  // namespace icts