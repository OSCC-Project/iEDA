/**
 * @file HCTS.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "HCTS.h"

#include <filesystem>
#include <queue>
#include <random>
#include <vector>

#include "CtsReport.h"
#include "time/Time.hh"
namespace icts {
void HCTS::run()
{
  LOG_INFO << "#################### [HCTS] ####################";
  LOG_INFO << "[HCTS] build net: " << _net_name;
  LOG_INFO << "[HCTS] Flip-Flops num: " << _instances.size();
  CTSAPIInst.saveToLog("#################### [HCTS] ####################");
  CTSAPIInst.saveToLog("[HCTS] build net: ", _net_name);
  CTSAPIInst.saveToLog("[HCTS] Flip-Flops num: ", _instances.size());
  _root = biCluster(_instances);  // make topo & embed location
  heuristicBuffering();
  timingPropagation(_root);
  makeTopos();
  LOG_INFO << "[HCTS] Insert Buffer num: " << _clock_topos.size();
  CTSAPIInst.saveToLog("[HCTS] Insert Buffer num: ", _clock_topos.size());
#ifdef TIMING_LOG
  reportTiming();
#endif
}
// func

HNode* HCTS::biCluster(const std::vector<CtsInstance*>& insts) const
{
  if (insts.size() == 1) {
    auto* node = new HNode(insts[0]);
    auto* config = CTSAPIInst.get_config();
    auto max_sink_tran = config->get_max_sink_tran();
    auto sink_cap = CTSAPIInst.getSinkCap(insts[0]);
    node->set_type(HNodeType::kSink);
    node->set_cap_load(sink_cap);
    node->set_sub_total_cap(sink_cap);
    return node;
  }
  HNode* left = nullptr;
  HNode* right = nullptr;
  if (insts.size() == 2) {
    left = biCluster(std::vector<CtsInstance*>{insts[0]});
    right = biCluster(std::vector<CtsInstance*>{insts[1]});
  } else {
    auto clusters = kMeans(insts, 2);
    left = biCluster(clusters[0]);
    right = biCluster(clusters[1]);
  }
  auto* parent = new HNode(left, right);
  // 1. set loc

  //   auto loc = medianCenter(insts);
  //   parent->setLocation(loc);
  updateCapCenterLoc(parent);
  // 2. set sub total cap
  updateSubTotalCap(parent);

  return parent;
}

std::vector<std::vector<CtsInstance*>> HCTS::kMeans(const std::vector<CtsInstance*>& instances, const size_t& k,
                                                    const size_t& max_iter) const
{
  std::vector<CtsPoint<int64_t>> centers;
  size_t num_instances = instances.size();
  std::vector<int> assignments(num_instances);

  // Randomly choose first center from instances
  // std::random_device rd;
  // std::mt19937 gen(rd());
  std::mt19937::result_type seed = 0;
  std::mt19937 gen(seed);
  std::uniform_int_distribution<> dis(0, num_instances - 1);
  centers.push_back(instances[dis(gen)]->get_location());
  // Choose k-1 remaining centers using kmeans++ algorithm
  while (centers.size() < k) {
    std::vector<double> distances(num_instances, std::numeric_limits<double>::max());
    for (size_t i = 0; i < num_instances; i++) {
      CtsPoint<int64_t> instance_location = instances[i]->get_location();
      double min_distance = std::numeric_limits<double>::max();
      for (size_t j = 0; j < centers.size(); j++) {
        double distance = calcManhattanDist(instance_location, centers[j]);
        min_distance = std::min(min_distance, distance);
      }
      distances[i] = min_distance * min_distance;  // square distance
    }

    std::discrete_distribution<> distribution(distances.begin(), distances.end());
    int selected_index = distribution(gen);
    centers.emplace_back(instances[selected_index]->get_location());
  }

  size_t num_iterations = 0;
  double prev_cap_variance = std::numeric_limits<double>::max();
  while (num_iterations++ < max_iter) {
    // Assignment step
    for (size_t i = 0; i < num_instances; i++) {
      CtsPoint<int64_t> instance_location = instances[i]->get_location();
      double min_distance = std::numeric_limits<double>::max();
      int min_center_index = -1;
      for (size_t j = 0; j < centers.size(); j++) {
        double distance = calcManhattanDist(instance_location, centers[j]);
        if (distance < min_distance) {
          min_distance = distance;
          min_center_index = j;
        }
      }
      assignments[i] = min_center_index;
    }
    // Check cap variance
    std::vector<double> cluster_cap(k, 0);
    for (size_t i = 0; i < num_instances; i++) {
      int center_index = assignments[i];
      cluster_cap[center_index] += CTSAPIInst.getSinkCap(instances[i]);
    }
    double sum = std::accumulate(std::begin(cluster_cap), std::end(cluster_cap), 0.0);
    double mean = sum / cluster_cap.size();
    double cap_variance = 0.0;
    std::for_each(std::begin(cluster_cap), std::end(cluster_cap), [&](const double d) { cap_variance += pow(d - mean, 2); });
    cap_variance /= cluster_cap.size();
    // Check for convergence
    if (cap_variance < prev_cap_variance) {
      break;
    }
    prev_cap_variance = std::min(prev_cap_variance, cap_variance);
    // Update step
    std::vector<CtsPoint<int64_t>> new_centers(k, CtsPoint<int64_t>(0, 0));
    std::vector<int> center_counts(k, 0);
    for (size_t i = 0; i < num_instances; i++) {
      Point instance_location = instances[i]->get_location();
      int center_index = assignments[i];
      new_centers[center_index] += instance_location;
      center_counts[center_index]++;
    }
    for (size_t i = 0; i < k; i++) {
      if (center_counts[i] > 0) {
        new_centers[i] /= center_counts[i];
      }
    }
    centers = new_centers;
  }
  // Collect results
  std::vector<std::vector<CtsInstance*>> clusters(k);
  for (size_t i = 0; i < num_instances; i++) {
    int center_index = assignments[i];
    clusters[center_index].push_back(instances[i]);
  }

  return clusters;
}

HNode* HCTS::biPartition(const std::vector<CtsInstance*>& instances) const
{
    // TBD
  return nullptr;
}

Point HCTS::medianCenter(const std::vector<CtsInstance*>& instances) const
{
  if (instances.empty()) {
    LOG_FATAL << "Empty instances";
  }
  if (instances.size() == 1) {
    return instances[0]->get_location();
  }
  if (instances.size() == 2) {
    return meanCenter(instances);
  }
  std::vector<int> x_coords;
  std::vector<int> y_coords;
  for (auto* inst : instances) {
    x_coords.push_back(inst->get_location().x());
    y_coords.push_back(inst->get_location().y());
  }
  std::sort(x_coords.begin(), x_coords.end());
  std::sort(y_coords.begin(), y_coords.end());
  int x_median = x_coords[x_coords.size() / 2];
  int y_median = y_coords[y_coords.size() / 2];
  return Point(x_median, y_median);
}

Point HCTS::meanCenter(const std::vector<CtsInstance*>& instances) const
{
  if (instances.empty()) {
    LOG_FATAL << "Empty instances";
  }
  int x_sum = 0;
  int y_sum = 0;
  for (auto* inst : instances) {
    x_sum += inst->get_location().x();
    y_sum += inst->get_location().y();
  }
  return Point(x_sum / instances.size(), y_sum / instances.size());
}

void HCTS::netPropagation(HNode* node) const
{
}

void HCTS::timingPropagation(HNode* root) const
{
  fanoutPropagation(root);
  netLengthPropagation(root);
  capPropagation(root);
  slewPropagation(root);
}

void HCTS::netLengthPropagation(HNode* node) const
{
  // bottom up propagate net length
  auto* left = node->get_left();
  auto* right = node->get_right();
  if (!left && !right) {
    return;
  }
  double net_length = 0;
  if (left) {
    netLengthPropagation(left);
    auto length = calcLength(node, left);
    net_length += left->isSteiner() ? length + left->get_net_length() : length;
  }
  if (right) {
    netLengthPropagation(right);
    auto length = calcLength(node, right);
    net_length += right->isSteiner() ? length + right->get_net_length() : length;
  }
  node->set_net_length(net_length);
}

void HCTS::fanoutPropagation(HNode* node) const
{
  // bottom up propagate fanout
  auto* left = node->get_left();
  auto* right = node->get_right();
  if (!left && !right) {
    return;
  }
  size_t fanout = 0;
  if (left) {
    fanoutPropagation(left);
    fanout += left->isSteiner() ? left->get_fanout() : 1;
  }
  if (right) {
    fanoutPropagation(right);
    fanout += right->isSteiner() ? right->get_fanout() : 1;
  }
  node->set_fanout(fanout);
}

void HCTS::capPropagation(HNode* node) const
{
  // bottom up propagate cap_out and cap_load
  auto* left = node->get_left();
  auto* right = node->get_right();
  if (!left && !right) {
    return;
  }
  double cap_out = 0.0;
  if (left) {
    capPropagation(left);
    cap_out += _unit_cap * calcLength(node, left) + left->get_cap_load();
  }
  if (right) {
    capPropagation(right);
    cap_out += _unit_cap * calcLength(node, right) + right->get_cap_load();
  }
  node->set_cap_out(cap_out);
  if (node->isSteiner()) {
    node->set_cap_load(cap_out);
  }
}

void HCTS::slewPropagation(HNode* node) const
{
  // propagate the slew and delay
  auto* left = node->get_left();
  auto* right = node->get_right();
  if (!left && !right) {
    return;
  }
  double slew_out = node->get_slew_in();
  double insert_delay = 0.0;
  if (node->isBuffer()) {
    auto slew_in = node->get_slew_in();
    auto cap_out = node->get_cap_out();

    slew_out = _lib->calcInsertSlew(slew_in, cap_out);
    insert_delay = _lib->calcDelay(slew_in, cap_out);
    node->set_insertion_delay(insert_delay);
  }

  if (left) {
    auto wire_slew = calcIdealSlew(node, left);
    auto slew_in = calcGeometricMean(wire_slew, slew_out);
    left->set_slew_in(slew_in);

    auto delay = node->get_delay() + insert_delay + calcElmoreDelay(node, left);
    left->set_delay(delay);

    slewPropagation(left);
  }
  if (right) {
    auto wire_slew = calcIdealSlew(node, right);
    auto slew_in = calcGeometricMean(wire_slew, slew_out);
    right->set_slew_in(slew_in);

    auto delay = node->get_delay() + insert_delay + calcElmoreDelay(node, right);
    right->set_delay(delay);

    slewPropagation(right);
  }
}

void HCTS::heuristicBuffering() const
{
  _root->set_feasible_cap(_max_cap);
  _root->set_slew_in(_max_buf_tran);
  setBuffer(_root);
  recursiveBuffering(_root);
}

void HCTS::recursiveBuffering(HNode* node) const
{
  if (node->get_sub_total_cap() <= node->get_feasible_cap()) {
    return;
  }
  allocateRemainCap(node);
  auto* left = node->get_left();
  auto* right = node->get_right();
  if (left) {
    auto* left_feasible_node = capFeasibleNode(node, node->get_left());
    left_feasible_node->set_level(node->get_level() + 1);
    recursiveBuffering(left_feasible_node);
  }
  if (right) {
    auto* right_feasible_node = capFeasibleNode(node, node->get_right());
    right_feasible_node->set_level(node->get_level() + 1);
    recursiveBuffering(right_feasible_node);
  }
}

void HCTS::allocateRemainCap(HNode* node) const
{
  auto* left = node->get_left();
  auto* right = node->get_right();
  auto remain_cap = node->get_feasible_cap();
  if (left && right) {
    // parameter
    auto remain_cap_ratio = left->get_sub_total_cap() > right->get_sub_total_cap() ? 1.5 : 0.5;
    right->set_feasible_cap(remain_cap / (1 + remain_cap_ratio));
    left->set_feasible_cap(remain_cap / (1 + remain_cap_ratio) * remain_cap_ratio);
    return;
  }
  if (left) {
    left->set_feasible_cap(remain_cap);
    return;
  }
  if (right) {
    right->set_feasible_cap(remain_cap);
    return;
  }
}

HNode* HCTS::capFeasibleNode(HNode* parent, HNode* child) const
{
  if (child->isSink()) {
    return child;
  }
  auto length = calcLength(parent, child);
  auto wire_cap = length * _unit_cap;
  auto remain_cap = child->get_feasible_cap();
  auto pin_cap = _lib->get_init_cap();
  // case 1: sub total cap can be cover with a buffer in child
  if (remain_cap >= child->get_sub_total_cap() + pin_cap) {
    setBuffer(child);
    child->set_feasible_cap(_max_cap);
    return child;
  }
  // case 2: remain cap can't cover the wire cap
  if (remain_cap <= wire_cap + pin_cap) {
    auto dist_to_parent = static_cast<int>((remain_cap - pin_cap) / _unit_cap * _db_unit);
    auto loc = internalPoint(parent->getLocation(), child->getLocation(), dist_to_parent);
    auto* insert_node = makeBuffer(parent, child, loc);
    insert_node->set_feasible_cap(_max_cap);
    insert_node->set_sub_total_cap(calcLength(insert_node, child) * _unit_cap + child->get_sub_total_cap());
    return insert_node;
  }
  // case 3: remain cap can cover the wire cap but not enough 2 pin cap
  if (remain_cap <= wire_cap + 2 * pin_cap) {
    setBuffer(child);
    child->set_feasible_cap(_max_cap);
    return child;
  }
  // case 4: sub total cap can be cover with a buffer between parent and child
  if (remain_cap < child->get_sub_total_cap() + pin_cap && child->get_sub_total_cap() + pin_cap < remain_cap + wire_cap) {
    auto dist_to_child = static_cast<int>((child->get_sub_total_cap() - remain_cap) / _unit_cap * _db_unit);
    auto loc = internalPoint(child->getLocation(), parent->getLocation(), dist_to_child);
    auto* insert_node = makeBuffer(parent, child, loc);
    insert_node->set_feasible_cap(_max_cap);
    insert_node->set_sub_total_cap(calcLength(insert_node, child) * _unit_cap + child->get_sub_total_cap());
    return insert_node;
  }
  // case 5: sub total cap can't be cover but remain cap enough
  if (remain_cap - wire_cap > 2 * pin_cap) {
    child->set_feasible_cap(remain_cap - wire_cap);
    return child;
  }
  // case 6: child remian cap not enough
  auto dist_to_parent = static_cast<int>((remain_cap - pin_cap) / _unit_cap * _db_unit);
  auto loc = internalPoint(parent->getLocation(), child->getLocation(), dist_to_parent);
  auto* insert_node = makeBuffer(parent, child, loc);
  insert_node->set_feasible_cap(_max_cap);
  insert_node->set_sub_total_cap(calcLength(insert_node, child) * _unit_cap + child->get_sub_total_cap());
  return insert_node;
}

void HCTS::makeTopos()
{
  std::queue<HNode*> node_queue;
  node_queue.push(_root);
  while (!node_queue.empty()) {
    auto* current = node_queue.front();
    node_queue.pop();
    if (current->isBuffer()) {
      auto topo = makeTopo(current);
      _clock_topos.emplace_back(topo);
      _node_map[current->getName()] = current;
      CTSAPIInst.addHCtsNode(current);
    }
    auto* left = current->get_left();
    auto* right = current->get_right();
    if (left) {
      node_queue.push(left);
    }
    if (right) {
      node_queue.push(right);
    }
  }
  std::reverse(_clock_topos.begin(), _clock_topos.end());
}

ClockTopo HCTS::makeTopo(HNode* root) const
{
  auto topo_name = CTSAPIInst.toString(_net_name, "_", root->get_id());
  ClockTopo topo(topo_name);

  topo.add_driver(root->get_inst());
  std::queue<HNode*> q;
  if (root->get_left()) {
    q.push(root->get_left());
  }
  if (root->get_right()) {
    q.push(root->get_right());
  }
  size_t steiner_id = 0;
  while (!q.empty()) {
    auto* current = q.front();
    q.pop();
    if (!current) {
      continue;
    }
    if (!current->isSteiner()) {
      topo.add_load(current->get_inst());
    }
    // add wire from parent
    auto current_loc = current->getLocation();
    auto* parent = current->get_parent();
    auto parent_loc = parent->getLocation();

    if (pgl::rectilinear(parent_loc, current_loc)) {
      topo.add_signal_wire(CtsSignalWire(Endpoint(parent->getName(), parent_loc), Endpoint(current->getName(), current_loc)));
    } else {
      auto trunk_loc = Point(parent_loc.x(), current_loc.y());
      auto trunk_name = "steiner_" + std::to_string(steiner_id++);
      topo.add_signal_wire(CtsSignalWire(Endpoint(parent->getName(), parent_loc), Endpoint(trunk_name, trunk_loc)));
      topo.add_signal_wire(CtsSignalWire(Endpoint(trunk_name, trunk_loc), Endpoint(current->getName(), current_loc)));
    }

    if (current->isSteiner()) {
      q.push(current->get_left());
      q.push(current->get_right());
    }
  }
  return topo;
}

// instantiation
HNode* HCTS::genBufferNode() const
{
  auto* buf_node = new HNode();
  auto id = buf_node->get_id();
  auto* buf_inst
      = new CtsInstance(_net_name + "_buf_" + std::to_string(id), _lib->get_cell_master(), CtsInstanceType::kBuffer, Point(-1, -1));
  buf_node->set_inst(buf_inst);
  buf_node->set_type(HNodeType::kBuffer);
  buf_node->set_cap_load(_lib->get_init_cap());
  return buf_node;
}

void HCTS::setBuffer(HNode* node) const
{
  node->set_type(HNodeType::kBuffer);
  auto* inst = node->get_inst();
  auto id = node->get_id();
  inst->set_name(_net_name + "_buf_" + std::to_string(id));
  inst->set_cell_master(_lib->get_cell_master());
  node->set_cap_load(_lib->get_init_cap());
  CTSAPIInst.placeInstance(inst);
}

HNode* HCTS::makeBuffer(HNode* parent, HNode* child, const Point& loc) const
{
  auto* buf_node = genBufferNode();
  buf_node->setLocation(loc);
  connect(parent, buf_node, child);
  CTSAPIInst.placeInstance(buf_node->get_inst());
  return buf_node;
}

void HCTS::connect(HNode* top, HNode* mid, HNode* bottom) const
{
  mid->set_parent(top);
  mid->set_left(bottom);
  bottom->set_parent(mid);
  if (top->get_left() == bottom) {
    top->set_left(mid);
  } else {
    top->set_right(mid);
  }
}

// basic update
void HCTS::updateSubTotalCap(HNode* node) const
{
  auto* left = node->get_left();
  auto* right = node->get_right();
  double sub_left_total_cap = 0.0;
  double sub_right_total_cap = 0.0;
  if (left) {
    sub_left_total_cap = left->get_sub_total_cap() + _unit_cap * calcLength(node, left);
    if (left->isSink()) {
      sub_left_total_cap += left->get_cap_load();
    }
  }
  if (right) {
    sub_right_total_cap = right->get_sub_total_cap() + _unit_cap * calcLength(node, right);
    if (right->isSink()) {
      sub_right_total_cap += right->get_cap_load();
    }
  }
  // parameter
  node->set_sub_total_cap(sub_left_total_cap + sub_right_total_cap);
}

void HCTS::updateCapCenterLoc(HNode* node) const
{
  auto* left = node->get_left();
  auto* right = node->get_right();
  auto left_sub_total_cap = left->get_sub_total_cap();
  auto right_sub_total_cap = right->get_sub_total_cap();

  auto wire_cap = _unit_cap * calcLength(left, right);
  auto left_loc = left->getLocation();
  auto right_loc = right->getLocation();
  if (fabs(left_sub_total_cap - right_sub_total_cap) > wire_cap) {
    auto loc = (left_loc + right_loc) / 2;
    node->setLocation(loc);
    return;
  }
  auto mean_cap = (wire_cap + left_sub_total_cap + right_sub_total_cap) / 2;
  auto dist_to_left = static_cast<int>((mean_cap - left_sub_total_cap) / _unit_cap * _db_unit);
  auto loc = internalPoint(left_loc, right_loc, dist_to_left);
  node->setLocation(loc);
}

// basic calc

// report
void HCTS::reportTiming() const
{
  auto timing_rpt = CtsReportTable::createReportTable("HCTS Timing Log", CtsReportType::kHCTS_LOG);
  for (auto [_, node] : _node_map) {
    auto loc_str = CTSAPIInst.toString("(", node->getLocation().x(), ",", node->getLocation().y(), ")");
    (*timing_rpt) << node->get_id() << node->getName() << node->get_net_length() << loc_str << node->get_fanout() << node->get_delay()
                  << node->get_slew_in() << node->get_cap_out() << node->get_insertion_delay() << node->get_sub_total_cap()
                  << node->get_level() << TABLE_ENDLINE;
  }
  auto dir = CTSAPIInst.get_config()->get_sta_workspace() + "/hcts_log";
  auto file_name = _net_name + "_hcts_log.rpt";
  auto save_path = dir + "/" + file_name;
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
  }
  std::ofstream outfile(save_path);
  outfile << "Generate the report at " << Time::getNowWallTime() << std::endl;
  outfile << timing_rpt->c_str();
  outfile.close();
}
}  // namespace icts