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
#include "report_route.h"

namespace iplf {

// check if a endpoint of a segment is overlayed by the layer_shape
static void checkSegPointsOverlay(std::set<std::pair<int32_t, int32_t>>& points, IdbRegularWireSegment* seg, IdbLayerShape* layer_shape)
{
  if (seg->is_rect()) {
    return;
  }
  IdbLayer* shapeLayer = layer_shape->get_layer();
  for (auto* rect : layer_shape->get_rect_list()) {
    // check via
    std::vector<std::pair<int32_t, int32_t>> pts;
    if (seg->is_via()) {
      for (auto* via : seg->get_via_list()) {
        if (via->get_bottom_layer_shape().get_layer() == shapeLayer || via->get_top_layer_shape().get_layer() == shapeLayer) {
          if (rect->containPoint(via->get_coordinate())) {
            auto* coord = via->get_coordinate();
            points.insert(std::pair{coord->get_x(), coord->get_y()});
            goto outercontinue;
          }
        }
      }
    }
    // check wire segment
    if (shapeLayer != seg->get_layer()) {
      continue;
    }
    for (auto* pt : seg->get_point_list()) {
      if (rect->containPoint(pt)) {
        pts.emplace_back(std::pair{pt->get_x(), pt->get_y()});
      }
    }
    if (pts.size() == seg->get_point_list().size()) {
      continue;
    }
    for (auto pt : pts) {
      points.insert(pt);
    }
  outercontinue:
    continue;
  }
};

static int32_t pinLength(IdbPin* pin, IdbNet* net)
{
  /**
   * the wire may be like:
   * [PIN_A] ------[PIN_B]-------[PIN_C]
   * consider the wire length go through PIN_B :
   *    use manhattan distance between joinpoints.
   */
  int32_t length = 0;
  std::set<std::pair<int32_t, int32_t>> points;
  for (auto* wire : net->get_wire_list()->get_wire_list()) {
    for (auto* seg : wire->get_segment_list()) {
      for (auto* layer_shape : pin->get_port_box_list()) {
        checkSegPointsOverlay(points, seg, layer_shape);
      }
    }
  }
  if (points.size() < 2) {
    return 0;
  }
  int32_t x0 = std::numeric_limits<int32_t>::max(), y0 = x0, x1 = 0, y1 = 0;
  for (auto [x, y] : points) {
    x0 = std::min(x, x0);
    y0 = std::min(y, y0);
    x1 = std::max(x, x1);
    y1 = std::max(y, y1);
  }
  length = x1 - x0 + y1 - y0;
  return length;
}

NetStatistics NetStatistics::extractNetInfo(IdbNet* net)
{
  NetStatistics net_info;
  net_info.Name = net->get_net_name();
  net_info.FanIn = net->get_driving_pin() ? 1 : 0;
  net_info.FanOut = net->get_load_pins().size();
  net_info.PinNum = net->get_pin_number();
  int32_t x0 = std::numeric_limits<int32_t>::max(), y0 = x0, x1 = 0, y1 = 0;
  auto hpwl = [&x0, &y0, &x1, &y1](int32_t x, int32_t y) mutable {
    x0 = std::min(x0, x);
    y0 = std::min(y0, y);
    x1 = std::max(x1, x);
    y1 = std::max(y1, y);
  };

  for (auto* wires : net->get_wire_list()->get_wire_list()) {
    for (auto* seg : wires->get_segment_list()) {
      int layer_index = seg->get_layer()->get_id();
      net_info.RouteLayerLength[layer_index] += seg->length();
      for (auto* via : seg->get_via_list()) {
        int cutlayer_index = via->get_cut_layer_shape().get_layer()->get_id();
        net_info.CutLayerVias[cutlayer_index]++;
        net_info.Vias[via->get_name()]++;
      }
    }
  }

  auto& pinlist = net->get_instance_pin_list()->get_pin_list();

  for (auto* pin : pinlist) {
    if (pinlist.size() > 2) {
      auto* layer = pin->get_bottom_routing_layer_shape()->get_layer();
      net_info.RouteLayerLength[layer->get_id()] += pinLength(pin, net);
    }
    auto* coord = pin->get_average_coordinate();
    hpwl(coord->get_x(), coord->get_y());
  }
  net_info.HPWL = y1 - y0 + x1 - x0;
  net_info.TotalLength = std::accumulate(net_info.RouteLayerLength.begin(), net_info.RouteLayerLength.end(), 0L);
  net_info.TotalVias = std::accumulate(net_info.CutLayerVias.begin(), net_info.CutLayerVias.end(), 0L);
  return net_info;
}

std::string formatDouble(double d, int precision = 3)
{
  std::stringstream ss;
  ss << std::fixed << std::setprecision(precision) << d;
  return ss.str();
}

/**
 * @brief
 *    cut a string into multi lines.
 * @param str
 * @param n
 * @param width
 * @return vector<std::string_view>
 */
static vector<std::string_view> cutString(std::string_view str, int n, int width)
{
  if (n <= 0) {
    return {};
  }
  int64_t str_size = str.size();
  if (str_size > n * width) {
    width = (str_size + 1) / n;
  }
  vector<std::string_view> vec(n);
  int i = 0;
  int pos = 0;
  for (; pos + width < str_size; pos += width) {
    vec[i++] = str.substr(pos, width);
  }
  vec[i] = str.substr(pos, str_size - pos);
  return vec;
}

void ReportRoute::createNetReport(IdbNet* net)
{
  double dbu = dmInst->get_idb_layout()->get_units()->get_micron_dbu();
  auto net_info = NetStatistics::extractNetInfo(net);
  double ratio = static_cast<double>(net_info.TotalLength) / net_info.HPWL;
  int size = std::max(net_info.RouteLayerLength.size(), net_info.CutLayerVias.size());
  while (size > 0 && net_info.RouteLayerLength[size - 1] == 0 && net_info.CutLayerVias[size - 1] == 0) {
    --size;
  }
  auto routing_layers = dmInst->get_idb_layout()->get_layers()->get_routing_layers();
  auto cut_layers = dmInst->get_idb_layout()->get_layers()->get_cut_layers();
  const int NAME_LENGTH = 42;
  auto name_slice = cutString(net_info.Name, size + 1, NAME_LENGTH);

  std::vector<std::string> header
      = {"\nNet Name", "  HPWL\nLength", " Routing\nLayer", "\nLength", " Cut\nLayer", " Via\nCount", "Wire Length\n To HPWL\nLength Ratio",
         "Fan_in",     "Fan_out"};
  auto net_tbl = std::make_shared<ieda::ReportTable>("Net Information", header, -1);
  *net_tbl << name_slice[0] << formatDouble(net_info.HPWL / dbu) << "Total" << formatDouble(net_info.TotalLength / dbu) << "Total"
           << net_info.TotalVias << formatDouble(ratio) << net_info.FanIn << net_info.FanOut << TABLE_ENDLINE;

  for (int i = 0; i < size; ++i) {
    *net_tbl << name_slice[i + 1] << TABLE_SKIP << routing_layers[i]->get_name() << formatDouble(net_info.RouteLayerLength[i] / dbu)
             << cut_layers[i]->get_name() << net_info.CutLayerVias[i] << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_ENDLINE;
  }
  net_tbl->row(0).set_cell_text_align(fort::text_align::center);
  net_tbl->column(3).set_cell_text_align(fort::text_align::right);
  net_tbl->column(5).set_cell_text_align(fort::text_align::right);
  net_tbl->column(6).set_cell_text_align(fort::text_align::right);
  this->add_table(net_tbl);

  auto name_slice_v = cutString(net_info.Name, net_info.Vias.size(), NAME_LENGTH * 3 / 2);
  std::vector<std::string> vheader = {"Net Name", "Via-Instance Name", "Count"};
  auto via_tbl = std::make_shared<ieda::ReportTable>("", vheader, -1);
  int index = 0;
  for (auto [via, cnt] : net_info.Vias) {
    *via_tbl << name_slice_v[index++] << via << cnt << TABLE_ENDLINE;
  }
  this->add_table(via_tbl);
}

std::shared_ptr<ieda::ReportTable> ReportRoute::getDesignStatsTable(int64_t pins_number)
{
  auto* design = dmInst->get_idb_design();
  std::stringstream title;
  title << "Statistics for design " << design->get_design_name() << " :";
  std::vector<std::string> header = {"Design Metric", "Count"};
  auto tbl = std::make_shared<ieda::ReportTable>(title.str().c_str(), header, -1);
  *tbl << "Instances" << design->get_instance_list()->get_num() << TABLE_ENDLINE;
  *tbl << "Nets" << design->get_net_list()->get_num() << TABLE_ENDLINE;
  *tbl << "Pins" << pins_number << TABLE_ENDLINE;
  *tbl << "IO Terminals" << design->get_io_pin_list()->get_pin_num() << TABLE_ENDLINE;
  return tbl;
}

void ReportRoute::createSummaryReport()
{
  auto& netlist = dmInst->get_idb_design()->get_net_list()->get_net_list();

  vector<int64_t> net_pins(11, 0);
  NetStatistics summary;
  vector<int64_t> lengths;

  lengths.reserve(netlist.size());
  for (IdbNet* net : netlist) {
    NetStatistics net_stats = NetStatistics::extractNetInfo(net);
    net_pins[net_stats.PinNum > 10 ? 10 : net_stats.PinNum]++;
    lengths.push_back(net_stats.TotalLength);
    summary += net_stats;
  }
  this->add_table(getDesignStatsTable(summary.PinNum));
  this->add_table(getPinStatsTable(net_pins, netlist.size()));
  this->add_table(getWireLengthStatsTable(summary.RouteLayerLength));
  this->add_table(getViaCutStatsTable(summary.CutLayerVias));
  this->add_table(getLengthRangeTable(lengths, 10000));
}

std::shared_ptr<ieda::ReportTable> ReportRoute::getPinStatsTable(vector<int64_t>& net_pins, int64_t nets)
{
  std::vector<std::string> header{"Number Of\nConnected Pins", "Number\nof Pins", "Percentage\nof Nets"};
  auto tbl = std::make_shared<ieda::ReportTable>("Net Pin Statistics :", header, -1);
  for (size_t i = 1; i < net_pins.size() - 1; ++i) {
    *tbl << i << net_pins[i] << formatDouble((100.0 * net_pins[i]) / nets) << TABLE_ENDLINE;
  }
  std::string pins = std::string(">=") + std::to_string(net_pins.size() - 1);
  *tbl << pins << net_pins.back() << formatDouble((100.0 * net_pins.back()) / nets) << TABLE_ENDLINE;
  tbl->set_cell_text_align(fort::text_align::right);
  return tbl;
}

std::shared_ptr<ieda::ReportTable> ReportRoute::getWireLengthStatsTable(const std::vector<int64_t>& routing_layer_length)
{
  auto routing_layers = dmInst->get_idb_layout()->get_layers()->get_routing_layers();
  int32_t dbu = dmInst->get_idb_layout()->get_units()->get_micron_dbu();
  std::vector<std::string> header{"Layer Name", "Wire Length(dbu)", "Wire Length(um)"};
  auto tbl = std::make_shared<ieda::ReportTable>("Wire Length Statistics :", header, -1);
  int64_t total = 0;
  for (size_t i = 0; i < routing_layers.size(); ++i) {
    int64_t layer_len = routing_layer_length[i];
    total += layer_len;
    *tbl << routing_layers[i]->get_name() << layer_len << formatDouble(static_cast<double>(layer_len) / dbu) << TABLE_ENDLINE;
  }
  *tbl << TABLE_HEAD << "Total" << total << formatDouble(static_cast<double>(total) / dbu) << TABLE_ENDLINE;
  tbl->column(0).set_cell_text_align(fort::text_align::center);
  tbl->column(1).set_cell_text_align(fort::text_align::right);
  tbl->column(2).set_cell_text_align(fort::text_align::right);
  return tbl;
}

std::shared_ptr<ieda::ReportTable> ReportRoute::getViaCutStatsTable(const std::vector<int64_t>& via_cut_nums)
{
  auto cutlayers = dmInst->get_idb_layout()->get_layers()->get_cut_layers();
  std::vector<std::string> header{"Via-Cut Name", "Count"};
  auto tbl = std::make_shared<ieda::ReportTable>("Via Count Statistics :", header, -1);
  int64_t total = 0;
  for (size_t i = 0; i < cutlayers.size(); ++i) {
    int64_t cnt = via_cut_nums[i];
    total += cnt;
    *tbl << cutlayers[i]->get_name() << cnt << TABLE_ENDLINE;
  }
  *tbl << TABLE_HEAD << "Total" << total << TABLE_ENDLINE;
  tbl->column(0).set_cell_text_align(fort::text_align::center);
  tbl->column(1).set_cell_text_align(fort::text_align::right);
  return tbl;
}

std::shared_ptr<ieda::ReportTable> ReportRoute::getLengthRangeTable(std::vector<int64_t>& lengths, int64_t d)
{
  std::sort(lengths.begin(), lengths.end());

  vector<int64_t> distribution{0};
  int64_t cur = d;
  int64_t tail = 0;
  double dbu = dmInst->get_idb_layout()->get_units()->get_micron_dbu();
  std::vector<std::string> header{"Length Range", "Number of Nets\nwith Wire Length"};
  auto tbl = std::make_shared<ieda::ReportTable>("Net Length Distribution :", header, -1);
  for (size_t i = 0; i < lengths.size(); ++i) {
    if (lengths[i] < cur) {
      distribution.back()++;
    } else if (lengths[i] < cur + d) {
      cur += d;
      distribution.push_back(1);
    } else {
      tail = lengths[i];
      distribution.push_back(lengths.size() - i);
      break;
    }
  }
  std::stringstream ss;

  for (size_t i = 0; i < distribution.size() - 1; ++i) {
    ss << formatDouble(i * d / dbu, 2) << " um ~ " << formatDouble((i + 1) * d / dbu, 2) << " um";
    *tbl << ss.str() << distribution[i] << TABLE_ENDLINE;
    std::stringstream().swap(ss);
  }
  ss << formatDouble(tail / dbu, 2) << " um ~ " << formatDouble(lengths.back() / dbu, 2) << " um";
  *tbl << ss.str() << distribution.back() << TABLE_ENDLINE;

  return tbl;
}

NetStatistics& NetStatistics::operator+=(const NetStatistics& ref)
{
  HPWL += ref.HPWL;
  TotalLength += ref.TotalLength;
  TotalVias += ref.TotalVias;
  for (size_t i = 0; i < RouteLayerLength.size(); ++i) {
    RouteLayerLength[i] += ref.RouteLayerLength[i];
  }
  for (size_t i = 0; i < CutLayerVias.size(); ++i) {
    CutLayerVias[i] += ref.CutLayerVias[i];
  }
  FanIn += ref.FanIn, FanOut += ref.FanOut, PinNum += ref.PinNum;
  for (auto& [via, cnt] : ref.Vias) {
    Vias[via] += cnt;
  }
  return *this;
}

}  // namespace iplf