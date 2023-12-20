/**
 * @file SAEvaluateWl.hh
 * @author Yuezuo Liu (yuezuoliu@163.com)
 * @brief
 * @version 0.1
 * @date 2023-12-1
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IMP_SA_EVALUATE_WL_H
#define IMP_SA_EVALUATE_WL_H

#include <vector>

#include "SAEvaluate.hh"
#include "SAPlacement.hh"
#include "../Evaluator.hh"
namespace imp {

template <typename CoordType, typename RepresentType>
struct SAPlacement;

template <typename CoordType, typename RepresentType>
struct SAEvaluateWl : public SAEvaluate<CoordType, RepresentType>
{
 public:
  SAEvaluateWl(SAPlacement<CoordType, RepresentType>& placement, bool init_norm = true)
  {
    _cost_ratio["wirelength"] = 1;
    _cost_ratio["area"] = 0.1;
    const auto& pin_x_off = *(placement.get_pin_x_off());
    const auto& pin_y_off = *(placement.get_pin_y_off());
    const auto& dx = *(placement.get_dx());
    const auto& dy = *(placement.get_dy());
    const auto& pin2vertex = *(placement.get_pin2vertex());

    _pin_x_buffer.resize(pin_x_off.size());
    _pin_y_buffer.resize(pin_y_off.size());
    _lx_buffer = *(placement.get_initial_lx());
    _ly_buffer = *(placement.get_initial_ly());

    for (size_t i = 0; i < pin2vertex.size(); i++) {
      size_t j = pin2vertex[i];
      _pin_x_buffer[i] = _lx_buffer[j] + dx[j] / 2.0 + pin_x_off[i];
      _pin_y_buffer[i] = _ly_buffer[j] + dy[j] / 2.0 + pin_y_off[i];
    }
    if (init_norm) {
      SAEvaluate<CoordType, RepresentType>::initCostNorm(placement);
    }
  }
  ~SAEvaluateWl() = default;
  SAEvaluateWl(const SAEvaluateWl& other) = default;
  SAEvaluateWl& operator=(const SAEvaluateWl& other)
  {
    _pin_x_buffer = other._pin_x_buffer;
    _pin_y_buffer = other._pin_y_buffer;
    _lx_buffer = other._lx_buffer;
    _ly_buffer = other._ly_buffer;
    return *this;
  }
  double operator()(SAPlacement<CoordType, RepresentType>& placement)
  {
    std::map<std::string, double> normed_cost_dict = SAEvaluate<CoordType, RepresentType>::normalizeCostDict(calCostDict(placement));
    double total_cost = 0;
    for (const auto [cost_name, cost_value] : normed_cost_dict) {
      total_cost += _cost_ratio[cost_name] * cost_value;
    }
    return total_cost;
  }
  void set_cost_wirelength_ratio(double cost_wirelength_ratio) { _cost_ratio["wirelength"] = cost_wirelength_ratio; }
  void set_cost_area_ratio(double cost_area_ratio) { _cost_ratio["area"] = cost_area_ratio; }

  const std::vector<CoordType>& get_lx_buffer() const { return _lx_buffer; }
  const std::vector<CoordType>& get_ly_buffer() const { return _ly_buffer; }
  double get_cost_wirelength_ratio() const { return _cost_ratio.at("wirelength"); }
  double get_cost_area_ratio() const { return _cost_ratio.at("area"); }
  double get_cost_delta_avg() const
  {
    double delta_avg = 0.;
    auto& cost_avg = SAEvaluate<CoordType, RepresentType>::get_cost_delta_dict();
    for (auto& [cost_name, delta_value] : cost_avg) {
      delta_avg += _cost_ratio.at(cost_name) * delta_value;
    }
    return delta_avg;
  }

 private:
  std::vector<double> _pin_x_buffer;
  std::vector<double> _pin_y_buffer;
  std::vector<CoordType> _lx_buffer;
  std::vector<CoordType> _ly_buffer;
  std::pair<CoordType, CoordType> _bound;
  std::map<std::string, double> _cost_ratio;

  std::map<std::string, double> calCostDict(SAPlacement<CoordType, RepresentType>& placement) override
  {
    _bound = placement.packing(_lx_buffer, _ly_buffer);
    std::map<std::string, double> cost_dict;
    if (_cost_ratio["wirelength"] != 0) {
      cost_dict["wirelength"] = WireLength(placement);
    }
    if (_cost_ratio["area"] != 0) {
      cost_dict["area"] = Area(placement);
    }
    return cost_dict;
  }

  double Area(SAPlacement<CoordType, RepresentType>& placement)
  {
    return _bound.first / placement.get_region_dx() * _bound.second / placement.get_region_dy();
  }
  double WireLength(SAPlacement<CoordType, RepresentType>& placement)
  {
    const auto& pin_x_off = *(placement.get_pin_x_off());
    const auto& pin_y_off = *(placement.get_pin_y_off());
    const auto& dx = *(placement.get_dx());
    const auto& dy = *(placement.get_dy());
    const auto& pin2vertex = *(placement.get_pin2vertex());
    const auto& net_span = *(placement.get_net_span());
    const size_t& num_moveable = placement.get_num_moveable();

    for (size_t i = 0; i < pin2vertex.size(); i++) {
      if (pin2vertex[i] >= num_moveable)
        continue;
      size_t j = pin2vertex[i];
      _pin_x_buffer[i] = _lx_buffer[j] + dx[j] / 2.0 + pin_x_off[i];
      _pin_y_buffer[i] = _ly_buffer[j] + dy[j] / 2.0 + pin_y_off[i];
    }
    double wl = hpwl(_pin_x_buffer, _pin_y_buffer, net_span, 16);
    return wl;
  }
};

}  // namespace imp

#endif