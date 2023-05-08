#include "WLNet.hpp"

#include "flute.h"

namespace eval {

void WLNet::add_pin(const int64_t& x, const int64_t& y)
{
  WLPin* pin = new WLPin();
  pin->set_x(x);
  pin->set_y(y);
  _pin_list.push_back(pin);
}

void WLNet::add_driver_pin(const int64_t& x, const int64_t& y, const std::string& name)
{
  WLPin* pin = new WLPin();
  pin->set_x(x);
  pin->set_y(y);
  pin->set_name(name);
  _driver_pin = pin;
  _pin_list.push_back(pin);
}

void WLNet::add_sink_pin(const int64_t& x, const int64_t& y, const std::string& name)
{
  WLPin* pin = new WLPin();
  pin->set_x(x);
  pin->set_y(y);
  pin->set_name(name);
  _sink_pin_list.push_back(pin);
  _pin_list.push_back(pin);
}

int64_t WLNet::wireLoadModel()
{
  int64_t wlm = 0;
  const int MAX_FANOUT = 8;
  const int SLOPE = 5;
  const int UNIT = 1000;

  int64_t fanout = _sink_pin_list.size();

  switch (fanout) {
    case 1:
      wlm = 1.3207;
      break;
    case 2:
      wlm = 2.9813;
      break;
    case 3:
      wlm = 5.1135;
      break;
    case 4:
      wlm = 7.6639;
      break;
    case 5:
      wlm = 10.0334;
      break;
    case 6:
      wlm = 12.2296;
      break;
    case 8:
      wlm = 19.3185;
      break;
    default:
      wlm = (fanout - MAX_FANOUT) * SLOPE + 19.3185;
      break;
  }
  return wlm * UNIT;
}

int64_t WLNet::HPWL()
{
  int64_t HPWL = 0;
  int64_t pins_max_x = 0;
  int64_t pins_max_y = 0;
  int64_t pins_min_x = LONG_LONG_MAX;
  int64_t pins_min_y = LONG_LONG_MAX;
  for (auto& pin : _pin_list) {
    auto pin_coord = pin->get_coord();
    if (pin_coord.get_x() > pins_max_x) {
      pins_max_x = pin_coord.get_x();
    }
    if (pin_coord.get_y() > pins_max_y) {
      pins_max_y = pin_coord.get_y();
    }
    if (pin_coord.get_x() < pins_min_x) {
      pins_min_x = pin_coord.get_x();
    }
    if (pin_coord.get_y() < pins_min_y) {
      pins_min_y = pin_coord.get_y();
    }
  }
  HPWL = (pins_max_x - pins_min_x + pins_max_y - pins_min_y);
  return HPWL;
}

int64_t WLNet::LShapedWL(const std::string& sink_pin_name)
{
  int64_t LShapedWL = 0;

  std::map<std::string, WLPin*> name2pin_map;
  for (auto& sink_pin : _sink_pin_list) {
    std::string name = sink_pin->get_name();
    name2pin_map[name] = sink_pin;
  }

  auto it = name2pin_map.find(sink_pin_name);
  if (it != name2pin_map.end()) {
    LShapedWL = _driver_pin->get_coord().computeDist(name2pin_map[sink_pin_name]->get_coord());
  } else {
    LShapedWL = 0;
  }

  return LShapedWL;
}

int64_t WLNet::HTree()
{
  int64_t HTree = 0;
  int64_t x_direction_gravity = 0;
  for (auto& pin : _pin_list) {
    x_direction_gravity += pin->get_coord().get_x();
  }
  x_direction_gravity /= _pin_list.size();

  int64_t x_direction_length = 0;
  int64_t pins_max_y = 0;
  int64_t pins_min_y = LONG_LONG_MAX;
  for (auto& pin : _pin_list) {
    auto pin_coord = pin->get_coord();
    x_direction_length += abs(pin_coord.get_x() - x_direction_gravity);
    if (pin_coord.get_y() > pins_max_y) {
      pins_max_y = pin_coord.get_y();
    }
    if (pin_coord.get_y() < pins_min_y) {
      pins_min_y = pin_coord.get_y();
    }
  }
  HTree = (x_direction_length + pins_max_y - pins_min_y);
  return HTree;
}

int64_t WLNet::VTree()
{
  int64_t VTree = 0;
  int64_t y_direction_gravity = 0;
  for (auto& pin : _pin_list) {
    y_direction_gravity += pin->get_coord().get_y();
  }
  y_direction_gravity /= _pin_list.size();

  int64_t y_direction_length = 0;
  int64_t pins_max_x = 0;
  int64_t pins_min_x = LONG_LONG_MAX;
  for (auto& pin : _pin_list) {
    auto pin_coord = pin->get_coord();
    y_direction_length += abs(pin_coord.get_y() - y_direction_gravity);
    if (pin_coord.get_x() > pins_max_x) {
      pins_max_x = pin_coord.get_x();
    }
    if (pin_coord.get_x() < pins_min_x) {
      pins_min_x = pin_coord.get_x();
    }
  }
  VTree = (y_direction_length + pins_max_x - pins_min_x);
  return VTree;
}

int64_t WLNet::Star()
{
  int64_t Star = 0;
  Point<int64_t> fake_point;
  int64_t x_direction_gravity = 0;
  int64_t y_direction_gravity = 0;
  for (auto& pin : _pin_list) {
    x_direction_gravity += pin->get_coord().get_x();
    y_direction_gravity += pin->get_coord().get_y();
  }
  x_direction_gravity /= _pin_list.size();
  y_direction_gravity /= _pin_list.size();
  fake_point.set_x(x_direction_gravity);
  fake_point.set_y(y_direction_gravity);

  for (auto& pin : _pin_list) {
    Star += pin->get_coord().computeDist(fake_point);
  }
  return Star;
}

int64_t WLNet::Clique()
{
  int64_t Clique = 0;
  for (size_t i = 0; i < _pin_list.size() - 1; i++) {
    for (size_t j = i + 1; j < _pin_list.size(); j++) {
      Clique += (_pin_list[i]->get_coord().computeDist(_pin_list[j]->get_coord()));
    }
  }
  if (_pin_list.size() > 1) {
    Clique = Clique / (_pin_list.size() - 1);
  } else {
    Clique = 0;
  }
  return Clique;
}

int64_t WLNet::B2B()
{
  int64_t B2B = 0;
  int64_t B2B_x = 0;
  int64_t B2B_y = 0;

  int64_t pins_max_x = 0;
  int64_t pins_max_y = 0;
  int64_t pins_min_x = LONG_LONG_MAX;
  int64_t pins_min_y = LONG_LONG_MAX;
  for (auto& pin : _pin_list) {
    auto pin_coord = pin->get_coord();
    if (pin_coord.get_x() > pins_max_x) {
      pins_max_x = pin_coord.get_x();
    }
    if (pin_coord.get_y() > pins_max_y) {
      pins_max_y = pin_coord.get_y();
    }
    if (pin_coord.get_x() < pins_min_x) {
      pins_min_x = pin_coord.get_x();
    }
    if (pin_coord.get_y() < pins_min_y) {
      pins_min_y = pin_coord.get_y();
    }
  }

  for (int i = 0; i < (int) _pin_list.size(); i++) {
    if (_pin_list[i]->get_coord().get_x() == pins_min_x) {
      for (int j = 0; j < (int) _pin_list.size(); j++) {
        B2B_x += (_pin_list[j]->get_coord().get_x() - pins_min_x);
      }
    }
    if (_pin_list[i]->get_coord().get_x() == pins_max_x) {
      for (int j = 0; j < (int) _pin_list.size(); j++) {
        B2B_x += (pins_max_x - _pin_list[j]->get_coord().get_x());
      }
    }
    if (_pin_list[i]->get_coord().get_y() == pins_min_y) {
      for (int j = 0; j < (int) _pin_list.size(); j++) {
        B2B_y += (_pin_list[j]->get_coord().get_y() - pins_min_y);
      }
    }
    if (_pin_list[i]->get_coord().get_y() == pins_max_y) {
      for (int j = 0; j < (int) _pin_list.size(); j++) {
        B2B_y += (pins_max_y - _pin_list[j]->get_coord().get_y());
      }
    }
  }
  if (_pin_list.size() > 1) {
    B2B = (B2B_x + B2B_y - (pins_max_x - pins_min_x) - (pins_max_y - pins_min_y)) / (_pin_list.size() - 1);
  } else {
    B2B = 0;
  }
  return B2B;
}

int64_t WLNet::FluteWL()
{
  double total_StWL = 0.0;

  int num_pin = _pin_list.size();
  if (num_pin == 2) {
    double wirelength = fabs(_pin_list[0]->get_coord().get_x() - _pin_list[1]->get_coord().get_x())
                        + fabs(_pin_list[0]->get_coord().get_y() - _pin_list[1]->get_coord().get_y());
    total_StWL += wirelength;
  }

  else if (num_pin > 2) {
    int* x = new int[num_pin];
    int* y = new int[num_pin];

    if (get_driver_pin()) {
      x[0] = (int) get_driver_pin()->get_coord().get_x();
      y[0] = (int) get_driver_pin()->get_coord().get_y();
      int j = 1;
      for (auto& theSink : get_sink_pin_list()) {
        x[j] = (int) theSink->get_coord().get_x();
        y[j] = (int) theSink->get_coord().get_y();
        j++;
      }
    } else {
      int j = 0;
      for (auto& theSink : get_pin_list()) {
        x[j] = (int) theSink->get_coord().get_x();
        y[j] = (int) theSink->get_coord().get_y();
        j++;
      }
    }

    Flute::Tree flutetree = Flute::flute(num_pin, x, y, 8);
    delete[] x;
    delete[] y;

    int branchnum = 2 * flutetree.deg - 2;
    for (int j = 0; j < branchnum; ++j) {
      int n = flutetree.branch[j].n;
      if (j == n) {
        continue;
      }
      double wirelength = fabs((double) flutetree.branch[j].x - (double) flutetree.branch[n].x)
                          + fabs((double) flutetree.branch[j].y - (double) flutetree.branch[n].y);
      total_StWL += wirelength;
    }
    free(flutetree.branch);
  }
  return total_StWL;
}


int64_t WLNet::planeRouteWL()
{
  int64_t total_planeWL = 0;
  // for (auto& pin_pair : _pin_pair_list) {
  //   Point<int64_t> first_coord(pin_pair.first->get_coord().get_x(), pin_pair.first->get_coord().get_y());
  //   Point<int64_t> second_coord(pin_pair.second->get_coord().get_x(), pin_pair.second->get_coord().get_y());
  //   double wirelength = first_coord.computeDist(second_coord);
  //   total_planeWL += wirelength;
  // }
  return total_planeWL;
}

int64_t WLNet::spaceRouteWL()
{
  int64_t total_spaceWL = 0;
  // for (auto& pin_pair : _pin_pair_list) {
  //   Point<int64_t> first_coord(pin_pair.first->get_coord().get_x(), pin_pair.first->get_coord().get_y());
  //   Point<int64_t> second_coord(pin_pair.second->get_coord().get_x(), pin_pair.second->get_coord().get_y());
  //   if (first_coord == second_coord) {
  //     total_spaceWL += pin_pair.first->get_layer_thickness() * 2;
  //   } else {
  //     total_spaceWL += first_coord.computeDist(second_coord);
  //   }
  // }
  return total_spaceWL;
}

int64_t WLNet::detailRouteWL()
{
  // int64_t total_drWL = 0;
  // for (auto& pin_pair : _pin_pair_list) {
  //   Point<int64_t> first_coord(pin_pair.first->get_coord().get_x(), pin_pair.first->get_coord().get_y());
  //   Point<int64_t> second_coord(pin_pair.second->get_coord().get_x(), pin_pair.second->get_coord().get_y());
  //   double wirelength = first_coord.computeDist(second_coord);
  //   total_drWL += wirelength;
  // }
  // return total_drWL;
  return _real_wirelength;
}
}  // namespace eval
