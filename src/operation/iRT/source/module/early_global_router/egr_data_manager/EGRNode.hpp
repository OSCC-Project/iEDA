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

#include "EGRResourceType.hpp"

namespace irt {

class EGRNode : public PlanarRect
{
 public:
  EGRNode(/* args */) = default;
  ~EGRNode() = default;
  // getter
  double get_north_supply() const { return _north_supply; }
  double get_south_supply() const { return _south_supply; }
  double get_west_supply() const { return _west_supply; }
  double get_east_supply() const { return _east_supply; }
  double get_track_supply() const { return _track_supply; }
  double get_north_demand() const { return _north_demand; }
  double get_south_demand() const { return _south_demand; }
  double get_west_demand() const { return _west_demand; }
  double get_east_demand() const { return _east_demand; }
  double get_track_demand() const { return _track_demand; }
  // setter
  void set_north_supply(const double north_supply) { _north_supply = north_supply; }
  void set_south_supply(const double south_supply) { _south_supply = south_supply; }
  void set_west_supply(const double west_supply) { _west_supply = west_supply; }
  void set_east_supply(const double east_supply) { _east_supply = east_supply; }
  void set_track_supply(const double track_supply) { _track_supply = track_supply; }
  void set_north_demand(const double north_demand) { _north_demand = north_demand; }
  void set_south_demand(const double south_demand) { _south_demand = south_demand; }
  void set_west_demand(const double west_demand) { _west_demand = west_demand; }
  void set_east_demand(const double east_demand) { _east_demand = east_demand; }
  void set_track_demand(const double track_demand) { _track_demand = track_demand; }
  // func
  void addSupply(EGRResourceType resource_type, double supply);
  void addDemand(EGRResourceType resource_type, double demand);
  double getCost(EGRResourceType resource_type);
  double getRemain(EGRResourceType resource_type);
  double getOverflow(EGRResourceType resource_type);
  double getDemand(EGRResourceType resource_type);
  double getSupply(EGRResourceType resource_type);
  void setDemand(EGRResourceType resource_type, double demand);
  void setSupply(EGRResourceType resource_type, double supply);

 private:
  double _north_supply = 0;
  double _south_supply = 0;
  double _west_supply = 0;
  double _east_supply = 0;
  double _track_supply = 0;

  double _north_demand = 0;
  double _south_demand = 0;
  double _west_demand = 0;
  double _east_demand = 0;
  double _track_demand = 0;
};

inline void EGRNode::addSupply(EGRResourceType resource_type, double supply)
{
  if (std::isnan(supply)) {
    LOG_INST.error(Loc::current(), "Add nan error!");
  }
  switch (resource_type) {
    case EGRResourceType::kEast:
      _east_supply += supply;
      break;
    case EGRResourceType::kWest:
      _west_supply += supply;
      break;
    case EGRResourceType::kSouth:
      _south_supply += supply;
      break;
    case EGRResourceType::kNorth:
      _north_supply += supply;
      break;
    case EGRResourceType::kTrack:
      _track_supply += supply;
      break;
    default:
      LOG_INST.error(Loc::current(), "The resource type is invalid!");
      break;
  }
}

inline void EGRNode::addDemand(EGRResourceType resource_type, double demand)
{
  if (std::isnan(demand)) {
    LOG_INST.error(Loc::current(), "Add nan error!");
  }
  switch (resource_type) {
    case EGRResourceType::kEast:
      _east_demand += demand;
      break;
    case EGRResourceType::kWest:
      _west_demand += demand;
      break;
    case EGRResourceType::kSouth:
      _south_demand += demand;
      break;
    case EGRResourceType::kNorth:
      _north_demand += demand;
      break;
    case EGRResourceType::kTrack:
      _track_demand += demand;
      break;
    default:
      LOG_INST.error(Loc::current(), "The resource type is invalid!");
      break;
  }
}

inline double EGRNode::getCost(EGRResourceType resource_type)
{
  double demand = -1;
  double supply = -1;
  switch (resource_type) {
    case EGRResourceType::kEast:
      demand = _east_demand;
      supply = _east_supply;
      break;
    case EGRResourceType::kWest:
      demand = _west_demand;
      supply = _west_supply;
      break;
    case EGRResourceType::kSouth:
      demand = _south_demand;
      supply = _south_supply;
      break;
    case EGRResourceType::kNorth:
      demand = _east_demand;
      supply = _east_supply;
      break;
    case EGRResourceType::kTrack:
      demand = _east_demand;
      supply = _east_supply;
      break;
    default:
      LOG_INST.error(Loc::current(), "The resource type is invalid!");
      break;
  }
  if (supply - demand > DBL_ERROR && supply > DBL_ERROR) {
    return demand / supply;
  }
  return std::pow(demand + 1 - supply, 3);
}

inline double EGRNode::getRemain(EGRResourceType resource_type)
{
  switch (resource_type) {
    case EGRResourceType::kEast:
      return _east_supply - _east_demand;
    case EGRResourceType::kWest:
      return _west_supply - _west_demand;
    case EGRResourceType::kSouth:
      return _south_supply - _south_demand;
    case EGRResourceType::kNorth:
      return _north_supply - _north_demand;
    case EGRResourceType::kTrack:
      return _track_supply - _track_demand;
    default:
      LOG_INST.error(Loc::current(), "The resource type is invalid!");
      break;
  }
  return 0;
}

inline double EGRNode::getOverflow(EGRResourceType resource_type)
{
  return -getRemain(resource_type);
}

inline double EGRNode::getDemand(EGRResourceType resource_type)
{
  switch (resource_type) {
    case EGRResourceType::kEast:
      return _east_demand;
    case EGRResourceType::kWest:
      return _west_demand;
    case EGRResourceType::kSouth:
      return _south_demand;
    case EGRResourceType::kNorth:
      return _north_demand;
    case EGRResourceType::kTrack:
      return _track_demand;
    default:
      LOG_INST.error(Loc::current(), "The resource type is invalid!");
      break;
  }
  return 0;
}

inline double EGRNode::getSupply(EGRResourceType resource_type)
{
  switch (resource_type) {
    case EGRResourceType::kEast:
      return _east_supply;
    case EGRResourceType::kWest:
      return _west_supply;
    case EGRResourceType::kSouth:
      return _south_supply;
    case EGRResourceType::kNorth:
      return _north_supply;
    case EGRResourceType::kTrack:
      return _track_supply;
    default:
      LOG_INST.error(Loc::current(), "The resource type is invalid!");
      break;
  }
  return 0;
}

inline void EGRNode::setDemand(EGRResourceType resource_type, double demand)
{
  if (std::isnan(demand)) {
    LOG_INST.error(Loc::current(), "Set nan error!");
  }
  switch (resource_type) {
    case EGRResourceType::kEast:
      _east_demand = demand;
      break;
    case EGRResourceType::kWest:
      _west_demand = demand;
      break;
    case EGRResourceType::kSouth:
      _south_demand = demand;
      break;
    case EGRResourceType::kNorth:
      _north_demand = demand;
      break;
    case EGRResourceType::kTrack:
      _track_demand = demand;
      break;
    default:
      LOG_INST.error(Loc::current(), "The resource type is invalid!");
      break;
  }
  return;
}
inline void EGRNode::setSupply(EGRResourceType resource_type, double supply)
{
  if (std::isnan(supply)) {
    LOG_INST.error(Loc::current(), "Set nan error!");
  }
  switch (resource_type) {
    case EGRResourceType::kEast:
      _east_supply = supply;
      break;
    case EGRResourceType::kWest:
      _west_supply = supply;
      break;
    case EGRResourceType::kSouth:
      _south_supply = supply;
      break;
    case EGRResourceType::kNorth:
      _north_supply = supply;
      break;
    case EGRResourceType::kTrack:
      _track_supply = supply;
      break;
    default:
      LOG_INST.error(Loc::current(), "The resource type is invalid!");
      break;
  }
  return;
}

}  // namespace irt