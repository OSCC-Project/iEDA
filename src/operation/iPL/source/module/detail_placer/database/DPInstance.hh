/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-03-01 17:26:24
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-10 15:00:07
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/database/DPInstance.hh
 * @Description: Instance structure for Detail placement.
 * 
 * Copyright (c) 2023 by iEDA, All Rights Reserved. 
 */
#ifndef IPL_DPINSTANCE_H
#define IPL_DPINSTANCE_H

#include <string>
#include <vector>

#include "data/Rectangle.hh"
#include "data/Orient.hh"

#include "DPCell.hh"
#include "DPPin.hh"

namespace ipl {

class DPCluster;
class DPInterval;
class DPRegion;

enum class DPINSTANCE_STATE
{
  kNone,
  kUnPlaced,
  kPlaced,
  kFixed
};

class DPInstance
{
public:
    DPInstance() = delete;
    explicit DPInstance(std::string name);
    DPInstance(const DPInstance&) = delete;
    DPInstance(DPInstance&&) = delete;
    ~DPInstance();

    DPInstance& operator=(const DPInstance&) = delete;
    DPInstance& operator=(DPInstance&&) = delete;

    // getter
    std::string get_name() const { return _name;}
    DPCell* get_master() const { return _master;}
    std::vector<DPPin*> get_pin_list() const { return _pin_list;}
    Rectangle<int32_t> get_shape() const { return _shape;}
    Point<int32_t> get_coordi() const { return _shape.get_lower_left();}
    Orient get_orient() const { return _orient;}
    DPINSTANCE_STATE get_state() const { return _state;}
    DPRegion* get_belong_region() const { return _belong_region;}
    int32_t get_internal_id() const { return _cluster_internal_id;}
    DPCluster* get_belong_cluster() const { return _belong_cluster;}
    double get_weight() const { return _weight;}

    // setter
    void set_master(DPCell* master) { _master = master;}
    void add_pin(DPPin* pin) { _pin_list.push_back(pin);}
    void set_shape(Rectangle<int32_t> shape) {_shape = std::move(shape);}
    void set_orient(Orient orient) { _orient = std::move(orient);}
    void set_state(DPINSTANCE_STATE state) { _state = state;}
    void set_belong_region(DPRegion* region) { _belong_region = region;}
    void set_internal_id(int32_t internal_id) { _cluster_internal_id = internal_id;}
    void set_belong_cluster(DPCluster* belong_cluster) { _belong_cluster = belong_cluster;}
    void set_weight(double weight) { _weight = weight;}

    // function
    void updateCoordi(int32_t llx, int32_t lly);
    std::pair<int32_t, int32_t> calInstPinModifyOffest(DPPin* pin);

private:
    std::string _name;
    DPCell* _master;
    std::vector<DPPin*> _pin_list;
    Rectangle<int32_t> _shape;
    Orient _orient;
    DPINSTANCE_STATE _state;
    DPRegion* _belong_region;
    int32_t _cluster_internal_id;
    DPCluster* _belong_cluster;
    double _weight;

    void updatePinsCoordi();
};
}
#endif