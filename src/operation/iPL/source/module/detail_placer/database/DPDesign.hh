/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-03-03 19:04:14
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-10 11:15:02
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/database/DPDesign.hh
 * @Description: Design of detail placement
 * 
 * Copyright (c) 2023 by iEDA, All Rights Reserved. 
 */
#ifndef IPL_DPDESIGN_H
#define IPL_DPDESIGN_H

#include <string>
#include <vector>
#include <map>

#include "data/Instance.hh"

#include "DPInstance.hh"
#include "DPNet.hh"
#include "DPPin.hh"
#include "DPCluster.hh"

namespace ipl {
class DPDesign
{
public:
    DPDesign();
    
    DPDesign(const DPDesign&) = delete;
    DPDesign(DPDesign&&) = delete;
    ~DPDesign();

    DPDesign& operator=(const DPDesign&) = delete;
    DPDesign& operator=(DPDesign&&) = delete;

    // getter
    const std::vector<DPInstance*> get_inst_list() const { return _dpInstance_list;}
    const std::vector<DPNet*> get_net_list() const { return _dpNet_list;}
    const std::vector<DPPin*> get_pin_list() const { return _dpPin_list;}

    // setter
    void add_instance(DPInstance* inst);
    void add_net(DPNet* net);
    void add_pin(DPPin* pin);
    void add_cluster(DPCluster* cluster);
    
    // function
    void connectInst(DPInstance* dp_inst, Instance* pl_inst);

    DPInstance* find_instance(std::string inst_name);
    DPNet* find_net(std::string net_name);
    DPPin* find_pin(std::string pin_name);
    DPCluster* find_cluster(std::string cluster_name);

    void clearClusterInfo();
    void deleteCluster(std::string cluster_name);

    void writeBackToPL(int32_t shift_x, int32_t shift_y);

    int64_t calInstAffectiveHPWL(DPInstance* inst);

private:
    std::vector<DPInstance*> _dpInstance_list;
    std::vector<DPNet*> _dpNet_list;
    std::vector<DPPin*> _dpPin_list;

    std::map<std::string, DPInstance*> _dpInstance_map;
    std::map<std::string, DPNet*> _dpNet_map;
    std::map<std::string, DPPin*> _dpPin_map;

    std::map<std::string, DPCluster*> _dpCluster_map;

    std::map<DPInstance*, Instance*> _dpInst_inst_map;
    std::map<Instance*, DPInstance*> _inst_dpInst_map;
};
}
#endif