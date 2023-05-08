
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "MPDB.hh"
#include "Setting.hh"
#include "config/Config.hh"
#include "partition/HierPartition.hh"
#include "partition/MPPartition.hh"
#include "simulate_anneal/MPEvaluation.hh"
#include "simulate_anneal/SolutionFactory.hh"
#include "SimulateAnneal.hh"

using std::string;
using std::vector;

namespace ipl::imp {

class MacroPlacer
{
 public:
  MacroPlacer(MPDB* mdb, ipl::Config* config) : _mdb(mdb)
  {
    _mp_config = config->get_mp_config();
    init();
  }
  ~MacroPlacer() = default;
  // open functions
  void runMacroPlacer();

 private:
  void init();
  void updateDensity();
  void setFixedMacro();
  void addHalo();
  void addBlockage();
  void addGuidance();
  void writeSummary(double time);
  void initLocation();
  // data
  MPDB* _mdb;
  Setting* _set;
  MacroPlacerConfig _mp_config;
};

}  // namespace ipl::imp