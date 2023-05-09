#pragma once

#include <map>
#include <string>
#include <vector>

#include "DBWrapper.hh"
#include "IPLDBWDatabase.hh"

namespace ipl::imp {

class IPLDBWrapper : public DBWrapper
{
 public:
  IPLDBWrapper() = delete;
  explicit IPLDBWrapper(ipl::PlacerDB* ipl_db);
  ~IPLDBWrapper() override;

  // Layout
  const FPLayout* get_layout() const { return _iplw_database->_layout; }

  // Design
  FPDesign* get_design() const { return _iplw_database->_design; }

  // Function
  void writeDef(string file_name) override{};
  void writeBackSourceDataBase() override;

 private:
  IPLDBWDatabase* _iplw_database;

  void wrapIPLData();
  void wrapLayout(const ipl::Layout* ipl_layout);
  void wrapDesign(ipl::Design* ipl_design);
  void wrapInstancelist(ipl::Design* ipl_design);
  void wrapNetlist(ipl::Design* ipl_design);
  FPPin* wrapPin(ipl::Pin* ipl_pin);
};
}  // namespace ipl::imp
