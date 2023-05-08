#pragma once

#include "database/FPDesign.hh"
#include "database/FPLayout.hh"

namespace ipl::imp {

class DBWrapper
{
 public:
  DBWrapper() = default;
  virtual ~DBWrapper() = default;

  // Layout
  virtual const FPLayout* get_layout() const = 0;

  // Design
  virtual FPDesign* get_design() const = 0;

  // Function
  virtual void writeDef(string file_name) = 0;
  virtual void writeBackSourceDataBase() = 0;
};

}  // namespace ipl::imp