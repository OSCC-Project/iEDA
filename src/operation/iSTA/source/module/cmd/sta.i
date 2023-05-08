/*The shared module name*/
%module ista

////////////////////////////////////////////////////////////////
/*The code below will be copied to the generated wrap file.*/
%{

#include "sdc/SdcClock.hh"
#include "sdc/SdcConstrain.hh"
#include "sta/Sta.hh"
#include <iostream>
#include <set>
#include <vector>

#include "netlist/Pin.hh"
#include "netlist/Port.hh"
#include "netlist/Netlist.hh"

using std::vector;
using std::set;
using ista::Sta;
using ista::SdcConstrain;
using ista::SdcClock;

using ista::Pin;
using ista::Port;
using ista::Netlist;
using ista::DesignObject;

/*The function change the tcl list to std set.*/
template <class T>
std::set<T> *
tclListSet(Tcl_Obj *const source,
       swig_type_info *swig_T,
       Tcl_Interp *interp)
{
  int argc;
  Tcl_Obj **argv;

  if (Tcl_ListObjGetElements(interp, source, &argc, &argv) == TCL_OK
      && argc > 0) {
    std::set<T> * the_set = new std::set<T>;
    for (int i = 0; i < argc; i++) {
      void *obj;
      /*convert tcl obj to c pointer.*/
      SWIG_ConvertPtr(argv[i], &obj, swig_T, false);
      the_set->insert(reinterpret_cast<T>(obj));
    }
    return the_set;
  }
  else
    return nullptr;
}

%}

////////////////////////////////////////////////////////////////

%include "std_vector.i"

using namespace std;



/*convert pin to tcl obj. */
%typemap(out) Pin* {
  /*convert the */
  Tcl_Obj *obj = SWIG_NewInstanceObj($1, $1_descriptor, false);
  Tcl_SetObjResult(interp, obj);
}

/*convert the tcl list Pin to std::set */
%typemap(in) set<Pin*>* {
  $1 = tclListSet<Pin*>($input, SWIGTYPE_p_Pin, interp);
}

/*convert the tcl list Port to std::set */
%typemap(in) set<DesignObject*>* {
  $1 = tclListSet<DesignObject*>($input, SWIGTYPE_p_DesignObject, interp);
}

/*convert the std::set<Pin*>* to tcl obj. */
%typemap(out) set<Pin*>* {
  Tcl_Obj *list = Tcl_NewListObj(0, nullptr);
  set<Pin*>* pins = $1;
  for(auto pin : *pins) {
    Tcl_Obj *obj = SWIG_NewInstanceObj(pin, SWIGTYPE_p_Pin, false);
    Tcl_ListObjAppendElement(interp, list, obj);
  }
  delete pins;
  Tcl_SetObjResult(interp, list);
}

/*convert the std::vector<DesignObject*>* to tcl obj. */
%typemap(out) vector<DesignObject*>* {
  Tcl_Obj *list = Tcl_NewListObj(0, nullptr);
  vector<DesignObject*>* design_objs = $1;
  for(auto design_obj : *design_objs) {
    Tcl_Obj *obj = SWIG_NewInstanceObj(design_obj, SWIGTYPE_p_DesignObject, false);
    Tcl_ListObjAppendElement(interp, list, obj);
  }
  delete design_objs;
  Tcl_SetObjResult(interp, list);
}

/*convert the tcl list to std::vecotr. */
%typemap(in) vector<double>* {
  int argc;
  Tcl_Obj **argv;
  vector<double>* values = nullptr;

  if (Tcl_ListObjGetElements(interp, $input, &argc, &argv) == TCL_OK) {
    if (argc)
      values = new vector<double>;
    for (int i = 0; i < argc; i++) {
      char *arg = Tcl_GetString(argv[i]);
      double value;
      if (Tcl_GetDouble(interp, arg, &value) == TCL_OK) {
        values->push_back(value);
      }	
      else {
        delete values;
        return TCL_ERROR;
      }
    }
  }
  $1 = values;
}

/*make_clock tcl cmd define. */
%inline %{

  // Assumes is_object is true.
  const char *
  object_type(const char *obj)
  {
    // The tcl obj is such as  "_f8a059f7ff7f0000_p_Port".
    // we want to get the last type name.
    return &obj[1 + sizeof(void*) * 2 + 3];
  }

  bool
  is_object(const char *obj)
  {
    // _hexaddress_p_type
    const char *s = obj;
    char ch = *s++;
    if (ch != '_')
      return false;
    while (*s && isxdigit(*s))
      s++;
    if ((s - obj - 1) == sizeof(void*) * 2
        && *s && *s++ == '_'
        && *s && *s++ == 'p'
        && *s && *s++ == '_') {
      while (*s && *s != ' ')
        s++;
      return *s == '\0';
    }
    else {
      return false; 
    }
      
  }

  Pin *
  find_pin(const char *path_name)
  {
    // TODO
    return nullptr;
  }

  DesignObject *
  find_port(const char *path_name)
  {
    // TODO
    return nullptr;
  }

  vector<DesignObject*>*
  find_ports_matching(const char *pattern,
          bool regexp,
          bool nocase)
  {    
    Sta* ista = Sta::getOrCreateSta();
    Netlist* design_netlist = ista->get_netlist();
    vector<DesignObject*>* ports = new vector<DesignObject*>(design_netlist->findPort(pattern, regexp, nocase));
    return ports;
  }

  vector<Pin*>*
  find_port_pins_matching(const char *pattern,
        bool regexp,
        bool nocase)
  {
    vector<Pin*>* pins = new vector<Pin*>;
    return pins;
  }
  
  vector<Pin*>*
  find_pins_matching(const char *pattern,
         bool regexp,
         bool nocase)
  {
    vector<Pin*>* pins = new vector<Pin*>;
    return pins;
  }


  void
  make_clock(const char *name,
      set<DesignObject*>* pins,
      bool add_to_pins,
      double period,
      vector<double>* waveform,
      char *comment)
  {
      std::cout << "create clock " << name << " period " << period << "\n";
      Sta* ista = Sta::getOrCreateSta();
      SdcConstrain* the_constrain = ista->getConstrain();
      SdcClock* the_clock = new SdcClock(name);
      the_clock->set_period(period);
      the_constrain->addClock(the_clock);
      the_clock->set_objs(std::move(*pins));

      auto objs = the_clock->get_objs();
      for (auto obj : objs)
      {
        DLOG_INFO << "clock obj " << obj->get_name();    
      }  

      delete pins;

      DLOG_INFO << "success create clock";
  }

  double
  time_ui_sta(double value)
  {
      return 1.0 * value;
  }

%}

