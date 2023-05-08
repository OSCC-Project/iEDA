
namespace eval ista {

################################################################

proc define_cmd_args { cmd arglist } {
  variable cmd_args

  set cmd_args($cmd) $arglist
  namespace export $cmd
}
################################################################
proc is_keyword_arg { arg } {
  if { [string length $arg] >= 2 \
     && [string index $arg 0] == "-" \
     && [string is alpha [string index $arg 1]] } {
    return 1
  } else {
    return 0
  }
}

################################################################
proc parse_key_args { cmd arg_var key_var keys {flag_var ""} {flags {}} \
            {unknown_key_is_error 1} } {
  upvar 1 $arg_var args
  upvar 1 $key_var key_value
  upvar 1 $flag_var flag_present
  set args_rtn {}
  while { $args != "" } {
    set arg [lindex $args 0]
    if { [is_keyword_arg $arg] } {
      set key_index [lsearch -exact $keys $arg]
      if { $key_index >= 0 } {
    set key $arg
    if { [llength $args] == 1 } {
      sta_error "$cmd $key missing value."
    }
    set key_value($key) [lindex $args 1]
    set args [lrange $args 1 end]
      } else {
    set flag_index [lsearch -exact $flags $arg]
    if { $flag_index >= 0 } {
      set flag_present($arg) 1
    } else {
      # No exact keyword/flag match found.
      # Try finding a keyword/flag that begins with
      # the same substring.
      set wild_arg "${arg}*"
      set key_index [lsearch -glob $keys $wild_arg]
      if { $key_index >= 0 } {
        set key [lindex $keys $key_index]
        if { [llength $args] == 1 } {
          sta_error "$cmd $key missing value."
        }
        set key_value($key) [lindex $args 1]
        set args [lrange $args 1 end]
      } else {
        set flag_index [lsearch -glob $flags $wild_arg]
        if { $flag_index >= 0 } {
          set flag [lindex $flags $flag_index]
          set flag_present($flag) 1
        } elseif { $unknown_key_is_error } {
          sta_error "$cmd $arg is not a known keyword or flag."
        } else {
          lappend args_rtn $arg
        }
      }
    }
      }
    } else {
      lappend args_rtn $arg
    }
    set args [lrange $args 1 end]
  }
  set args $args_rtn
}

################################################################
proc check_argc_eq0or1 { cmd arglist } {
  set argc [llength $arglist]
  if { $argc != 0 && $argc != 1 } {
    sta_error "$cmd requires zero or one positional arguments."
  }
}

################################################################

proc sta_warn { msg } {
  variable sdc_file
  variable sdc_line
  if { [info exists sdc_file] } {
    puts "Warning: [file tail $sdc_file], $sdc_line $msg"
  } else {
    puts "Warning: $msg"
  }
}

proc sta_error { msg } {
  variable sdc_file
  variable sdc_line
  if { [info exists sdc_file] } {
    error "Error: [file tail $sdc_file], $sdc_line $msg"
  } else {
    error "Error: $msg"
  }
}

proc sta_warn_error { warn_error msg } {
  if { $warn_error == "warn" } {
    sta_warn $msg
  } else {
    sta_error $msg
  }
}

proc check_positive_float { cmd_arg arg } {
  if {!([string is double $arg] && $arg >= 0.0)} {
    sta_error "$cmd_arg '$arg' is not a positive float."
  }
}

proc check_float { cmd_arg arg } {
  if {![string is double $arg]} {
    sta_error "$cmd_arg '$arg' is not a float."
  }
}

################################################################
proc parse_comment_key { keys_var } {
  upvar 1 $keys_var keys
  
  set comment ""
  if { [info exists keys(-comment)] } {
    set comment $keys(-comment)
  }
  return $comment
}
################################################################
proc get_ports_or_pins { pattern } {
  set matches [find_port_pins_matching $pattern 0 0]
  if { $matches != {} } {
    return $matches
  } else {
    return [find_pins_matching $pattern 0 0]
  }
}

################################################################
proc get_port_pins_error { arg_name arglist } {
  set pins {}
  # Copy backslashes that will be removed by foreach.
  set arglilst [string map {\\ \\\\} $arglist]
  foreach arg $arglist {
    if {[llength $arg] > 1} {
      # Embedded list.
      set pins [concat $pins [get_port_pins_error $arg_name $arg]]
    } elseif { [is_object $arg] } {
        set object_type [object_type $arg]
        lappend pins $arg
    } elseif { $arg != {} } {
        set arg_pins [get_ports_or_pins $arg]
      if { $arg_pins != {} } {
        set pins [concat $pins $arg_pins]
      } else {
        sta_error "pin '$arg' not found."
      }
    }
  }
  return $pins
}

################################################################
proc check_nocase_flag { flags_var } {
  upvar 1 $flags_var flags
  if { [info exists flags(-nocase)] && ![info exists flags(-regexp)] } {
    sta_warn "-nocase ignored without -regexp."
  }
}

################################################################
#
# Timing Constraints
#
################################################################

define_cmd_args "get_ports" \
  {[-quiet] [-filter expr] [-regexp] [-nocase] [-of_objects objects] [patterns]}

define_cmd_args "create_clock" \
  {[-name name] [-period period] [-waveform waveform] [-add]\
     [-comment comment] [pins]}

# Find top level design ports matching pattern.
proc get_ports { args } {
  parse_key_args "get_ports" args keys {-of_objects -filter} \
    flags {-regexp -nocase -quiet}
  check_nocase_flag flags
  
  set regexp [info exists flags(-regexp)]
  set nocase [info exists flags(-nocase)]
  # Copy backslashes that will be removed by foreach.
  set patterns [string map {\\ \\\\} [lindex $args 0]]
  set ports {}
  if [info exists keys(-of_objects)] {
    if { $args != {} } {
      sta_warn "patterns argument not supported with -of_objects."
    }
    set nets [get_nets_warn "objects" $keys(-of_objects)]
    foreach net $nets {
      set ports [concat $ports [$net ports]]
    }
  } else {
    # check_argc_eq1 "get_ports" $args
    foreach pattern $patterns {
      set matches [find_ports_matching $pattern $regexp $nocase]
      if { $matches != {} } {
        set ports [concat $ports $matches]
      } else {
      if {![info exists flags(-quiet)]} {
        sta_warn "port '$pattern' not found."
      }
      }
    }
  }
  if [info exists keys(-filter)] {
    set ports [filter_ports1 $keys(-filter) $ports]
  }
  return $ports
}


proc create_clock { args } {
  # parse args
  parse_key_args "create_clock" args \
    keys {-name -period -waveform -comment} \
    flags {-add}

  # check num of arg  
  check_argc_eq0or1 "create_clock" $args

  # arg 
  set argc [llength $args]
  if { $argc == 0 } {
    set pins {}
  } elseif { $argc == 1 } {
    set pins [get_port_pins_error "pins" [lindex $args 0]]
  }
  
  # -name -add
  set add [info exists flags(-add)]
  if [info exists keys(-name)] {
    set name $keys(-name)
  } elseif { $pins != {} } {
    if { $add } {
      sta_error "-add requires -name."
    }
    # Default clock name is the first pin name.
    set name [get_full_name [lindex $pins 0]]
  } else {
    sta_error "-name or port_pin_list must be specified."
  }
  
  # -period
  if [info exists keys(-period)] {
    set period $keys(-period)
    check_positive_float "period" $period
    set period [time_ui_sta $period]
  } else {
    sta_error "missing -period argument."
  }
  
  # -waveform
  if [info exists keys(-waveform)] {
    set wave_arg $keys(-waveform)
    if { [expr [llength $wave_arg] % 2] != 0 } {
      sta_error "-waveform edge_list must have an even number of edge times."
    }
    set first_edge 1
    set prev_edge 0
    set waveform {}
    foreach edge $wave_arg {
      check_float "-waveform edge" $edge
      set edge [time_ui_sta $edge]
      if { $first_edge && $edge > $period } {
    set edge [expr $edge - $period]
      }
      if { !$first_edge && $edge < $prev_edge } {
    sta_warn "adjusting non-increasing clock -waveform edge times."
    set edge [expr $edge + $period]
      }
      if { $edge > [expr $period * 2] } {
    sta_warn "-waveform time greater than two periods."
      }
      lappend waveform $edge
      set prev_edge $edge
      set first_edge 0
    }
  } else {
    set waveform [list 0 [expr $period / 2.0]]
  }
  
  # -comment
  set comment [parse_comment_key keys]  

  # puts $pins
  make_clock $name $pins $add $period $waveform $comment
}

proc test_sdc {} {
  puts "hello sdc test ..."
}

namespace export *

# sta namespace end.
}

