#pragma once
#include "Type.hh"
#include "liberty/Lib.hh"
namespace idb {
using ista::LibPort; using ista::LibArc; using ista::LibArcSet; using ista::LibCell;
using ista::LibDelayTableModel; using ista::LibLutTableTemplate; using ista::LibPowerArcSet;
using ista::LibPowerTableModel; using ista::LibTable; using ista::LibLibrary; using ista::LibObject;
using ista::AnalysisMode; using ista::CapacitiveUnit; using ista::ConvertCapUnit;
using ista::TransType; using ista::TimeUnit; using ista::ModeTransIndex;
}

// bridge iCTS liberty-expression API to iEDA Rust liberty expr
inline void liberty_free_expr(RustLibertyExpr* e) { rust_free_expr(e); }
inline RustLibertyExpr* liberty_get_expr_left(RustLibertyExpr* e) { return rust_get_expr_left(e); }
inline RustLibertyExpr* liberty_get_expr_right(RustLibertyExpr* e) { return rust_get_expr_right(e); }
