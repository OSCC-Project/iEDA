# iPNP: Power Network Planning

## Overview

iPNP is a chip power network layout tool that balances congestion and IR drop, powered by a simulated annealing algorithm. It supports functionalities including power network generation, IR drop analysis, and congestion evaluation.

![Input Image Description](docs/iPNP%20introduction.png)

## Supported Features

1. **Design File Processing**: Reads DEF/LEF files to construct an IDB builder, adds power rails/vias, and exports DEF files.  
2. **Region Partitioning**: Divides the chip layout into regions with customizable power templates to control rail density.  
3. **Power Routing**: Implements power rails and vias across layers based on predefined templates.  
4. **Pre-layout Acceleration**: Integrates with iPL tool for rapid initial placement.  
5. **Congestion Evaluation**: Uses EGR tool to calculate overflow per layer.  
6. **IR Drop Analysis**: Employs IR evaluation tool to compute voltage drop for each instance.  
7. **Optimization Engine**: Utilizes simulated annealing to iteratively minimize overflow and IR drop scores.

---

## Installation

iPNP is located in the `src/operation/iPNP/` directory. After compilation, the executable resides in `bin/`.

---

## Usage

### Configuration File Preparation

Create a JSON-formatted configuration file (e.g., `pnp_config.json`) containing all runtime parameters.

### Running iPNP

Multiple execution modes are supported:

**Command Line Mode**:
```bash
cd bin/
./iPNP -c /path/to/pnp_config.json
```

**Specify Output File**:
```bash
./iPNP -c /path/to/pnp_config.json -o /path/to/output.def
```

**Interactive Mode**:
```bash
./iPNP -i
```

**Run TCL Script**:
```bash
./iPNP -s /path/to/script.tcl
```

**Help Menu**:
```bash
./iPNP -h
```

![Help Menu Screenshot](docs/iPNP_help.png)

### TCL Commands

Use the following command in interactive mode:
```tcl
run_pnp -config /path/to/pnp_config.json
```

---

## Configuration File Details

The JSON configuration file comprises the following sections:

### Design Specifications
```json
"design": {
  "lef_files": ["file1.lef", "file2.lef"],  // Technology/library files
  "def_file": "input.def",                // Input DEF file
  "output_def_file": "output.def",        // Output DEF path
  "sdc_file": "timing.sdc"                // Timing constraints
}
```

### Library Files
```json
"lib": {
  "liberty_files": ["lib1.lib", "lib2.lib"]  // Liberty library list
}
```

### Timing Analysis
```json
"timing": {
  "design_workspace": "/path/to/workspace"  // IR drop output directory
}
```

### Power Network
```json
"power": {
  "power_net_name": "VDD"  // Power net identifier
}
```

### Congestion Evaluation
```json
"egr": {
  "map_path": "/path/to/map"  // Congestion map output path
}
```

### Grid Configuration
```json
"grid": {
  "power_layer": [9, 8, 7, 6, 5, 4, 3],  // Power layers (high to low)
  "ho_region_num": 2,                    // Horizontal regions
  "ver_region_num": 2                    // Vertical regions
}
```

### Power Templates
```json
"templates": {
  "horizontal": [  // Horizontal rail templates
    {
      "width": 8000.0,     // Rail width
      "pg_offset": 1600.0, // VDD/VSS spacing
      "space": 19200.0,    // Inter-rail spacing
      "offset": 8000.0     // Offset from die edge
    }
  ],
  "vertical": [  // Vertical rail templates
    {
      "width": 8000.0,
      "pg_offset": 1600.0,
      "space": 19200.0,
      "offset": 8000.0
    }
  ]
}
```

![Template Configuration Example](docs/template_info.png)

### Simulated Annealing Parameters
```json
"simulated_annealing": {
  "initial_temp": 100.0,          // Initial temperature
  "cooling_rate": 0.95,           // Cooling rate
  "min_temp": 0.1,                // Minimum temperature
  "iterations_per_temp": 10,      // Iterations per temperature
  "ir_drop_weight": 0.6,          // IR drop weight
  "overflow_weight": 0.4,         // Overflow weight
  "modifiable_layer_min": 3,      // Adjustable layer range (min)
  "modifiable_layer_max": 6       // Adjustable layer range (max)
}
```

---

## Configuration Example

```json
{
  "design": {
    "lef_files": [...],  // List of LEF files
    "def_file": "input.def",
    "output_def_file": "output.def",
    "sdc_file": "timing.sdc"
  },
  "lib": { "liberty_files": [...] },
  "timing": { "design_workspace": "/path/to/ir_results" },
  "power": { "power_net_name": "VDD" },
  "egr": { "map_path": "/path/to/congestion_maps" },
  "grid": {
    "power_layer": [9,8,7,6,5,4,3],
    "ho_region_num": 2,
    "ver_region_num": 2
  },
  "simulated_annealing": {
    "initial_temp": 100.0,
    "cooling_rate": 0.95,
    "min_temp": 0.1,
    "iterations_per_temp": 10,
    "ir_drop_weight": 0.6,
    "overflow_weight": 0.4,
    "modifiable_layer_min": 3,
    "modifiable_layer_max": 6
  },
  "templates": {
    "horizontal": [
      {
        "width": 8000.0,
        "pg_offset": 1600.0,
        "space": 19200.0,
        "offset": 8000.0
      },
      {
        "width": 8000.0,
        "pg_offset": 1600.0,
        "space": 38400.0,
        "offset": 8000.0
      },
      {
        "width": 8000.0,
        "pg_offset": 1600.0,
        "space": 38400.0,
        "offset": 27200.0
      }
    ],
    "vertical": [
      {
        "width": 8000.0,
        "pg_offset": 1600.0,
        "space": 19200.0,
        "offset": 8000.0
      },
      {
        "width": 8000.0,
        "pg_offset": 1600.0,
        "space": 38400.0,
        "offset": 8000.0
      },
      {
        "width": 8000.0,
        "pg_offset": 1600.0,
        "space": 38400.0,
        "offset": 27200.0
      }
    ]
  }
}
```

---

## Output Files

Upon completion, iPNP generates:
1. DEF file with power grid (`output_def_file`)
2. IR drop analysis report
3. Congestion evaluation results

---

## Notes

1. Verify all file paths exist and are accessible.
2. Power layers must be listed from highest to lowest.
3. Template parameters are specified in database units (DBU).