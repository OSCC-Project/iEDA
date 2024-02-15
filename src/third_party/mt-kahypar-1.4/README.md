<h1 align="center">Mt-KaHyPar - Multi-Threaded Karlsruhe Graph and Hypergraph Partitioner</h1>

License|Linux & Windows Build|Code Coverage|Zenodo
:--:|:--:|:--:|:--:
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)|[![Build Status](https://github.com/kahypar/mt-kahypar/actions/workflows/master_ci.yml/badge.svg)](https://github.com/kahypar/mt-kahypar/actions/workflows/mt_kahypar_ci.yml)|[![codecov](https://codecov.io/gh/kahypar/mt-kahypar/branch/master/graph/badge.svg?token=sNWRRtXZjI)](https://codecov.io/gh/kahypar/mt-kahypar)|[![DOI](https://zenodo.org/badge/205879380.svg)](https://zenodo.org/badge/latestdoi/205879380)

Table of Contents
-----------

   * [About Mt-KaHyPar](#about-mt-kahypar)
   * [Features](#features)
   * [Requirements](#requirements)
   * [Building Mt-KaHyPar](#building-mt-kahypar)
   * [Running Mt-KaHyPar](#running-mt-kahypar)
   * [The C Library Interface](#the-c-library-interface)
   * [The Python Library Interface](#the-python-library-interface)
   * [Supported Objective Functions](#supported-objective-functions)
   * [Custom Objective Functions](#custom-objective-functions)
   * [Improving Compile Times](#improving-compile-times)
   * [Licensing](#licensing)

About Mt-KaHyPar
-----------
Mt-KaHyPar is a shared-memory algorithm for partitioning graphs and hypergraphs. The balanced (hyper)graph partitioning problem
asks for a partition of the node set of a (hyper)graph into *k* disjoint blocks of roughly the same size (usually a small imbalance
is allowed by at most 1 + ε times the average block weight), while simultanously minimizing an objective function defined on the (hyper)edges. Mt-KaHyPar can optimize the cut-net, connectivity, sum-of-external-degree, and Steiner tree metric (see [Supported Objective Functions](#supported-objective-functions)).

<img src="https://cloud.githubusercontent.com/assets/484403/25314222/3a3bdbda-2840-11e7-9961-3bbc59b59177.png" alt="alt text" width="50%" height="50%"><img src="https://cloud.githubusercontent.com/assets/484403/25314225/3e061e42-2840-11e7-860c-028a345d1641.png" alt="alt text" width="50%" height="50%">

The highest-quality configuration of Mt-KaHyPar computes partitions that are on par with those produced by the best sequential partitioning algorithms, while being almost an order of magnitude faster with only *ten* threads (e.g., when compared to [KaFFPa](https://github.com/KaHIP/KaHIP) or [KaHyPar](https://kahypar.org/)). Besides our high-quality configuration, we provide several other faster configurations that are already able to outperform most of the existing partitioning algorithms with regard to solution quality and running time.
The figure below summarizes the time-quality trade-off of different hypergraph (left, connectivity metric) and graph partitioning algorithms (right, cut-net metric). The plot is based on an experiment with over 800 graphs and hypergraphs and relates the average solution quality and running time of each algorithm to the best achievable results. Points on the lower-left are considered better. Partially transparent markers indicate solvers producing more than 15% infeasible partitions (either imbalanced or timeout). For more details, we refer the reader to our [publications](#licensing).

![time_quality_trade_off](https://github.com/kahypar/mt-kahypar/assets/9654047/a5cc1c41-5ca5-496a-ba50-91965e73226b)

Features
-----------

Besides its fast and high-quality partitioning algorithm, Mt-KaHyPar provides many other useful features:

- **Scalability**: Mt-KaHyPar has excellent scaling behavior (up to 25 with 64 threads), while increasing the number of threads does not adversely affect the solution quality.
- **Deterministic Partitioning**: Mt-KaHyPar offers a deterministic partitioning algorithm, ensuring consistent solutions for the same input and random seed.
- **Large K Partitioning**: We provide a partitioning configuration for partitioning (hyper)graphs into a large number of blocks (e.g., k > 1024).
- **Graph Partitioning**: Mt-KaHyPar includes optimized data structures for graph partitioning, achieving a speedup by a factor of two for plain graphs.
- **Objective Functions**: Mt-KaHyPar can optimize the cut-net, connectivity, and sum-of-external-degree metric (for more details, see [Supported Objective Functions](#supported-objective-functions))
- **Mapping (Hyper)Graphs Onto Graphs**: In many applications of (hyper)graph partitioning, the blocks of a partition need to be assigned to architectures that can be represented as graphs. For instance, in parallel computations, the blocks may be assigned to processors on a computing cluster interconnected via communication links. It becomes advantageous to position nodes close to each other on the target graph if they are adjacent in the original (hyper)graph. However, conventional objective functions do not consider the topology of the target graph during partitioning. We therefore provide a mode that maps the nodes of a (hyper)graph onto the nodes of a target graph. During this process, the partitioning algorithm optimizes the Steiner tree metric. The objective here is to minimize the total weight of all minimal Steiner trees induced by the (hyper)edges of the hypergraph on the target graph. For more information about this metric, we refer the reader to the [Supported Objective Functions](#supported-objective-functions) section.
- **Fixed Vertices**: Fixed vertices are nodes that are preassigned to a particular block and are not allowed to change their block during partitioning.

Requirements
-----------

The Multi-Threaded Karlsruhe Graph and Hypergraph Partitioning Framework requires:

  - A 64-bit Linux or Windows operating system.
  - A modern, ![C++17](https://img.shields.io/badge/C++-17-blue.svg?style=flat)-ready compiler such as `g++` version 7 or higher, `clang` version 11.0.3 or higher, or `MinGW` compiler on Windows (tested with version 12.1).
 - The [cmake][cmake] build system (>= 3.16).
 - The [Boost - Program Options][Boost.Program_options] library and the boost header files (>= 1.48).
   If you don't want to install boost by yourself, you can add the `-DKAHYPAR_DOWNLOAD_BOOST=On` flag
   to the cmake command to download, extract, and build the neccessary dependencies automatically.
 - The [Intel Thread Building Blocks][tbb] library (TBB, minimum required version is OneTBB 2021.5.0).
   If you don't want to install TBB by yourself, you can add the `-DKAHYPAR_DOWNLOAD_TBB=On` flag (only available on Linux)
   to the cmake command to download oneTBB 2021.7.0 and extract the neccessary dependencies automatically.
   Mt-KaHyPar also compiles with older version of TBB. However, we observed unexpected behavior of a TBB function
   on which we rely on which causes on our side a segmentation fault in really rare cases. If you want to ignore these
   warning, you can add `-DKAHYPAR_ENFORCE_MINIMUM_TBB_VERSION=OFF` to the cmake build command.
 - The [Portable Hardware Locality][hwloc] library (hwloc)

### Linux

The following command will install most of the required dependencies on a Ubuntu machine:

    sudo apt-get install libtbb-dev libhwloc-dev libboost-program-options-dev

### Windows

The following instructions setup the environment used to build Mt-KaHyPar on Windows machines:

  1. Download and install [MSYS2][MSYS2] from the official website (https://www.msys2.org/).
  2. Launch the `MSYS2 MinGW x64` terminal.
  3. Update the package manager database by running the following command:

    pacman -Syu

  4. The following command will then install all required dependencies:

    pacman -S make mingw-w64-x86_64-cmake mingw-w64-x86_64-gcc mingw-w64-x86_64-python3 mingw-w64-x86_64-tbb

  5. Rename `libtbb12.dll.a` to `libtbb.dll.a` which is located in `C:\msys64\mingw64\lib` (or `/mingw64/lib` within the
     `MSYS2 MinGW x64` terminal)

Please **note** that Mt-KaHyPar was primarily tested and evaluated on Linux machines. While a Windows build has been provided and tested on `MSYS2` using `pacman` to install the required dependencies, we cannot provide any performance guarantees or ensure that the Windows version is free of bugs. At this stage, Windows support is experimental. We are happy to accept contributions to improve Windows support.

Building Mt-KaHyPar
-----------

To build Mt-KaHyPar, you can run the `build.sh` script (creates a `build` folder) or use the following commands:

1. Clone the repository including submodules:

   ```git clone --depth=2 --recursive https://github.com/kahypar/mt-kahypar.git```

2. Create a build directory: `mkdir build && cd build`
3. *Only on Windows machines*: `export CMAKE_GENERATOR="MSYS Makefiles"`
3. Run cmake: `cmake .. -DCMAKE_BUILD_TYPE=RELEASE` (on Windows machines add `-DKAHYPAR_DOWNLOAD_BOOST=On`)
4. Run make: `make MtKaHyPar -j`

The build produces the executable `MtKaHyPar`, which can be found in `build/mt-kahypar/application/`.

Running Mt-KaHyPar
-----------

To partition a **hypergraph** with our default configuration, you can use the following command:

    ./mt-kahypar/application/MtKaHyPar -h <path-to-hgr> --preset-type=default -t <# threads> -k <# blocks> -e <imbalance (e.g. 0.03)> -o km1

### Partitioning Configurations

Mt-KaHyPar provides several partitioning configurations with different time-quality trade-offs. The configurations are stored in `ini` files located in the `config` folder. However, we recomment to use the `--preset-type` command line parameter to run Mt-KaHyPar with a specific partitioning configuration:

    --preset-type=<large_k/deterministic/default/quality/highest_quality>

- `large_k`: configuration for partitioning (hyper)graphs into a large number of blocks (e.g. >= 1024 blocks, `config/large_k_preset.ini`)
- `deterministic`: configuration for deterministic partitioning (`config/deterministic_preset.ini`, corresponds to Mt-KaHyPar-SDet in our publications)
- `default`: computes good partitions very fast (`config/default_preset.ini`, corresponds to Mt-KaHyPar-D in our publications)
- `quality`: computes high-quality partitions (`config/quality_preset.ini`, corresponds to Mt-KaHyPar-D-F in our publications)
- `highest_quality`: highest-quality configuration (`config/quality_flow_preset.ini`, corresponds to Mt-KaHyPar-Q-F in our publications)

The presets can be ranked from lowest to the highest-quality as follows: `large_k`, `deterministic`,
`default`, `quality`, and `highest_quality`.
We recommend to use the `default` configuration to compute good partitions very fast and the `quality` configuration to compute high-quality solutions. The `highest_quality` configuration computes better partitions than our `quality` configuration by 0.5% on average at the cost of a two times longer running time for medium-sized instances (up to 100 million pins). When you have to partition a (hyper)graph into a large number of blocks (e.g., >= 1024 blocks), you can use our `large_k` configuration. However, we only recommend to use this if you experience high running times with one of our other configurations as this can significantly worsen the partitioning quality.

### Objective Functions

Mt-KaHyPar can optimize the cut-net, connectivity, and sum-of-external-degree metric (see [Supported Objective Functions](#supported-objective-functions)).

    -o <cut/km1/soed>

### Mapping (Hyper)Graphs onto Graphs

To map a **(hyper)graph** onto a **target graph** with Mt-KaHyPar, you can add the following command line parameters to the partitioning call:

    -g <path-to-target-graph> -o steiner_tree

The target graph is expected to be in [Metis format](http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf). The nodes of the (hyper)graph are then mapped onto the nodes of the target graph, while optimizing the Steiner tree metric (see [Supported Objective Functions](#supported-objective-functions)).

### Graph Partitioning

To partition a **graph** with Mt-KaHyPar, you can add the following command line parameters to the partitioning call:

    -h <path-to-graph> --instance-type=graph --input-file-format=<metis/hmetis> -o cut

Mt-KaHyPar then uses optimized data structures for graph partitioning, which speedups the partitioning time by a factor of two compared to our hypergraph partitioning code. Per default, we expect the input in [hMetis format](http://glaros.dtc.umn.edu/gkhome/fetch/sw/hmetis/manual.pdf), but you can read graph files in [Metis format](http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf) via `--input-file-format=metis`.

### Fixed Vertices

Fixed vertices are nodes that are preassigned to particular block and are not allowed to change their block during partitioning. Mt-KaHyPar reads fixed vertices from a file in the [hMetis fix file format](http://glaros.dtc.umn.edu/gkhome/fetch/sw/hmetis/manual.pdf), which can be provided via the following command line parameter:

    -f <path-to-fixed-vertex-file>

Note that fixed vertices are only supported in our `default`, `quality`, and `highest_quality` configuration.

### Individual Target Block Weights

Per default, Mt-KaHyPar enforces that the weight of each block must be smaller than the average block weight (weight of the hypergraph divided by the number of blocks) times (1 + ε). However, you can provide individual target block weights for each block via

    --part-weights=weight_of_block_0 weight_of_block_1 ... weight_of_block_k

Note that the sum of all individual target block weights must be larger than the total weight of all nodes.

### Write Partition to Output File

To enable writing the partition to a file after partitioning, you can add the following command line parameters to the partitioning call:

    --write-partition-file=true --partition-output-folder=<path/to/folder>

The partition file name is generated automatically based on parameters such as `k`, `imbalance`, `seed` and the input file name and will be located in the folder specified by `--partition-output-folder`. If you do not provide an partition output folder, the partition file will be placed in the same folder as the input hypergraph file.

### Other Useful Program Options

There are several useful options that can provide you with additional insights during and after the partitioning process:

- `--verbose=true`: Displays detailed information on the partitioning process
- `--show-detailed-timings=true`: Shows detailed subtimings of each phase of the algorithm at the end of partitioning
- `--enable-progress-bar=true`: Shows a progess bar during the coarsening and refinement phase


If you want to change other configuration parameters manually, please run `--help` for a detailed description of the different program options.

The C Library Interface
-----------

We provide a simple C-style interface to use Mt-KaHyPar as a library.  The library can be built and installed via

```sh
make install.mtkahypar # use sudo (Linux) or run shell as an adminstrator (Windows) to install system-wide
```

Note: When installing locally, the build will exit with an error due to missing permissions.
However, the library is still built successfully and is available in the build folder.

The library interface can be found in `include/libmtkahypar.h` with a detailed documentation. We also provide several examples in the folder `lib/examples` that show how to use the library.

Here is a short example how you can partition a hypergraph using our library interface:

```cpp
#include <memory>
#include <vector>
#include <iostream>
#include <thread>

#include <libmtkahypar.h>

int main(int argc, char* argv[]) {

  // Initialize thread pool
  mt_kahypar_initialize_thread_pool(
    std::thread::hardware_concurrency() /* use all available cores */,
    true /* activate interleaved NUMA allocation policy */ );

  // Setup partitioning context
  mt_kahypar_context_t* context = mt_kahypar_context_new();
  mt_kahypar_load_preset(context, DEFAULT /* corresponds to MT-KaHyPar-D */);
  // In the following, we partition a hypergraph into two blocks
  // with an allowed imbalance of 3% and optimize the connective metric (KM1)
  mt_kahypar_set_partitioning_parameters(context,
    2 /* number of blocks */, 0.03 /* imbalance parameter */,
    KM1 /* objective function */);
  mt_kahypar_set_seed(42 /* seed */);
  // Enable logging
  mt_kahypar_set_context_parameter(context, VERBOSE, "1");

  // Load Hypergraph for DEFAULT preset
  mt_kahypar_hypergraph_t hypergraph =
    mt_kahypar_read_hypergraph_from_file(
      "path/to/hypergraph/file", DEFAULT, HMETIS /* file format */);

  // Partition Hypergraph
  mt_kahypar_partitioned_hypergraph_t partitioned_hg =
    mt_kahypar_partition(hypergraph, context);

  // Extract Partition
  std::unique_ptr<mt_kahypar_partition_id_t[]> partition =
    std::make_unique<mt_kahypar_partition_id_t[]>(mt_kahypar_num_hypernodes(hypergraph));
  mt_kahypar_get_partition(partitioned_hg, partition.get());

  // Extract Block Weights
  std::unique_ptr<mt_kahypar_hypernode_weight_t[]> block_weights =
    std::make_unique<mt_kahypar_hypernode_weight_t[]>(2);
  mt_kahypar_get_block_weights(partitioned_hg, block_weights.get());

  // Compute Metrics
  const double imbalance = mt_kahypar_imbalance(partitioned_hg, context);
  const double km1 = mt_kahypar_km1(partitioned_hg);

  // Output Results
  std::cout << "Partitioning Results:" << std::endl;
  std::cout << "Imbalance         = " << imbalance << std::endl;
  std::cout << "Km1               = " << km1 << std::endl;
  std::cout << "Weight of Block 0 = " << block_weights[0] << std::endl;
  std::cout << "Weight of Block 1 = " << block_weights[1] << std::endl;

  mt_kahypar_free_context(context);
  mt_kahypar_free_hypergraph(hypergraph);
  mt_kahypar_free_partitioned_hypergraph(partitioned_hg);
}
```

To compile the program using `g++` run:

```sh
g++ -std=c++17 -DNDEBUG -O3 your_program.cc -o your_program -lmtkahypar
```

To execute the binary, you need to ensure that the installation directory
(probably `/usr/local/lib` (Linux) and `C:\Program Files (x86)\MtKaHyPar\bin` (Windows) for system-wide installation)
is included in the dynamic library path.
The path can be updated on Linux with:

```sh
LD_LIBRARY_PATH="$LD_LIBRARY_PATH;/usr/local/lib"
export LD_LIBRARY_PATH
```

On Windows, add `C:\Program Files (x86)\KaHyPar\bin` to `PATH` in the environment variables settings.

To remove the library from your system use the provided uninstall target:

```sh
make uninstall-mtkahypar
```

**Note** that we internally use different data structures to represent a (hyper)graph based on the corresponding configuration (`mt_kahypar_preset_type_t`). The `mt_kahypar_hypergraph_t` structure stores a pointer to this data structure and also a type description. Therefore, you can not partition a (hyper)graph with all available configurations once it is loaded or constructed. However, you can check the compatibility of a hypergraph with a configuration with the following code:

```cpp
mt_kahypar_context_t context = mt_kahypar_context_new();
mt_kahypar_load_preset(context, QUALITY);
// Check if the hypergraph is compatible with the QUALITY preset
if ( mt_kahypar_check_compatibility(hypergraph, QUALITY) ) {
  mt_kahypar_partitioned_hypergraph_t partitioned_hg =
    mt_kahypar_partition(hypergraph, context);
}
```

The Python Library Interface
-----------

You can install the Python library interface via

```sh
make mtkahypar_python
```

This will create a shared library in the `build/python` folder (`mtkahypar.so` on Linux and `mtkahypar.pyd` on Windows).
Copy the libary to your Python project directory to import Mt-KaHyPar as a Python module.

A documentation of the Python module can be found in `python/module.cpp`, or by importing the module (`import mtkahypar`) and calling `help(mtkahypar)` in Python. We also provide several examples that show how to use the Python interface in the folder `python/examples`.

Here is a short example how you can partition a hypergraph using our Python interface:

```py
import multiprocessing
import mtkahypar

# Initialize thread pool
mtkahypar.initializeThreadPool(multiprocessing.cpu_count()) # use all available cores

# Setup partitioning context
context = mtkahypar.Context()
context.loadPreset(mtkahypar.PresetType.DEFAULT) # corresponds to Mt-KaHyPar-D
# In the following, we partition a hypergraph into two blocks
# with an allowed imbalance of 3% and optimize the connectivity metric
context.setPartitioningParameters(
  2,                       # number of blocks
  0.03,                    # imbalance parameter
  mtkahypar.Objective.KM1) # objective function
mtkahypar.setSeed(42)      # seed
context.logging = True     # enables partitioning output

# Load hypergraph from file
hypergraph = mtkahypar.Hypergraph(
  "path/to/hypergraph/file", # hypergraph file
  mtkahypar.FileFormat.HMETIS) # hypergraph is stored in hMetis file format

# Partition hypergraph
partitioned_hg = hypergraph.partition(context)

# Output metrics
print("Partition Stats:")
print("Imbalance = " + str(partitioned_hg.imbalance()))
print("km1       = " + str(partitioned_hg.km1()))
print("Block Weights:")
print("Weight of Block 0 = " + str(partitioned_hg.blockWeight(0)))
print("Weight of Block 1 = " + str(partitioned_hg.blockWeight(1)))
```

We also provide an optimized graph data structure for partitioning plain graphs. The following example loads and partitions a graph:

```py
# Load graph from file
graph = mtkahypar.Graph(
  "path/to/graph/file", # graph file
  mtkahypar.FileFormat.METIS) # graph is stored in Metis file format

# Partition graph
partitioned_graph = graph.partition(context)
```
**Note** that when you want to partition a hypergraph into large number of blocks (e.g., k > 1024), you can use our `LARGE_K` confguration and the `partitionIntoLargeK(...)` function of the hypergraph object. If you use an other configuration for large k partitioning, you may run into memory and running time issues during partitioning. However, this depends on the size of the hypergraph and the memory capacity of your target machine. For partitioning plain graphs, you can load the `LARGE_K` configuration, but you can still use the `partition(...)` function of the graph object. Here is an example that partitions a hypergraph into 1024 blocks:

```py
# Setup partitioning context
context = mtkahypar.Context()
context.loadPreset(mtkahypar.PresetType.LARGE_K)
# In the following, we partition a hypergraph into 1024 blocks
# with an allowed imbalance of 3% and optimize the connectivity metric
context.setPartitioningParameters(1024, 0.03, mtkahypar.Objective.KM1, 42)

# Load and partition hypergraph
hypergraph = mtkahypar.Hypergraph("path/to/hypergraph/file", mtkahypar.FileFormat.HMETIS)
partitioned_hg = hypergraph.partitionIntoLargeK(context)
```

Supported Objective Functions
-----------

Mt-KaHyPar can optimize several objective functions which we explain in the following in more detail.

**Cut-Net Metric**

![cut_net](https://github.com/kahypar/mt-kahypar/assets/9654047/bc7fc7c7-8ac4-4711-8aec-d0526ef2452c)

The cut-net metric is defined as total weight of all nets spanning more than one block of the partition Π (also called *cut nets*).


**Connectivity Metric**

![connectivity](https://github.com/kahypar/mt-kahypar/assets/9654047/1c586ff4-63c3-4260-9ef5-98a76578be46)

The connectivity metric additionally multiplies the weight of each cut net with the number of blocks λ(e) spanned by that net minus one. Thus, the connectivity metric tries to minimize the number of blocks connected by each net.


**Sum-of-External-Degree Metric**

![soed](https://github.com/kahypar/mt-kahypar/assets/9654047/4006fb4c-ac85-452e-a0d9-93d4dc7842ad)

The sum-of-external-degree metric is similar to the connectivity metric, but does not subtract one from the number of blocks λ(e) spanned by a net. A peculiarity of this objective function is that removing a net from the cut reduces the metric by 2ω(e), while reducing the connectivity by one reduces the metric only by ω(e). Thus, the objective function prefers removing nets from the cut, while as secondary criteria it tries to reduce the connectivity of the nets.

**Steiner Tree Metric**

![steiner_tree](https://github.com/kahypar/mt-kahypar/assets/9654047/926ef7d7-bb6b-4959-af0c-75ebd6f6299f)

The Steiner tree metric is the most versatile metric that we provide at the moment. A Steiner tree is a tree with minimal weight that connects a subset of the nodes on a graph (a more detailed definition can be found [here][SteinerTrees]). For a subset with exactly two nodes, finding a Steiner tree reverts to computing the shortest path between the two nodes. When optimizing the Steiner tree metric, we map the node set of a hypergraph H onto the nodes of a target graph G. The objective is to minimize the total weight of all Steiner trees induced by the nets of H on G.
For a net e, dist(Λ(e)) is the weight of the minimal Steiner tree connecting the blocks Λ(e) spanned by net e on G. The Steiner tree metric can be used to accurately model wire-lengths in VLSI design or communication costs in distributed systems when some processors do not communicate with each other directly or with different speeds.

Note that finding a Steiner tree is an NP-hard problem. We therefore enforce a strict upper bound on the number of nodes of the target graph G which are 64 nodes at the moment. If you want to map a hypergraph onto larger targer graphs, you can use recursive multisectioning. For example, if you want to map a hypergraph onto a graph with 4096 nodes, you can first partition the hypergraph into 64 blocks, and then map each block of the partition onto a subgraph of the target graph with 64 nodes. We plan to integrate this technique into Mt-KaHyPar in the future.

Custom Objective Functions
-----------

We have implemented a common interface for all gain computation techniques that we use in our refinement algorithms. This enables us to extend Mt-KaHyPar with new objective functions without having to modify the internal implementation of the refinement algorithms. A step-by-step guide how you can implement your own objective function can be found [here][CustomObjectiveFunction].

Improving Compile Times
-----------

Mt-KaHyPar implements several graph and hypergraph data structures, and supports different objective functions. Each combination of (hyper)graph data structure and objective function is passed to our partitioning algorithms as template parameters. This increases the compile time of Mt-KaHyPar. We therefore provide cmake command line options to disable some of the features of Mt-KaHyPar for faster compilation. The following list summarizes the available parameters:
```cmake
-DKAHYPAR_ENABLE_GRAPH_PARTITIONING_FEATURES=On/Off # enables/disables graph partitioning features
-DKAHYPAR_ENABLE_HIGHEST_QUALITY_FEATURES=On/Off # enables/disables our highest-quality configuration
-DKAHYPAR_ENABLE_LARGE_K_PARTITIONING_FEATURES=On/Off # enables/distables large k partitioning features
-DKAHYPAR_ENABLE_SOED_METRIC=On/Off # enables/disables sum-of-external-degree metric
-DKAHYPAR_ENABLE_STEINER_TREE_METRIC=On/Off # enables/disables Steiner tree metric
```
If you turn off all features, only the `deterministic`, `default`, and `quality` configuration are available for optimizing the cut-net or connectivity metric. Using a disabled feature will throw an error. Note that you can only disable the features in our binary, not in the C and Python interface.

Bug Reports
-----------

We encourage you to report any problems with Mt-KaHyPar via the [github issue tracking system](https://github.com/kittobi1992/mt-kahypar/issues) of the project.

Licensing
---------

Mt-KaHyPar is a free software provided under the MIT License.
For more information see the [LICENSE file][LF].
We distribute this framework freely to foster the use and development of hypergraph partitioning tools.
If you use Mt-KaHyPar in an academic setting please cite the appropriate papers.

    // Mt-KaHyPar-D
    @inproceedings{MT-KAHYPAR-D,
      title     = {Scalable Shared-Memory Hypergraph Partitioning},
      author    = {Gottesbüren, Lars and
                   Heuer, Tobias and
                   Sanders, Peter and
                   Schlag, Sebastian},
      booktitle = {23rd Workshop on Algorithm Engineering and Experiments (ALENEX 2021)},
      pages     = {16--30},
      year      = {2021},
      publisher = {SIAM},
      doi       = {10.1137/1.9781611976472.2},
    }

    // Mt-KaHyPar-Q
    @inproceedings{MT-KAHYPAR-Q,
      title     = {Shared-Memory $n$-level Hypergraph Partitioning},
      author    = {Lars Gottesb{\"{u}}ren and
                   Tobias Heuer and
                   Peter Sanders and
                   Sebastian Schlag},
      booktitle = {24th Workshop on Algorithm Engineering and Experiments (ALENEX 2022)},
      year      = {2022},
      publisher = {SIAM},
      month     = {01},
      doi       = {10.1137/1.9781611977042.11}
    }

    // Mt-KaHyPar-Q-F
    @inproceedings{MT-KaHyPar-Q-F,
      title       =	{Parallel Flow-Based Hypergraph Partitioning},
      author      =	{Lars Gottesb\"{u}ren and
                     Tobias Heuer and
                     Peter Sanders},
      booktitle   =	{20th International Symposium on Experimental Algorithms (SEA 2022)},
      pages       =	{5:1--5:21},
      year        =	{2022},
      volume      =	{233},
      publisher   =	{Schloss Dagstuhl -- Leibniz-Zentrum f{\"u}r Informatik},
      doi         =	{10.4230/LIPIcs.SEA.2022.5}
    }

    // Deterministic Partitioning
    @inproceedings{MT-KAHYPAR-SDET,
      author    = {Lars Gottesb{\"{u}}ren and
                   Michael Hamann},
      title     = {Deterministic Parallel Hypergraph Partitioning},
      booktitle = {European Conference on Parallel Processing (Euro-Par)},
      volume    = {13440},
      pages     = {301--316},
      publisher = {Springer},
      year      = {2022},
      doi       = {10.1007/978-3-031-12597-3\_19},
    }

    // Unconstrained Refinement (Under Review)
    @article{MT-KAHYPAR-UNCONSTRAINED,
      title       = {Parallel Unconstrained Local Search for Partitioning Irregular Graphs},
      author      = {Nikolai Maas and
                     Lars Gottesb{\"{u}}ren and
                     Daniel Seemaier},
      institution = {Karlsruhe Institute of Technology},
      year        = {2023},
      url         = {https://arxiv.org/abs/2308.15494}
    }

    // Dissertation of Lars Gottesbüren
    @phdthesis{MT-KAHYPAR-DIS-GOTTESBUEREN,
      author         = {Lars Gottesb\"{u}ren},
      year           = {2023},
      title          = {Parallel and Flow-Based High-Quality Hypergraph Partitioning},
      doi            = {10.5445/IR/1000157894},
      pagetotal      = {256},
      school         = {Karlsruhe Institute of Technology}
    }

    // Dissertation of Tobias Heuer
    @phdthesis{MT-KAHYPAR-DIS-HEUER,
        author       = {Heuer, Tobias},
        year         = {2022},
        title        = {Scalable High-Quality Graph and Hypergraph Partitioning},
        doi          = {10.5445/IR/1000152872},
        pagetotal    = {242},
        school       = {Karlsruhe Institute of Technology}
    }

    // Mt-KaHyPar Journal Paper (Under Review)
    @article{MT-KAHYPAR-JOURNAL,
      title       = {Scalable High-Quality Hypergraph Partitioning},
      author      = {Lars Gottesb{\"u}ren and
                     Tobias Heuer and
                     Nikolai Maas and
                     Peter Sanders and
                     Sebastian Schlag},
      institution = {Karlsruhe Institute of Technology},
      year        = {2023},
      url         = {https://arxiv.org/abs/2303.17679}
    }

Contributing
------------
If you are interested in contributing to the Mt-KaHyPar framework
feel free to contact us or create an issue on the
[issue tracking system](https://github.com/kahypar/mt-kahypar/issues).

[cmake]: http://www.cmake.org/ "CMake tool"
[Boost.Program_options]: http://www.boost.org/doc/libs/1_58_0/doc/html/program_options.html
[tbb]: https://software.intel.com/content/www/us/en/develop/tools/threading-building-blocks.html
[hwloc]: https://www.open-mpi.org/projects/hwloc/
[LF]: https://github.com/kahypar/mt-kahypar/blob/master/LICENSE "License"
[SetA]: http://algo2.iti.kit.edu/heuer/alenex21/instances.html?benchmark=set_a
[SetB]: http://algo2.iti.kit.edu/heuer/alenex21/instances.html?benchmark=set_b
[ExperimentalResults]: https://algo2.iti.kit.edu/heuer/mt_kahypar/
[MSYS2]: https://www.msys2.org/
[CustomObjectiveFunction]: https://github.com/kahypar/mt-kahypar/tree/master/mt-kahypar/partition/refinement/gains
[SteinerTrees]: https://en.wikipedia.org/wiki/Steiner_tree_problem#Steiner_tree_in_graphs_and_variants
