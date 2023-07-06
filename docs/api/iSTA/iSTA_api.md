# iSTA API
## API list
| API Command | Type | Description |
| :--- | :--- | :--- |
| [set_num_threads](#set_num_threads) | xxx | set the numbers of threads |
| [set_design_work_space](#set_design_work_space) | xxx | set the directory to output the timing reports |
| [readLiberty](#readLiberty) | xxx | read the liberty files |
| [readDesign](#readDesign) | xxx | read the design verilog file |
| [readSpef](#readSpef) | xxx | read the spef file |
| [readSdc](#readSdc) | xxx | read the sdc file |
| [readAocv](#readAocv) | xxx | read the aocv files |
| [makeOrFindRCTreeNode](#makeOrFindRCTreeNode) | xxx | make RC tree internal node |
| [makeOrFindRCTreeNode](#makeOrFindRCTreeNode) | xxx | make RC tree pin node |
| [incrCap](#incrCap) | xxx | set the node’s cap |
| [makeResistor](#makeResistor) | xxx | make resistor edge of RC tree |
| [updateRCTreeInfo](#updateRCTreeInfo) | xxx | update the RC info after making the RC tree |
| [buildRCTree](#buildRCTree) | xxx | build the RC tree according to the spef file |
| [initRcTree](#initRcTree) | xxx | init one RC tree |
| [resetRcTree](#resetRcTree) | xxx | reset the RC tree to nullptr |
| [buildGraph](#buildGraph) | xxx | build the STA graph data |
| [isBuildGraph](#isBuildGraph) | xxx | judge whether the STA steps has build the STA graph |
| [resetGraph](#resetGraph) | xxx | reset the STA graph |
| [resetGraphData](#resetGraphData) | xxx | reset the STA graph data |
| [insertBuffer](#insertBuffer) | xxx | insert the buffer need to change the netlist |
| [removeBuffer](#removeBuffer) | xxx | remove buffer need to change the netlist |
| [repowerInstance](#repowerInstance) | xxx | change the size or the level of an existing instance |
| [moveInstance](#moveInstance) | xxx | move the instance to a new location |
| [writeVerilog](#writeVerilog) | xxx | write the verilog file according to the netlist data structure |
| [incrUpdateTiming](#incrUpdateTiming) | xxx | incremental propagation to update the timing data |
| [updateTiming](#updateTiming) | xxx | update the timing data |
| [setSignificantDigits](#setSignificantDigits) | xxx | set the significant digits of the timing report |
| [reportTiming](#reportTiming) | xxx | generate the timing reports |
| [reportSlew](#reportSlew) | xxx | report the slew of the pin |
| [reportAT](#reportAT) | xxx | report the arrival time at a pin |
| [reportRT](#reportRT) | xxx | report the required arrival time at a pin |
| [reportSlack](#reportSlack) | xxx | report the slack at a pin |
| [reportWNS](#reportWNS) | xxx | report the worst negative slack of the clock group path |
| [reportTNS](#reportTNS) | xxx | report the total negative slack of the clock group path |
| [reportClockSkew](#reportClockSkew) | xxx | report the skew between two clocks |
| [reportInstDelay](#reportInstDelay) | xxx | report the instance delay |
| [reportInstWorstArcDelay](#reportInstWorstArcDelay) | xxx | report the worst arc delay for the specified instance |
| [reportNetDelay](#reportNetDelay) | xxx | report the net delay |
| [checkCapacitance](#checkCapacitance) | xxx | check the real pin capacitance and the limit pin capacitance in liberty,calculate the capacitance slack |
| [checkFanout](#checkFanout) | xxx | check the real fanout nums and the limit fanout nums in liberty,calculate the fanout slack |
| [checkSlew](#checkSlew) | xxx | check the real slew and the limit slew in liberty,calculate the slew slack |
| [getCellType](#getCellType) | xxx | get the cell type of the cell |
| [getCellArea](#getCellArea) | xxx | get the area of the cell |
| [isSequentialCell](#isSequentialCell) | xxx | judege whether the instance is sequential cell |
| [isClock](#isClock) | xxx | judege whether the pin is clock pin |
| [isLoad](#isLoad) | xxx | judege whether whether the pin is load |

---

### set_num_threads <a id="set_num_threads"></a>
Set the numbers of threads. <br>
```C++
TimingEngine &set_num_threads(unsigned num_thread)
```

**Parameters** <br>
- num_thread: the numbers of the threads to set

**Return Value** <br>
-value : *this


<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### set_design_work_space <a id="set_design_work_space"></a>
Set the directory to output the timing reports. <br>
```C++
void set_design_work_space(const char *design_work_space)
```

**Parameters** <br>
- design_work_space: the file directory of the timing reports to write

**Return Value** <br>
-value : void


<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### readLiberty <a id="readLiberty"></a>
Read the liberty files. <br>
```C++
TimingEngine &readLiberty(std::vector<std::string> &lib_files)
```

**Parameters** <br>
- lib_files: the file paths of the liberty files to read

**Return Value** <br>
-value : *this


<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### readDesign <a id="readDesign"></a>
Read the design verilog file. <br>
```C++
TimingEngine &readDesign(const char *verilog_file)
```
**Parameters** <br>
- verilog_file : the file path of the verilog file to read

**Return Value** <br>
-value :*this


<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### readSpef <a id="readSpef"></a>
Read the spef file. <br>
```C++
TimingEngine &readSpef(const char *spef_file)
```
**Parameters** <br>
- spef_file : the file path of the spef file to read

**Return Value** <br>
-value :*this


<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### readSdc <a id="readSdc"></a>
Read the sdc file. <br>
```C++
TimingEngine &readSdc(const char *sdc_file)
```
**Parameters** <br>
- sdc_file : the file path of the sdc file to read

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### readAocv <a id="readAocv"></a>
Read the sdc file. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### makeOrFindRCTreeNode <a id="makeOrFindRCTreeNode"></a>
Make RC tree internal node. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### makeOrFindRCTreeNode <a id="makeOrFindRCTreeNode"></a>
Make RC tree internal node. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### incrCap <a id="incrCap"></a>
Set the node’s cap. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### makeResistor <a id="makeResistor"></a>
Make resistor edge of RC tree. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### updateRCTreeInfo <a id="updateRCTreeInfo"></a>
Update the RC info after making the RC tree. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### buildRCTree <a id="buildRCTree"></a>
Build the RC tree according to the spef file. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### initRcTree <a id="initRcTree"></a>
Init one RC tree. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### resetRcTree <a id="resetRcTree"></a>
Reset the RC tree to nullptr. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### buildGraph <a id="buildGraph"></a>
Build the STA graph data. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### isBuildGraph <a id="isBuildGraph"></a>
Judge whether the STA steps has build the STA graph. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### resetGraph <a id="resetGraph"></a>
Reset the STA graph. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### resetGraphData <a id="resetGraphData"></a>
Reset the STA graph data. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### insertBuffer <a id="insertBuffer"></a>
Insert the buffer need to change the netlist. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### removeBuffer <a id="removeBuffer"></a>
Remove buffer need to change the netlist. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### repowerInstance <a id="repowerInstance"></a>
Change the size or  the level of an existing instance. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### moveInstance <a id="moveInstance"></a>
Move the instance to a new location. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### writeVerilog <a id="writeVerilog"></a>
Write the verilog file according to the netlist data structure. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### incrUpdateTiming <a id="incrUpdateTiming"></a>
Incremental propagation to update the timing data. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### updateTiming <a id="updateTiming"></a>
Update the timing data. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### setSignificantDigits <a id="setSignificantDigits"></a>
Set the significant digits of the timing report. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### reportTiming <a id="reportTiming"></a>
Generate the timing report. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### reportSlew <a id="reportSlew"></a>
Report the slew of the pin. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### reportAT <a id="reportAT"></a>
Report the arrival time at a pin. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### reportRT <a id="reportRT"></a>
Report the required arrival time at a pin. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### reportSlack <a id="reportSlack"></a>
Report the slack at a pin. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### reportWNS <a id="reportWNS"></a>
Report the worst negative slack of the clock group path. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### reportTNS <a id="reportTNS"></a>
Report the total negative slack of the clock group path. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### reportClockSkew <a id="reportClockSkew"></a>
Report the skew between two clocks. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### reportInstDelay <a id="reportInstDelay"></a>
Report the instance delay. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### reportInstWorstArcDelay <a id="reportInstWorstArcDelay"></a>
Report the worst arc delay for the specified instance. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### reportNetDelay <a id="reportNetDelay"></a>
Report the net delay. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### checkCapacitance <a id="checkCapacitance"></a>
Check the real pin capacitance and the limit pin capacitance in liberty,calculate the capacitance slack. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### checkFanout <a id="checkFanout"></a>
Check the real fanout nums and the limit fanout nums in liberty,calculate the fanout slack. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### checkSlew <a id="checkSlew"></a>
Check the real slew and the limit slew in liberty,calculate the slew slack. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### getCellType <a id="getCellType"></a>
Get the cell type of the cell. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### getCellArea <a id="getCellArea"></a>
Get the area of the cell. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### isSequentialCell <a id="isSequentialCell"></a>
Judege whether the instance is sequential cell. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### isClock <a id="isClock"></a>
Judege whether the pin is clock pin. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---

### isLoad <a id="isLoad"></a>
Judege whether whether the pin is load. <br>
```C++
readDesign
```
**Parameters** <br>
- param_1 : xxx
- param_2 : xxx

**Return Value** <br>
-value : xxx

**Notes** <br>
xxxxxxxxxxxxxxxxxxxx

<div align="right"><b><a href="#iSTA API">↥ back to top</a></b></div>

---