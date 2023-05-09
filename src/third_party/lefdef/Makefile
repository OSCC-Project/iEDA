#-------------------------------------------------------------------------
#
#  Copyright (c) 2021 Rajit Manohar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.
#
#-------------------------------------------------------------------------

all:
	(cd lef; make)
	(cd def; make)

INST=$(ACT_HOME)/scripts/install

install:
	@if [ ! -d $(ACT_HOME)/include/lef ]; then mkdir $(ACT_HOME)/include/lef; fi
	@if [ ! -d $(ACT_HOME)/include/def ]; then mkdir $(ACT_HOME)/include/def; fi
	@(cd lef/include; for i in *; do $(INST) $$i $(ACT_HOME)/include/lef/$$i; done)
	@(cd def/include; for i in *; do $(INST) $$i $(ACT_HOME)/include/def/$$i; done)
	@(cd lef/lib; for i in *; do $(INST) $$i $(ACT_HOME)/lib/$$i; done)
	@(cd def/lib; for i in *; do $(INST) $$i $(ACT_HOME)/lib/$$i; done)

clean:
	(cd lef; make clean)
	(cd def; make clean)
