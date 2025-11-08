# ICS55 GCD

This directory contains the scripts and configuration files to run iEDA on the GCD design.

To run this flow, please refer to the `run_iEDA.sh` script located in this directory.

ICS55 PDK is required to run this design. Please ensure you have downloaded ICS55 PDK and set the `PDK_DIR` environment variable to point to the iPD foundry scripts directory. ICS55 PDK is available at: [icsprout55-pdk](https://github.com/openecos-projects/icsprout55-pdk)

Run flow with:

```bash
PDK_DIR=/home/test/iPD-git/scripts/foundry/ics55 ./run_iEDA.sh
```