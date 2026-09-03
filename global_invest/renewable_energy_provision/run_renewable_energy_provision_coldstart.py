"""Renewable energy from an empty project directory, to settle condition 10.

`run_mode='full'` mints a fresh timestamped project directory, so every table is rebuilt from
the staged inputs alone. The target: $170.83bn total, wind $110,313,990,626, solar
$53,954,218,147, geothermal $6.56bn.
"""
import hazelbean as hb

from global_invest.renewable_energy_provision.run_renewable_energy_provision import run_project


if __name__ == '__main__':

    p = hb.ProjectFlow(project_name='gep_renewable_energy_provision_coldstart', run_mode='full')
    run_project(p)

    print('COLD START PROJECT DIR: ' + p.project_dir)
