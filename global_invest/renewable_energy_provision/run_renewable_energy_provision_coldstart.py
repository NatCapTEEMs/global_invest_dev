"""Renewable energy from an empty project directory, to settle condition 10.

`run_mode='full'` mints a fresh timestamped project directory, so every table is rebuilt from
the staged inputs alone. The target: $173.49bn total, wind $112,524,374,333, solar
$54,410,978,171, geothermal $6,558,292,384.
"""
import hazelbean as hb

from global_invest.renewable_energy_provision.run_renewable_energy_provision import run_project


if __name__ == '__main__':

    p = hb.ProjectFlow(project_name='gep_renewable_energy_provision_coldstart', run_mode='full')
    run_project(p)

    print('COLD START PROJECT DIR: ' + p.project_dir)
