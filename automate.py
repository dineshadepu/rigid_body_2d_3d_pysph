#!/usr/bin/env python
import os
import matplotlib.pyplot as plt

from itertools import product
import json
from automan.api import PySPHProblem as Problem
from automan.api import Automator, Simulation, filter_by_name
from automan.jobs import free_cores

import numpy as np
import matplotlib
from pysph.solver.utils import load, get_files

matplotlib.use('pdf')

n_core = 24
n_thread = 24 * 2
backend = ' --openmp '


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def scheme_opts(params):
    if isinstance(params, tuple):
        return params[0]
    return params


def get_files_at_given_times(files, times):
    from pysph.solver.utils import load
    result = []
    count = 0
    for f in files:
        data = load(f)
        t = data['solver_data']['t']
        if count >= len(times):
            break
        if abs(t - times[count]) < t * 1e-8:
            result.append(f)
            count += 1
        elif t > times[count]:
            result.append(f)
            count += 1
    return result


def get_files_at_given_times_from_log(files, times, logfile):
    import re
    result = []
    time_pattern = r"output at time\ (\d+(?:\.\d+)?)"
    file_count, time_count = 0, 0
    with open(logfile, 'r') as f:
        for line in f:
            if time_count >= len(times):
                break
            t = re.findall(time_pattern, line)
            if t:
                if float(t[0]) in times:
                    result.append(files[file_count])
                    time_count += 1
                elif float(t[0]) > times[time_count]:
                    result.append(files[file_count])
                    time_count += 1
                file_count += 1
    return result


class StackOfCylinders(Problem):
    def get_name(self):
        return 'stack_of_cylinders'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/stack_of_cylinders.py' + backend

        # Base case info
        self.case_info = {
            'bui': (dict(
                pfreq=100,
                dem="bui",
                ), 'Bui'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class WedgeEntry2D(Problem):
    def get_name(self):
        return '2d_wedge_entry'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/2d_wedge_entry.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                nrbc=None,
                pfreq=200,
                tf=5.,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class WaterEntryOfCylinder2D(Problem):
    def get_name(self):
        return 'water_entry_of_cylinder_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/water_entry_of_cylinder_2d.py' + backend

        # Base case info
        self.case_info = {
            'WCSPH': (dict(
                nrbc=None,
                pfreq=200,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class RigidBodyRotatingAndSinking(Problem):
    def get_name(self):
        return 'rigid_body_rotating_and_sinking'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/rigid_body_rotating_and_sinking_in_tank_2d.py' + backend

        # Base case info
        self.case_info = {
            'WCSPH': (dict(
                nrbc=None,
                pfreq=200,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class DamBreakFlowAgainstSingleCube(Problem):
    def get_name(self):
        return 'dam_break_flow_against_a_single_cube'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dam_break_flow_against_a_single_cube.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pfreq=200,
                dem="canelas",
                # tf=2.0,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class DamBreakFlowAgainstThreeCubes(Problem):
    def get_name(self):
        return 'dam_break_flow_against_three_cubes'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dam_break_flow_against_three_cubes.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pfreq=200,
                dem="canelas",
                # tf=2.0,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class DamBreakFlowAgainstSixCubes(Problem):
    def get_name(self):
        return 'dam_break_flow_against_six_cubes'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dam_break_flow_against_six_cubes.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pfreq=200,
                dem="canelas",
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class BlockFloatingInSteadyTank(Problem):
    def get_name(self):
        return 'block_floating_in_steady_tank_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/block_floating_in_steady_tank_3d.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pfreq=200,
                dem="canelas",
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class BlockSinkingInTank(Problem):
    def get_name(self):
        return 'block_sinking_in_tank_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/block_sinking_in_tank_3d.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pfreq=200,
                dem="canelas",
                # tf=2.0,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class CylidersInWaterCollapsedUnderGravity(Problem):
    def get_name(self):
        return 'cyliders_in_water_collapsed_under_gravity'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/cyliders_in_water_collapsed_under_gravity.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pfreq=200,
                dem="canelas",
                # tf=2.0,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class DamBreakWithBodyTransport(Problem):
    def get_name(self):
        return 'dam_break_with_body_tranport_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dam_break_with_body_tranport_3d.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pfreq=200,
                dem="canelas",
                # tf=2.0,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class DamBreakWithMultipleBodiesTransport(Problem):
    def get_name(self):
        return 'dam_break_with_multiple_bodies_tranport_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dam_break_with_multiple_bodies_tranport_3d.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pfreq=200,
                dem="canelas",
                # tf=2.0,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Mohseni2021FreeSlidingOnASlope2D(Problem):
    def get_name(self):
        return 'mohseni_2021_free_sliding_on_a_slope_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/mohseni_2021_free_sliding_on_a_slope_2d.py' + backend

        # Base case info
        self.case_info = {
            'fric_coeff_0_2': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.2,
                tf=3.,
                ), r'$\mu=$0.2'),

            'fric_coeff_0_4': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.4,
                tf=3.,
                ), r'$\mu=$0.4'),

            'fric_coeff_tan_30': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=np.tan(np.pi/6),
                tf=3.,
                ), r'$\mu=$tan(30)'),

            'fric_coeff_0.6': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.6,
                tf=3.,
                ), r'$\mu=$0.6'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_velocity()
        self.move_figures()

    def plot_velocity(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        for name in self.case_info:
            t_analytical = data[name]['t_analytical']
            v_analytical = data[name]['v_analytical']

            t = data[name]['t']
            velocity_rbd = data[name]['velocity_rbd']

            plt.plot(t_analytical, v_analytical, label=self.case_info[name][1] + ' analytical')
            plt.scatter(t, velocity_rbd, s=1, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('velocity_vs_time.pdf'))
        plt.clf()
        plt.close()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class Mohseni2021FreeSlidingOnASlope3D(Problem):
    def get_name(self):
        return 'mohseni_2021_free_sliding_on_a_slope_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/mohseni_2021_free_sliding_on_a_slope_3d.py' + backend

        # Base case info
        self.case_info = {
            'fric_coeff_0_2': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.2,
                tf=3.,
                ), r'$\mu=$0.2'),

            'fric_coeff_0_4': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.4,
                tf=3.,
                ), r'$\mu=$0.4'),

            'fric_coeff_tan_30': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=np.tan(np.pi/6),
                tf=3.,
                ), r'$\mu=$tan(30)'),

            'fric_coeff_0.6': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.6,
                tf=3.,
                ), r'$\mu=$0.6'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_velocity()
        self.move_figures()

    def plot_velocity(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        for name in self.case_info:
            t_analytical = data[name]['t_analytical']
            v_analytical = data[name]['v_analytical']

            t = data[name]['t']
            velocity_rbd = data[name]['velocity_rbd']

            plt.plot(t_analytical, v_analytical, label=self.case_info[name][1] + ' analytical')
            plt.scatter(t, velocity_rbd, s=1, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('velocity_vs_time.pdf'))
        plt.clf()
        plt.close()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class Dinesh2022MultipleCubesColliding3D(Problem):
    def get_name(self):
        return 'dinesh_2022_multiple_cubes_colliding_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dinesh_2022_multiple_cubes_colliding.py' + backend

        # Base case info
        self.case_info = {
            'two_cubes_just_touching_no_gravity': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.2,
                tf=3.,
                ), r'$\mu=$0.2'),

            'two_cubes_overlapping_blow_up': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.2,
                tf=3.,
                ), r'$\mu=$0.2'),

            'multiple_cubes_colliding': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.2,
                tf=3.,
                ), r'$\mu=$0.2'),

            'many_cubes_resting_on_a_big_rigid_cube': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.2,
                tf=3.,
                ), r'$\mu=$0.2'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Dinesh2022HydrostaticTank2D(Problem):
    def get_name(self):
        return 'dinesh_2022_hydrostatic_tank_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dinesh_2022_hydrostatic_tank_2d.py' + backend

        # Base case info
        self.case_info = {
            # 'dx_0_005': (dict(
            #     scheme='rfc',
            #     pfreq=300,
            #     dx=5*1e-3,
            #     kr=1e5,
            #     fric_coeff=0.0,
            #     tf=0.5,
            #     ), 'dx=0.005 m'),

            # 'dx_0_003': (dict(
            #     scheme='rfc',
            #     pfreq=300,
            #     dx=3*1e-3,
            #     kr=1e5,
            #     fric_coeff=0.0,
            #     tf=0.5,
            #     ), 'dx=0.003 m'),

            'dx_0_002': (dict(
                scheme='rfc',
                pfreq=300,
                dx=2*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=0.5,
                ), 'dx=0.002 m'),

            'dx_0_001': (dict(
                scheme='rfc',
                pfreq=300,
                dx=1*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=0.5,
                ), 'dx=0.001 m'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Qiu2017FallingSolidInWater2D(Problem):
    def get_name(self):
        return 'qiu_2017_falling_solid_in_water_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/qiu_2017_falling_solid_in_water_2d.py' + backend

        # Base case info
        self.case_info = {
            # 'dx_0_005': (dict(
            #     scheme='rfc',
            #     pfreq=300,
            #     dx=5*1e-3,
            #     kr=1e5,
            #     fric_coeff=0.0,
            #     tf=0.5,
            #     ), 'dx=0.005 m'),

            # 'dx_0_003': (dict(
            #     scheme='rfc',
            #     pfreq=300,
            #     dx=3*1e-3,
            #     kr=1e5,
            #     fric_coeff=0.0,
            #     tf=0.5,
            #     ), 'dx=0.003 m'),

            'dx_0_002': (dict(
                scheme='rfc',
                edac=None,
                pfreq=300,
                dx=2*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=0.5,
                ), 'edac dx=0.002 m'),

            'dx_0_001': (dict(
                scheme='rfc',
                edac=None,
                pfreq=300,
                dx=1*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=0.5,
                ), 'edac dx=0.001 m'),

            'dx_0_002_no_edac': (dict(
                scheme='rfc',
                no_edac=None,
                pfreq=300,
                dx=2*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=0.5,
                ), 'dx=0.002 m'),

            'dx_0_001_no_edac': (dict(
                scheme='rfc',
                no_edac=None,
                pfreq=300,
                dx=1*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=0.5,
                ), 'dx=0.001 m'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_velocity()

    def plot_velocity(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        for name in self.case_info:
            t_experimental = data[name]['t_experimental']
            y_cm_experimental = data[name]['y_cm_experimental']

            t = data[name]['t']
            y_cm_simulated = data[name]['y_cm_simulated']

            plt.scatter(t, y_cm_simulated, s=1, label=self.case_info[name][1])

        # experimental plot should be only once plotted
        plt.plot(t_experimental, y_cm_experimental, label=self.case_info[name][1] + ' Experimental')

        plt.xlabel('time')
        plt.ylabel('')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('y_cm_vs_time.pdf'))
        plt.clf()
        plt.close()


class Qiu2017FallingSolidInWater3D(Problem):
    def get_name(self):
        return 'qiu_2017_falling_solid_in_water_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/qiu_2017_falling_solid_in_water_3d.py' + backend

        # Base case info
        self.case_info = {
            'case_1': (dict(
                scheme='rfc',
                pfreq=300,
                dx=5*1e-3,
                kr=1e5,
                fric_coeff=0.2,
                tf=0.5,
                ), r'$\mu=$0.2'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Qiu2017FloatingSolidInWater2D(Problem):
    def get_name(self):
        return 'qiu_2017_floating_solid_in_water_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/qiu_2017_floating_solid_in_water_2d.py' + backend

        # Base case info
        self.case_info = {
            'dx_0_002_edac': (dict(
                scheme='rfc',
                pfreq=300,
                edac=None,
                dx=2*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=1.5,
                ), 'edac dx=0.002 m'),

            'dx_0_001_edac': (dict(
                scheme='rfc',
                pfreq=300,
                edac=None,
                dx=1*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=1.5,
                ), 'edac dx=0.001 m'),

            'dx_0_002_no_edac': (dict(
                scheme='rfc',
                pfreq=300,
                no_edac=None,
                dx=2*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=1.5,
                ), 'dx=0.002 m'),

            'dx_0_001_no_edac': (dict(
                scheme='rfc',
                pfreq=300,
                no_edac=None,
                dx=1*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=1.5,
                ), 'dx=0.001 m'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_velocity()

    def plot_velocity(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        for name in self.case_info:
            t_experimental = data[name]['t_experimental']
            y_cm_experimental = data[name]['y_cm_experimental']

            t = data[name]['t']
            y_cm_simulated = data[name]['y_cm_simulated']

            plt.scatter(t, y_cm_simulated, s=1, label=self.case_info[name][1])

        # experimental plot should be only once plotted
        plt.plot(t_experimental, y_cm_experimental, label=self.case_info[name][1] + ' Experimental')

        plt.xlabel('time')
        plt.ylabel('')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('y_cm_vs_time.pdf'))
        plt.clf()
        plt.close()


class Qiu2017FloatingSolidInWater3D(Problem):
    def get_name(self):
        return 'qiu_2017_floating_solid_in_water_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/qiu_2017_floating_solid_in_water_3d.py' + backend

        # Base case info
        self.case_info = {
            'case_1': (dict(
                scheme='rfc',
                pfreq=300,
                dx=5*1e-3,
                kr=1e5,
                fric_coeff=0.2,
                tf=0.5,
                ), r'$\mu=$0.2'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Dinesh2022SteadyCubesOnAWall2D(Problem):
    def get_name(self):
        return 'dinesh_2022_steady_cubes_on_a_wall_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dinesh_2022_steady_cubes_on_a_wall_2d.py' + backend

        # Base case info
        self.case_info = {
            'two_cubes': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                # This has to be fixed. Friction coefficient between different bodies
                fric_coeff=0.2,
                tf=1.8,
                ), 'Two cubes'),

            'three_cubes': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                # This has to be fixed. Friction coefficient between different bodies
                fric_coeff=0.2,
                tf=2.,
                ), 'Three cubes'),

            'six_cubes': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                # This has to be fixed. Friction coefficient between different bodies
                fric_coeff=0.2,
                tf=2.,
                ), 'Six cubes'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Dinesh2022SteadyCubesOnAWall3D(Problem):
    def get_name(self):
        return 'dinesh_2022_steady_cubes_on_a_wall_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dinesh_2022_steady_cubes_on_a_wall_3d.py' + backend

        # Base case info
        self.case_info = {
            'two_cubes': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                # This has to be fixed. Friction coefficient between different bodies
                fric_coeff=0.2,
                tf=1.8,
                ), 'Two cubes'),

            # 'three_cubes': (dict(
            #     scheme='rfc',
            #     pfreq=300,
            #     kr=1e5,
            #     # This has to be fixed. Friction coefficient between different bodies
            #     fric_coeff=0.2,
            #     tf=2.,
            #     ), 'Three cubes'),

            # 'six_cubes': (dict(
            #     scheme='rfc',
            #     pfreq=300,
            #     kr=1e5,
            #     # This has to be fixed. Friction coefficient between different bodies
            #     fric_coeff=0.2,
            #     tf=2.,
            #     ), 'Six cubes'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Dinesh2022DamBreak3d(Problem):
    def get_name(self):
        return 'dinesh_2022_dam_break_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dinesh_2022_3d_dam_break.py' + backend

        # Base case info
        self.case_info = {
            'dx_0_002': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=2.,
                ), 'dx=0.002 m'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Amaro2019DamBreakingFlowHittingOneCube3d(Problem):
    def get_name(self):
        return 'amaro_2019_dam_breaking_flow_hitting_one_cube_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/amaro_2019_dam_breaking_flow_hitting_one_cube_3d.py' + backend

        # Base case info
        self.case_info = {
            'case_1': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                # This has to be fixed. Friction coefficient between different bodies
                fric_coeff=0.2,
                tf=3.,
                ), 'case 1'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Amaro2019DamBreakingFlowHittingThreeStackedCubes3d(Problem):
    def get_name(self):
        return 'amaro_2019_dam_breaking_flow_hitting_three_stacked_cubes_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/amaro_2019_dam_breaking_flow_hitting_three_stacked_cubes_3d.py' + backend

        # Base case info
        self.case_info = {
            'case_1': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                # This has to be fixed. Friction coefficient between different bodies
                fric_coeff=0.2,
                tf=3.5,
                ), 'case 1'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Amaro2019DamBreakingFlowHittingSixStackedCubes3d(Problem):
    def get_name(self):
        return 'amaro_2019_dam_breaking_flow_hitting_six_stacked_cubes_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/amaro_2019_dam_breaking_flow_hitting_six_stacked_cubes_3d.py' + backend

        # Base case info
        self.case_info = {
            'case_1': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                # This has to be fixed. Friction coefficient between different bodies
                fric_coeff=0.2,
                tf=2.,
                ), 'case 1'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('pdf')

    PROBLEMS = [
        # ========================
        # Only rigid body problems
        # ========================
        # Current paper problem
        # Dinesh2022BouncingCube3D, # tests the coefficient of restitution

        # # Current paper problem
        # Mohseni2021FreeSlidingOnASlope2D,
        # Mohseni2021FreeSlidingOnASlope3D,

        # # Current paper problem
        # Mohseni2021ControlledSlidingOnAFlatSurface2D,
        # Mohseni2021ControlledSlidingOnAFlatSurface3D,

        # # Current paper problem
        # StackOfCylinders2D, # Experimental validation

        # fixme: add figure and validate, you can try 2d case for
        # simplicity
        # Current paper problem
        Qiu2017FallingSolidInWater2D,
        Qiu2017FallingSolidInWater3D,

        # # Current paper problem
        Qiu2017FloatingSolidInWater2D,
        # Qiu2017FloatingSolidInWater3D,

        # Current paper problem
        Amaro2019DamBreakingFlowHittingOneCube3d,

        # Current paper problem
        Amaro2019DamBreakingFlowHittingSixStackedCubes3d,

        # Current paper problem
        Amaro2019DamBreakingFlowHittingThreeStackedCubes3d,

        # These are test problems for body transport under dam break 3d
        Dinesh2022DamBreak3d,
        # Dinesh2022HydrostaticTank2D,
        # Dinesh2022MultipleCubesColliding3D,
        # Dinesh2022SteadyCubesOnAWall2D,
        Dinesh2022SteadyCubesOnAWall3D
    ]

    automator = Automator(simulation_dir='outputs',
                          output_dir=os.path.join('manuscript', 'figures'),
                          all_problems=PROBLEMS)

    automator.run()
