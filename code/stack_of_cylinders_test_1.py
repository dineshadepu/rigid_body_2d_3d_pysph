"""
Simulation of solid-fluid mixture flow using moving particle methods
Shuai Zhang
"""
from __future__ import print_function
import numpy as np

from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from pysph.base.utils import (get_particle_array)

from rigid_fluid_coupling import RigidFluidCouplingScheme

from pysph.tools.geometry import get_2d_block
# from geometry import hydrostatic_tank_2d


def create_circle(diameter=1, spacing=0.01, center=None):
    radius = diameter/2.
    xtmp, ytmp = get_2d_block(spacing, diameter+spacing, diameter+spacing)
    x = []
    y = []
    for i in range(len(xtmp)):
        dist = xtmp[i]**2. + ytmp[i]**2.
        if dist < radius**2:
            x.append(xtmp[i])
            y.append(ytmp[i])

    x = np.array(x)
    y = np.array(y)
    x, y = (t.ravel() for t in (x, y))
    if center is None:
        return x, y
    else:
        return x + center[0], y + center[1]


def create_circle_1(diameter=1, spacing=0.05, center=None):
    dx = spacing
    x = [0.0]
    y = [0.0]
    r = spacing
    nt = 0
    radius = diameter / 2.

    tmp_dist = radius + spacing/2.
    i = 0
    while tmp_dist > spacing/2.:
        tmp_dist = radius - spacing/2. - i * spacing
        perimeter = 2. * np.pi * tmp_dist
        no_of_points = int(perimeter / spacing) + 1
        theta = np.linspace(0., 2. * np.pi, no_of_points)
        for t in theta[:-1]:
            x.append(tmp_dist * np.cos(t))
            y.append(tmp_dist * np.sin(t))
        i = i + 1

    x = np.array(x)
    y = np.array(y)
    x, y = (t.ravel() for t in (x, y))
    if center is None:
        return x, y
    else:
        return x + center[0], y + center[1]


class ZhangStackOfCylinders(Application):
    def initialize(self):
        self.dim = 2

        self.cylinder_radius = 1. / 2. * 1e-2
        self.cylinder_diameter = 1. * 1e-2
        self.cylinder_spacing = 0.5 * 1e-3
        self.cylinder_rho = 2000.

        self.dam_length = 26 * 1e-2
        self.dam_height = 26 * 1e-2
        self.dam_spacing = self.cylinder_spacing
        self.dam_layers = 2
        self.dam_rho = 2000.

        self.wall_height = 20 * 1e-2
        self.wall_spacing = 1e-3
        self.wall_layers = 2
        self.wall_time = 0.3
        self.wall_rho = 2000.

        # simulation properties
        self.hdx = 1.2
        self.alpha = 0.1
        self.gy = -9.81
        self.h = self.hdx * self.cylinder_spacing

        # solver data
        self.tf = 0.1
        self.dt = 1e-4

        # Rigid body collision related data
        self.limit = 6
        self.seval = None

    def create_particles(self):
        # get bodyid for each cylinder
        xc, yc = create_circle(
            self.cylinder_diameter, self.cylinder_spacing, [
                self.cylinder_radius,
                self.cylinder_radius + self.cylinder_spacing
            ])
        dem_id = 0
        m = self.cylinder_rho * self.cylinder_spacing**2
        h = self.hdx * self.cylinder_radius
        rad_s = self.cylinder_spacing / 2.
        cylinders = get_particle_array(name='cylinders', x=xc, y=yc,
                                       h=h, m=m, rad_s=rad_s)
        cylinders.add_property('dem_id', type='int', data=dem_id)
        cylinders.add_property('body_id', type='int', data=0)
        cylinders.add_constant('max_tng_contacts_limit', 10)
        cylinders.x[:] += self.cylinder_spacing/2.
        cylinders.y[:] -= self.cylinder_spacing

        xc1, yc1 = create_circle_1(
            self.cylinder_diameter, self.cylinder_spacing, [
                self.cylinder_radius,
                self.cylinder_radius + self.cylinder_spacing
            ])
        dem_id = 0
        m = self.cylinder_rho * self.cylinder_spacing**2
        h = self.hdx * self.cylinder_radius
        rad_s = self.cylinder_spacing / 2.
        cylinders1 = get_particle_array(name='cylinders1', x=xc1, y=yc1,
                                        h=h, m=m, rad_s=rad_s)
        cylinders1.add_property('dem_id', type='int', data=dem_id)
        cylinders1.add_property('body_id', type='int', data=0)
        cylinders1.add_constant('max_tng_contacts_limit', 10)
        cylinders1.x[:] += self.cylinder_spacing/2. + 2. * self.cylinder_diameter
        cylinders1.y[:] -= self.cylinder_spacing

        # create dam with normals
        xd, yd = get_2d_block(self.cylinder_spacing,
                              10. * self.cylinder_diameter,
                              2. * self.cylinder_spacing)

        dam = get_particle_array(x=xd,
                                 y=yd,
                                 rad_s=self.dam_spacing / 2.,
                                 name="dam")
        dam.add_property('dem_id', type='int', data=1)

        dam.y[:] -= max(dam.y) + self.cylinder_spacing / 2.
        self.scheme.setup_properties([cylinders, cylinders1, dam])

        # please run this function to know how
        # geometry looks like
        # from matplotlib import pyplot as plt
        # plt.scatter(cylinders.x, cylinders.y)
        # plt.scatter(dam.x, dam.y)
        # plt.scatter(wall.x, wall.y)
        # plt.axes().set_aspect('equal', 'datalim')
        # print("done")
        # plt.show()
        return [cylinders, cylinders1, dam]

    def create_scheme(self):
        rfc = RigidFluidCouplingScheme(rigid_bodies=['cylinders', 'cylinders1'],
                                       fluids=[],
                                       boundaries=['dam'],
                                       dim=2,
                                       rho0=self.cylinder_rho,
                                       h=self.h,
                                       nu=0.,
                                       p0=0.,
                                       c0=0.,
                                       kn=1e5,
                                       en=0.1,
                                       gy=self.gy)
        s = SchemeChooser(default='rfc', rfc=rfc)
        return s

    def configure_scheme(self):
        tf = self.tf

        self.scheme.configure_solver(dt=self.dt, tf=tf, pfreq=100)

    def create_cylinders_stack(self):
        # create a row of six cylinders
        x_six = np.array([])
        y_six = np.array([])
        x_tmp1, y_tmp1 = create_circle(
            self.cylinder_diameter, self.cylinder_spacing, [
                self.cylinder_radius,
                self.cylinder_radius + self.cylinder_spacing
            ])
        for i in range(6):
            x_tmp = x_tmp1 + i * (self.cylinder_diameter -
                                  self.cylinder_spacing / 2.)
            x_six = np.concatenate((x_six, x_tmp))
            y_six = np.concatenate((y_six, y_tmp1))

        # create three layers of six cylinder rows
        y_six_three = np.array([])
        x_six_three = np.array([])
        for i in range(3):
            x_six_three = np.concatenate((x_six_three, x_six))
            y_six_1 = y_six + 1.6 * i * self.cylinder_diameter
            y_six_three = np.concatenate((y_six_three, y_six_1))

        # create a row of five cylinders
        x_five = np.array([])
        y_five = np.array([])
        x_tmp1, y_tmp1 = create_circle(
            self.cylinder_diameter, self.cylinder_spacing, [
                2. * self.cylinder_radius, self.cylinder_radius +
                self.cylinder_spacing + self.cylinder_spacing / 2.
            ])

        for i in range(5):
            x_tmp = x_tmp1 + i * (self.cylinder_diameter -
                                  self.cylinder_spacing / 2.)
            x_five = np.concatenate((x_five, x_tmp))
            y_five = np.concatenate((y_five, y_tmp1))

        y_five = y_five + 0.75 * self.cylinder_diameter
        x_five = x_five

        # create three layers of five cylinder rows
        y_five_three = np.array([])
        x_five_three = np.array([])
        for i in range(3):
            x_five_three = np.concatenate((x_five_three, x_five))
            y_five_1 = y_five + 1.6 * i * self.cylinder_diameter
            y_five_three = np.concatenate((y_five_three, y_five_1))

        x = np.concatenate((x_six_three, x_five_three))
        y = np.concatenate((y_six_three, y_five_three))

        # create body_id
        no_particles_one_cylinder = len(x_tmp)
        total_bodies = 3 * 5 + 3 * 6

        body_id = np.array([], dtype=int)
        for i in range(total_bodies):
            b_id = np.ones(no_particles_one_cylinder, dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        return x, y, body_id

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        T = self.wall_time
        if (T - dt / 2) < t < (T + dt / 2):
            for pa in self.particles:
                if pa.name == 'wall':
                    pa.x += 0.25

    def post_process(self, fname):
        """This function will run once per time step after the time step is
        executed. For some time (self.wall_time), we will keep the wall near
        the cylinders such that they settle down to equilibrium and replicate
        the experiment.
        By running the example it becomes much clear.
        """
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        files = self.output_files
        print(len(files))
        t = []
        system_x = []
        system_y = []
        for sd, array in iter_output(files, 'cylinders'):
            _t = sd['t']
            t.append(_t)
            # get the system center
            cm_y = array.xcm[1]
            system_y.append(cm_y)

        import matplotlib.pyplot as plt
        t = np.asarray(t)
        t = t - 0.1
        # print(t)

        # data = np.loadtxt('../x_com_zhang.csv', delimiter=',')
        # tx, xcom_zhang = data[:, 0], data[:, 1]

        # plt.plot(tx, xcom_zhang, "s--", label='Simulated PySPH')
        # plt.plot(t, system_x, "s-", label='Experimental')
        # plt.xlabel("time")
        # plt.ylabel("x/L")
        # plt.legend()
        # plt.savefig("xcom", dpi=300)
        # plt.clf()

        # data = np.loadtxt('../y_com_zhang.csv', delimiter=',')
        # ty, ycom_zhang = data[:, 0], data[:, 1]

        # plt.plot(ty, ycom_zhang, "s--", label='Experimental')
        plt.plot(t, system_y, "s-", label='Simulated PySPH')
        plt.xlabel("time")
        plt.ylabel("y/L")
        plt.legend()
        plt.savefig("ycom", dpi=300)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['cylinders']
        b.plot.actor.property.point_size = 2.
        ''')


if __name__ == '__main__':
    app = ZhangStackOfCylinders()
    # app.create_particles()
    # app.geometry()
    app.run()
    app.post_process(app.info_filename)
