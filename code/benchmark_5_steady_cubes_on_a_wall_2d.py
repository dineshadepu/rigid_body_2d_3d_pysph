"""
Test the collision of two rigid bodues made of same particle array
"""
import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from rigid_body_3d import RigidBody3DScheme
from rigid_body_2d import RigidBody2DScheme
from geometry import hydrostatic_tank_2d

from pysph.examples.solid_mech.impact import add_properties
from pysph.examples.rigid_body.sphere_in_vessel_akinci import (create_boundary,
                                                               create_fluid,
                                                               create_sphere)
from pysph.tools.geometry import get_2d_block


class Dinesh2022SteadyCubesOnAWall2D(Application):
    def initialize(self):
        spacing = 0.05
        self.hdx = 1.3

        self.fluid_length = 1.0
        self.fluid_height = 1.0
        self.fluid_density = 1000.0
        self.fluid_spacing = spacing

        self.tank_height = 1.5
        self.tank_layers = 5
        self.tank_spacing = spacing

        self.body_height = 0.2
        self.body_length = 0.2
        self.body_density = 2000
        self.body_spacing = spacing / 2.
        self.body_h = self.hdx * self.body_spacing

        self.h = self.hdx * self.fluid_spacing

        # self.solid_rho = 500
        # self.m = 1000 * self.dx * self.dx
        self.co = 10 * np.sqrt(2 * 9.81 * self.fluid_height)
        self.p0 = self.fluid_density * self.co**2.
        self.c0 = self.co
        self.alpha = 0.1
        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.dim = 2

    def add_user_options(self, group):
        from pysph.sph.scheme import add_bool_argument
        add_bool_argument(group, 'two-cubes', dest='use_two_cubes',
                          default=False, help='Use two cubes')

        add_bool_argument(group, 'three-cubes', dest='use_three_cubes',
                          default=False, help='Use three cubes')

        add_bool_argument(group, 'pyramid-cubes', dest='use_pyramid_cubes',
                          default=False, help='Use pyramid cubes')

    def consume_user_options(self):
        self.use_two_cubes = self.options.use_two_cubes
        self.use_three_cubes = self.options.use_three_cubes
        self.use_pyramid_cubes = self.options.use_pyramid_cubes

    def create_two_cubes(self):
        xb1, yb1 = get_2d_block(dx=self.body_spacing,
                                length=self.body_length,
                                height=self.body_height)

        xb2, yb2 = get_2d_block(dx=self.body_spacing,
                                length=self.body_length,
                                height=self.body_height)
        yb2 += max(yb1) - min(yb2) + self.body_spacing * 1.

        xb = np.concatenate([xb1, xb2])
        yb = np.concatenate([yb1, yb2])

        body_id1 = np.zeros(len(xb1), dtype=int)
        body_id2 = np.ones(len(xb2), dtype=int)
        body_id = np.concatenate([body_id1, body_id2])

        dem_id = np.concatenate([body_id1, body_id2])

        return xb, yb, body_id, dem_id

    def create_three_cubes(self):
        xb1, yb1 = get_2d_block(dx=self.body_spacing,
                                length=self.body_length,
                                height=self.body_height)

        xb2, yb2 = get_2d_block(dx=self.body_spacing,
                                length=self.body_length,
                                height=self.body_height)

        xb3, yb3 = get_2d_block(dx=self.body_spacing,
                                length=self.body_length,
                                height=self.body_height)

        yb2 += max(yb1) - min(yb2) + self.body_spacing * 1.
        yb3 += max(yb2) - min(yb3) + self.body_spacing * 1.

        xb = np.concatenate([xb1, xb2, xb3])
        yb = np.concatenate([yb1, yb2, yb3])

        body_id1 = np.zeros(len(xb1), dtype=int)
        body_id2 = np.ones(len(xb2), dtype=int)
        body_id3 = np.ones(len(xb3), dtype=int) * 2
        body_id = np.concatenate([body_id1, body_id2, body_id3])

        dem_id = np.concatenate([body_id1, body_id2, body_id3])

        return xb, yb, body_id, dem_id

    def create_pyramid_cubes(self):
        xb1, yb1 = get_2d_block(dx=self.body_spacing,
                                length=self.body_length,
                                height=self.body_height)

        xb2, yb2 = get_2d_block(dx=self.body_spacing,
                                length=self.body_length,
                                height=self.body_height)

        xb3, yb3 = get_2d_block(dx=self.body_spacing,
                                length=self.body_length,
                                height=self.body_height)

        xb4, yb4 = get_2d_block(dx=self.body_spacing,
                                length=self.body_length,
                                height=self.body_height)

        xb5, yb5 = get_2d_block(dx=self.body_spacing,
                                length=self.body_length,
                                height=self.body_height)

        xb6, yb6 = get_2d_block(dx=self.body_spacing,
                                length=self.body_length,
                                height=self.body_height)

        xb1 -= self.body_length
        xb2 += max(xb1) - min(xb2) + self.body_length/3.
        xb3 += max(xb2) - min(xb3) + self.body_length/3.

        xb4 += min(xb1) - min(xb4)
        xb4 += (self.body_length - self.body_length/3.)
        yb4 += max(yb1) - min(yb4) + self.body_spacing * 1.

        yb5 += max(yb4) - max(yb5)
        xb5 += max(xb3) - max(xb5) - (self.body_length - self.body_length/3.)

        yb6 += max(yb4) - min(yb6) + self.body_spacing * 1.
        xb6 += max(xb4) - max(xb6)
        xb6 += (max(xb5) - min(xb4)) / 2. - self.body_length/2.

        xb = np.concatenate([xb1, xb2, xb3, xb4, xb5, xb6])
        yb = np.concatenate([yb1, yb2, yb3, yb4, yb5, yb6])

        body_id1 = np.zeros(len(xb1), dtype=int)
        body_id2 = np.ones(len(xb2), dtype=int)
        body_id3 = np.ones(len(xb3), dtype=int) * 2
        body_id4 = np.ones(len(xb4), dtype=int) * 3
        body_id5 = np.ones(len(xb5), dtype=int) * 4
        body_id6 = np.ones(len(xb5), dtype=int) * 5
        body_id = np.concatenate([body_id1, body_id2, body_id3,
                                  body_id4, body_id5, body_id6])

        dem_id = np.concatenate([body_id1, body_id2, body_id3, body_id4, body_id5,
                                 body_id6])

        return xb, yb, body_id, dem_id

    def get_boundary_particles(self, no_bodies):
        from boundary_particles import (get_boundary_identification_etvf_equations,
                                        add_boundary_identification_properties)
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.base.kernels import (QuinticSpline)
        # create a row of six cylinders
        x, y = get_2d_block(dx=self.body_spacing,
                            length=self.body_length,
                            height=self.body_height)

        m = self.body_density * self.body_spacing**self.dim
        h = self.hdx * self.body_spacing
        rad_s = self.body_spacing / 2.
        pa = get_particle_array(name='foo',
                                x=x,
                                y=y,
                                rho=self.body_density,
                                h=h,
                                m=m,
                                rad_s=rad_s,
                                constants={
                                    'E': 69 * 1e9,
                                    'poisson_ratio': 0.3,
                                })

        add_boundary_identification_properties(pa)
        # make sure your rho is not zero
        equations = get_boundary_identification_etvf_equations([pa.name],
                                                               [pa.name])

        sph_eval = SPHEvaluator(arrays=[pa],
                                equations=equations,
                                dim=self.dim,
                                kernel=QuinticSpline(dim=self.dim))

        sph_eval.evaluate(dt=0.1)

        tmp = pa.is_boundary
        is_boundary_tmp = np.tile(tmp, no_bodies)
        is_boundary = is_boundary_tmp.ravel()

        return is_boundary

    def create_particles(self):
        xf, yf, xt, yt = hydrostatic_tank_2d(
            self.fluid_length, self.fluid_height, self.tank_height,
            self.tank_layers, self.body_spacing, self.body_spacing)

        m_fluid = self.fluid_density * self.fluid_spacing**2.

        if self.use_two_cubes is True:
            xb, yb, body_id, dem_id = self.create_two_cubes()
        elif self.use_three_cubes is True:
            xb, yb, body_id, dem_id = self.create_three_cubes()
        elif self.use_pyramid_cubes is True:
            xb, yb, body_id, dem_id = self.create_pyramid_cubes()
        else:
            print("=====================================")
            print("Choose among the given configurations")
            print("=====================================")
            print("=                                   =")
            print("=      Two cubes                    =")
            print("=                                   =")
            print("=      Three cubes                  =")
            print("=                                   =")
            print("=                                   =")
            print("=      Pyramid cubes                =")
            print("=                                   =")
            print("=                                   =")
            print("=                                   =")
            print("=====================================")

        m = self.body_density * self.body_spacing**self.dim
        body = get_particle_array(name='body',
                                  x=xb,
                                  y=yb,
                                  h=self.body_h,
                                  m=m,
                                  rho=self.body_density,
                                  m_fluid=m_fluid,
                                  rad_s=self.body_spacing / 2.,
                                  constants={
                                      'E': 69 * 1e9,
                                      'poisson_ratio': 0.3,
                                      'spacing0': self.body_spacing,
                                  })
        body.y[:] += self.body_height * 2.

        body.add_property('body_id', type='int', data=body_id)
        body.add_property('dem_id', type='int', data=dem_id)
        body.add_constant('total_no_bodies', [max(body_id) + 2])

        # ===============================
        # create a tank
        # ===============================
        x, y = xt, yt
        dem_id = body_id
        m = self.body_density * self.body_spacing**2
        h = self.body_h
        rad_s = self.body_spacing / 2.

        tank = get_particle_array(name='tank',
                                  x=x,
                                  y=y,
                                  h=h,
                                  m=m,
                                  rho=self.body_density,
                                  rad_s=rad_s,
                                  constants={
                                      'E': 69 * 1e9,
                                      'poisson_ratio': 0.3,
                                  })
        max_dem_id = max(dem_id)
        tank.add_property('dem_id', type='int', data=max_dem_id + 1)

        # ==================================================
        # adjust the rigid body positions on top of the wall
        # ==================================================
        body.y[:] -= min(body.y) - min(tank.y)
        body.y[:] += self.tank_layers * self.body_spacing

        self.scheme.setup_properties([body, tank])

        # reset the boundary particles, this is due to improper boundary
        # particle identification by the setup properties
        is_boundary = self.get_boundary_particles(body.total_no_bodies[0] - 1)
        body.is_boundary[:] = is_boundary[:]

        body.add_property('contact_force_is_boundary')
        body.contact_force_is_boundary[:] = body.is_boundary[:]

        tank.add_property('contact_force_is_boundary')
        tank.contact_force_is_boundary[:] = tank.is_boundary[:]
        return [body, tank]

    def create_scheme(self):
        rb3d = RigidBody3DScheme(rigid_bodies=['body'],
                                 boundaries=['tank'],
                                 gx=self.gx,
                                 gy=self.gy,
                                 gz=self.gz,
                                 dim=2)

        rb2d = RigidBody2DScheme(rigid_bodies=['body'],
                                 boundaries=['tank'],
                                 gx=self.gx,
                                 gy=self.gy,
                                 gz=self.gz,
                                 dim=2)
        s = SchemeChooser(default='rb2d', rb3d=rb3d, rb2d=rb2d)
        return s

    def configure_scheme(self):
        dt = 1e-4
        print("DT: %s" % dt)
        tf = 0.5

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)


if __name__ == '__main__':
    app = Dinesh2022SteadyCubesOnAWall2D()
    app.run()
    # app.post_process(app.info_filename)
