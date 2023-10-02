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


class RigidFluidCoupling(Application):
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

    def create_particles(self):
        xf, yf, xt, yt = hydrostatic_tank_2d(
            self.fluid_length, self.fluid_height, self.tank_height,
            self.tank_layers, self.body_spacing, self.body_spacing)

        m_fluid = self.fluid_density * self.fluid_spacing**2.

        xb1, yb1 = get_2d_block(dx=self.body_spacing,
                                length=self.body_length,
                                height=self.body_height)
        m = self.body_density * self.body_spacing**self.dim

        xb2 = xb1 + self.body_length * 2
        yb2 = yb1

        xb = np.concatenate([xb1, xb2])
        yb = np.concatenate([yb1, yb2])

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
        body.x[:] -= self.body_length/2.
        body_id1 = np.zeros(len(xb1), dtype=int)
        body_id2 = np.ones(len(xb2), dtype=int)
        body_id = np.concatenate([body_id1, body_id2])

        dem_id = np.concatenate([body_id1, body_id2])

        body.add_property('body_id', type='int', data=body_id)
        body.add_property('dem_id', type='int', data=dem_id)
        body.add_constant('total_no_bodies', [3])

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

        self.scheme.setup_properties([body, tank])

        body.add_property('contact_force_is_boundary')
        body.contact_force_is_boundary[:] = body.is_boundary[:]

        tank.add_property('contact_force_is_boundary')
        tank.contact_force_is_boundary[:] = tank.is_boundary[:]

        # self.scheme.scheme.set_linear_velocity(
        #     body, np.array([
        #         0.5,
        #         0.,
        #         0.,
        #         -0.5,
        #         0.,
        #         0.,
        #     ]))

        # # remove particles outside the circle
        # indices = []
        # for i in range(len(tank.x)):
        #     if tank.is_boundary[i] == 0:
        #         indices.append(i)

        # tank.remove_particles(indices)

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
        tf = 1.

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
    # app.post_process(app.info_filename)
