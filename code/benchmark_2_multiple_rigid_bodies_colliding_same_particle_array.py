"""
Test the collision of two rigid bodues
"""
import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from rigid_body_3d import RigidBody3DScheme
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
        self.tank_layers = 3
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
        self.gy = 0.
        self.dim = 2

    def create_particles(self):
        xb, yb = get_2d_block(dx=self.body_spacing,
                              length=self.body_length,
                              height=self.body_height)
        m = self.body_density * self.body_spacing**self.dim
        body1 = get_particle_array(name='body1',
                                   x=xb,
                                   y=yb,
                                   h=self.body_h,
                                   m=m,
                                   rho=self.body_density,
                                   m_fluid=0.,
                                   rad_s=self.body_spacing / 2.,
                                   constants={
                                       'E': 69 * 1e9,
                                       'poisson_ratio': 0.3,
                                       'spacing0': self.body_spacing,
                                   })
        body_id = np.zeros(len(xb), dtype=int)
        dem_id = np.zeros(len(xb), dtype=int)
        body1.add_property('body_id', type='int', data=body_id)
        body1.add_property('dem_id', type='int', data=dem_id)
        body1.add_constant('total_no_bodies', [2])

        xb, yb = get_2d_block(dx=self.body_spacing,
                              length=self.body_length,
                              height=self.body_height)
        m = self.body_density * self.body_spacing**self.dim
        xb += 2. * self.body_length
        body2 = get_particle_array(name='body2',
                                   x=xb,
                                   y=yb,
                                   h=self.body_h,
                                   m=m,
                                   rho=self.body_density,
                                   m_fluid=0.,
                                   rad_s=self.body_spacing / 2.,
                                   constants={
                                       'E': 69 * 1e9,
                                       'poisson_ratio': 0.3,
                                       'spacing0': self.body_spacing,
                                   })
        body_id = np.zeros(len(xb), dtype=int)
        dem_id = np.ones(len(xb), dtype=int)
        body2.add_property('body_id', type='int', data=body_id)
        body2.add_property('dem_id', type='int', data=dem_id)
        body2.add_constant('total_no_bodies', [2])

        self.scheme.setup_properties([body1, body2])

        body1.add_property('contact_force_is_boundary')
        body1.contact_force_is_boundary[:] = body1.is_boundary[:]
        body2.add_property('contact_force_is_boundary')
        body2.contact_force_is_boundary[:] = body2.is_boundary[:]

        self.scheme.scheme.set_linear_velocity(body1, np.array([0.5, 0., 0.]))
        self.scheme.scheme.set_linear_velocity(body2, np.array([-0.5, 0., 0.]))
        return [body1, body2]

    def create_scheme(self):
        rb3d = RigidBody3DScheme(rigid_bodies=['body1', 'body2'],
                                 boundaries=None, dim=2)

        rb2d = RigidBody3DScheme(rigid_bodies=['body1', 'body2'],
                                 boundaries=None, dim=2)
        s = SchemeChooser(default='rb2d', rb3d=rb3d, rb2d=rb2d)
        return s

    def configure_scheme(self):
        dt = 0.125 * self.fluid_spacing * self.hdx / (self.co * 1.1)
        print("DT: %s" % dt)
        tf = 0.5

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
    # app.post_process(app.info_filename)


# ft_x, ft_y, z
# fn_x, fn_y, z
# u, v, w
# delta_lt_x, delta_lt_y, delta_lt_z
# contact_force_normal_x, contact_force_normal_y, z
