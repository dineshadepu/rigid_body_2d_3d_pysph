"""

We have use 3D formaulation rigid body using rotation matrices to evolve the
dynamics of the bodies. So we will use one formualtion for both 2d and 3d
simulations.

Only GTVF (Leap frog) integrator is used. This will make the interaction between
the rigid bodies seamless as the contact force law DEM is easier to implement.

"""
import numpy as np

from pysph.sph.scheme import add_bool_argument

from pysph.sph.equation import Equation
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.wc.gtvf import GTVFIntegrator
from pysph.base.kernels import (CubicSpline, WendlandQuintic, QuinticSpline)
from textwrap import dedent
from compyle.api import declare

from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep

# from pysph.sph.rigid_body import (BodyForce)

from pysph.base.kernels import (CubicSpline, WendlandQuintic, QuinticSpline,
                                WendlandQuinticC4, Gaussian, SuperGaussian)

from rigid_body_common import (add_properties_stride,
                               set_total_mass, set_center_of_mass,
                               set_body_frame_position_vectors,
                               set_body_frame_normal_vectors,
                               set_moment_of_inertia_and_its_inverse,
                               BodyForce, SumUpExternalForces,
                               normalize_R_orientation,
                               ComputeContactForceNormals,
                               ComputeContactForceDistanceAndClosestPoint,
                               ComputeContactForce)

# compute the boundary particles
from boundary_particles import (get_boundary_identification_etvf_equations,
                                add_boundary_identification_properties)
from numpy import sin, cos


class ClampWallPressure(Equation):
    r"""Clamp the wall pressure to non-negative values.
    """
    def post_loop(self, d_idx, d_p):
        if d_p[d_idx] < 0.0:
            d_p[d_idx] = 0.0


class ClampWallPressureFSI(Equation):
    r"""Clamp the wall pressure to non-negative values.
    """
    def post_loop(self, d_idx, d_p_fsi):
        if d_p_fsi[d_idx] < 0.0:
            d_p_fsi[d_idx] = 0.0


class ContinuityEquation(Equation):
    r"""Density rate:

    :math:`\frac{d\rho_a}{dt} = \sum_b m_b \boldsymbol{v}_{ab}\cdot
    \nabla_a W_{ab}`

    """
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_arho, s_idx, s_rho, s_m, DWIJ, VIJ):
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        fac = d_rho[d_idx] * s_m[s_idx] / s_rho[s_idx]
        d_arho[d_idx] += fac*vijdotdwij


class ContinuityEquationFSI(Equation):
    r"""Density rate:

    :math:`\frac{d\rho_a}{dt} = \sum_b m_b \boldsymbol{v}_{ab}\cdot
    \nabla_a W_{ab}`

    """
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_arho, s_idx, s_m_fsi, s_rho_fsi, DWIJ, VIJ):
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]

        fac = d_rho[d_idx] * s_m_fsi[s_idx] / s_rho_fsi[s_idx]
        d_arho[d_idx] += fac*vijdotdwij


class SolidWallPressureBCFSI(Equation):
    r"""Solid wall pressure boundary condition from Adami and Hu (transport
    velocity formulation).

    """
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz

        super(SolidWallPressureBCFSI, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p_fsi):
        d_p_fsi[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p_fsi, s_p, s_rho, d_au, d_av, d_aw, WIJ,
             XIJ):

        # numerator of Eq. (27) ax, ay and az are the prescribed wall
        # accelerations which must be defined for the wall boundary
        # particle
        gdotxij = (self.gx - d_au[d_idx])*XIJ[0] + \
            (self.gy - d_av[d_idx])*XIJ[1] + \
            (self.gz - d_aw[d_idx])*XIJ[2]

        # max_val = max(0., gdotxij)
        max_val = gdotxij
        d_p_fsi[d_idx] += s_p[s_idx] * WIJ + s_rho[s_idx] * max_val * WIJ

    def post_loop(self, d_idx, d_wij, d_p_fsi, d_rho_fsi):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_p_fsi[d_idx] /= d_wij[d_idx]

        # d_rho_fsi[d_idx] /= d_wij[d_idx]


class MomentumEquationPressureGradient(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationPressureGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

    def loop(self, d_rho, s_rho, d_idx, s_idx, d_p, s_p, s_m, d_au, d_av, d_aw,
             DWIJ, XIJ, RIJ, SPH_KERNEL, HIJ):
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pij = d_p[d_idx] / rhoi2 + s_p[s_idx] / rhoj2

        tmp = -s_m[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class MomentumEquationPressureGradientBoundary(Equation):
    def loop(self, d_rho, s_rho, d_idx, s_idx, d_p, s_p, s_m_fluid, d_au, d_av,
             d_aw, DWIJ, XIJ, RIJ, SPH_KERNEL, HIJ):
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pij = d_p[d_idx] / rhoi2 + s_p[s_idx] / rhoj2

        tmp = -s_m_fluid[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class ForceOnFluidDuetoRigidBody(Equation):
    def __init__(self, dest, sources):
        super(ForceOnFluidDuetoRigidBody, self).__init__(dest, sources)

    def loop(self, d_rho, s_rho_fsi, d_idx, s_idx, d_p, s_p_fsi, s_m_fsi,
             d_au, d_av, d_aw, DWIJ, XIJ, RIJ, SPH_KERNEL, HIJ):
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho_fsi[s_idx] * s_rho_fsi[s_idx]

        pij = d_p[d_idx] / rhoi2 + s_p_fsi[s_idx] / rhoj2

        tmp = -s_m_fsi[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class ForceOnRigidBodyDuetoFluid(Equation):
    """Force between a solid sphere and a SPH fluid particle.  This is
    implemented using Akinci's[1] force and additional force from solid
    bodies pressure which is implemented by Liu[2]

    [1]'Versatile Rigid-Fluid Coupling for Incompressible SPH'

    URL: https://graphics.ethz.ch/~sobarbar/papers/Sol12/Sol12.pdf

    [2]A 3D Simulation of a Moving Solid in Viscous Free-Surface Flows by
    Coupling SPH and DEM

    https://doi.org/10.1155/2017/3174904


    Note: Here forces for both the phases are added at once.
          Please make sure that this force is applied only once
          for both the particle properties.

    """
    def __init__(self, dest, sources):
        super(ForceOnRigidBodyDuetoFluid, self).__init__(dest, sources)

    def loop(self, d_idx, d_m_fsi, d_rho_fsi, d_p_fsi, s_idx, d_fx, d_fy,
             d_fz, DWIJ, s_m, s_p, s_rho):
        _t1 = s_p[s_idx] / (s_rho[s_idx]**2) + d_p_fsi[d_idx] / (d_rho_fsi[d_idx]**2)

        d_fx[d_idx] -= d_m_fsi[d_idx] * s_m[s_idx] * _t1 * DWIJ[0]
        d_fy[d_idx] -= d_m_fsi[d_idx] * s_m[s_idx] * _t1 * DWIJ[1]
        d_fz[d_idx] -= d_m_fsi[d_idx] * s_m[s_idx] * _t1 * DWIJ[2]


class RK2FluidStep(IntegratorStep):
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0,
                   d_w0, d_u, d_v, d_w, d_rho0, d_rho):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]

    def stage1(self, d_idx, d_m, d_vol, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0,
               d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av, d_aw,
               d_arho, dt):
        dtb2 = 0.5 * dt
        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2 * d_aw[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]
        d_vol[d_idx] = d_m[d_idx] / d_rho[d_idx]

    def stage2(self, d_idx, d_m, d_vol, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0,
               d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av, d_aw,
               d_arho, dt):
        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_aw[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]
        d_vol[d_idx] = d_m[d_idx] / d_rho[d_idx]


class GTVFFluidStep(IntegratorStep):
    def stage1(self, d_idx, d_m, d_vol, d_x, d_y, d_z, d_u, d_v, d_w, d_au,
               d_av, d_aw, dt):
        dtb2 = 0.5 * dt
        d_u[d_idx] = d_u[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_v[d_idx] + dtb2 * d_av[d_idx]
        d_w[d_idx] = d_w[d_idx] + dtb2 * d_aw[d_idx]

    def stage2(self, d_idx, d_m, d_vol, d_x, d_y, d_z, d_u, d_v, d_w, d_rho,
               d_au, d_av, d_aw, d_p, d_arho, d_ap, dt):
        d_x[d_idx] = d_x[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z[d_idx] + dt * d_w[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho[d_idx] + dt * d_arho[d_idx]
        d_p[d_idx] = d_p[d_idx] + dt * d_ap[d_idx]

        d_vol[d_idx] = d_m[d_idx] / d_rho[d_idx]

    def stage3(self, d_idx, d_m, d_vol, d_x, d_y, d_z, d_u, d_v, d_w, d_au,
               d_av, d_aw, dt):
        dtb2 = 0.5 * dt
        d_u[d_idx] = d_u[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_v[d_idx] + dtb2 * d_av[d_idx]
        d_w[d_idx] = d_w[d_idx] + dtb2 * d_aw[d_idx]


class GTVFRigidBody3DStep(IntegratorStep):
    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.vcm[i3 + j] = dst.vcm[i3 + j] + (dtb2 * dst.force[i3 + j] /
                                                     dst.total_mass[i])

            # move angular velocity to t + dt/2.
            # omega_dot is
            dst.ang_mom[i3:i3 +
                        3] = dst.ang_mom[i3:i3 + 3] + (dtb2 *
                                                       dst.torque[i3:i3 + 3])

            dst.omega[i3:i3 + 3] = np.matmul(
                dst.inertia_tensor_inverse_global_frame[i9:i9 + 9].reshape(
                    3, 3), dst.ang_mom[i3:i3 + 3])

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id, d_is_boundary):
        # Update the velocities to 1/2. time step
        # some variables to update the positions seamlessly

        bid, i9, i3, = declare('int', 3)
        bid = d_body_id[d_idx]
        i9 = 9 * bid
        i3 = 3 * bid

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[i9 + 0] * d_dx0[d_idx] + d_R[i9 + 1] * d_dy0[d_idx] +
              d_R[i9 + 2] * d_dz0[d_idx])
        dy = (d_R[i9 + 3] * d_dx0[d_idx] + d_R[i9 + 4] * d_dy0[d_idx] +
              d_R[i9 + 5] * d_dz0[d_idx])
        dz = (d_R[i9 + 6] * d_dx0[d_idx] + d_R[i9 + 7] * d_dy0[d_idx] +
              d_R[i9 + 8] * d_dz0[d_idx])

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[i3 + 1] * dz - d_omega[i3 + 2] * dy
        dv = d_omega[i3 + 2] * dx - d_omega[i3 + 0] * dz
        dw = d_omega[i3 + 0] * dy - d_omega[i3 + 1] * dx

        d_u[d_idx] = d_vcm[i3 + 0] + du
        d_v[d_idx] = d_vcm[i3 + 1] + dv
        d_w[d_idx] = d_vcm[i3 + 2] + dw

    def py_stage2(self, dst, t, dt):
        # move positions to t + dt time step
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.xcm[i3 + j] = dst.xcm[i3 + j] + dt * dst.vcm[i3 + j]

            # angular velocity in terms of matrix
            omega_mat = np.array([[0, -dst.omega[i3 + 2], dst.omega[i3 + 1]],
                                  [dst.omega[i3 + 2], 0, -dst.omega[i3 + 0]],
                                  [-dst.omega[i3 + 1], dst.omega[i3 + 0], 0]])

            # Currently the orientation is at time t
            R = dst.R[i9:i9 + 9].reshape(3, 3)

            # Rate of change of orientation is
            r_dot = np.matmul(omega_mat, R)
            r_dot = r_dot.ravel()

            # update the orientation to next time step
            dst.R[i9:i9 + 9] = dst.R[i9:i9 + 9] + r_dot * dt

            # normalize the orientation using Gram Schmidt process
            normalize_R_orientation(dst.R[i9:i9 + 9])

            # update the moment of inertia
            R = dst.R[i9:i9 + 9].reshape(3, 3)
            R_t = R.transpose()
            tmp = np.matmul(
                R,
                dst.inertia_tensor_inverse_body_frame[i9:i9 + 9].reshape(3, 3))
            dst.inertia_tensor_inverse_global_frame[i9:i9 + 9] = (np.matmul(
                tmp, R_t)).ravel()[:]

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id, d_normal0, d_normal,
               d_is_boundary):
        # some variables to update the positions seamlessly
        bid, i9, i3, idx3 = declare('int', 4)
        bid = d_body_id[d_idx]
        idx3 = 3 * d_idx
        i9 = 9 * bid
        i3 = 3 * bid

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[i9 + 0] * d_dx0[d_idx] + d_R[i9 + 1] * d_dy0[d_idx] +
              d_R[i9 + 2] * d_dz0[d_idx])
        dy = (d_R[i9 + 3] * d_dx0[d_idx] + d_R[i9 + 4] * d_dy0[d_idx] +
              d_R[i9 + 5] * d_dz0[d_idx])
        dz = (d_R[i9 + 6] * d_dx0[d_idx] + d_R[i9 + 7] * d_dy0[d_idx] +
              d_R[i9 + 8] * d_dz0[d_idx])

        d_x[d_idx] = d_xcm[i3 + 0] + dx
        d_y[d_idx] = d_xcm[i3 + 1] + dy
        d_z[d_idx] = d_xcm[i3 + 2] + dz

        # update normal vectors of the boundary
        if d_is_boundary[d_idx] == 1:
            d_normal[idx3 + 0] = (d_R[i9 + 0] * d_normal0[idx3] +
                                  d_R[i9 + 1] * d_normal0[idx3 + 1] +
                                  d_R[i9 + 2] * d_normal0[idx3 + 2])
            d_normal[idx3 + 1] = (d_R[i9 + 3] * d_normal0[idx3] +
                                  d_R[i9 + 4] * d_normal0[idx3 + 1] +
                                  d_R[i9 + 5] * d_normal0[idx3 + 2])
            d_normal[idx3 + 2] = (d_R[i9 + 6] * d_normal0[idx3] +
                                  d_R[i9 + 7] * d_normal0[idx3 + 1] +
                                  d_R[i9 + 8] * d_normal0[idx3 + 2])

    def py_stage3(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.vcm[i3 + j] = dst.vcm[i3 + j] + (dtb2 * dst.force[i3 + j] /
                                                     dst.total_mass[i])

            # move angular velocity to t + dt/2.
            # omega_dot is
            dst.ang_mom[i3:i3 +
                        3] = dst.ang_mom[i3:i3 + 3] + (dtb2 *
                                                       dst.torque[i3:i3 + 3])

            dst.omega[i3:i3 + 3] = np.matmul(
                dst.inertia_tensor_inverse_global_frame[i9:i9 + 9].reshape(
                    3, 3), dst.ang_mom[i3:i3 + 3])

    def stage3(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id, d_is_boundary):
        # Update the velocities to 1/2. time step
        # some variables to update the positions seamlessly

        bid, i9, i3, = declare('int', 3)
        bid = d_body_id[d_idx]
        i9 = 9 * bid
        i3 = 3 * bid

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[i9 + 0] * d_dx0[d_idx] + d_R[i9 + 1] * d_dy0[d_idx] +
              d_R[i9 + 2] * d_dz0[d_idx])
        dy = (d_R[i9 + 3] * d_dx0[d_idx] + d_R[i9 + 4] * d_dy0[d_idx] +
              d_R[i9 + 5] * d_dz0[d_idx])
        dz = (d_R[i9 + 6] * d_dx0[d_idx] + d_R[i9 + 7] * d_dy0[d_idx] +
              d_R[i9 + 8] * d_dz0[d_idx])

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[i3 + 1] * dz - d_omega[i3 + 2] * dy
        dv = d_omega[i3 + 2] * dx - d_omega[i3 + 0] * dz
        dw = d_omega[i3 + 0] * dy - d_omega[i3 + 1] * dx

        d_u[d_idx] = d_vcm[i3 + 0] + du
        d_v[d_idx] = d_vcm[i3 + 1] + dv
        d_w[d_idx] = d_vcm[i3 + 2] + dw


class EDACEquation(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu

        super(EDACEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ap):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, d_m, d_rho, d_ap, d_p, s_idx, s_m, s_rho, s_p,
             d_c0_ref, DWIJ, VIJ, XIJ, R2IJ, EPS):
        Vi = d_m[d_idx]/d_rho[d_idx]
        Vj = s_m[s_idx]/s_rho[s_idx]
        Vi2 = Vi * Vi
        Vj2 = Vj * Vj
        cs2 = d_c0_ref[0] * d_c0_ref[0]

        etai = d_rho[d_idx]
        etaj = s_rho[s_idx]
        etaij = 2 * self.nu * (etai * etaj)/(etai + etaj)

        # This is the same as continuity acceleration times cs^2
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        d_ap[d_idx] += rhoi/rhoj*cs2*s_m[s_idx]*vijdotdwij

        # Viscous damping of pressure.
        xijdotdwij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]
        tmp = 1.0/d_m[d_idx]*(Vi2 + Vj2)*etaij*xijdotdwij/(R2IJ + EPS)
        d_ap[d_idx] += tmp*(d_p[d_idx] - s_p[s_idx])


class EDACEquationFSI(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu

        super(EDACEquationFSI, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ap):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, d_m, d_rho, d_ap, d_p, d_c0_ref, s_idx, s_m_fsi,
             s_rho_fsi, s_p_fsi, DWIJ, VIJ, XIJ, R2IJ, EPS):
        Vi = d_m[d_idx]/d_rho[d_idx]
        Vj = s_m_fsi[s_idx]/s_rho_fsi[s_idx]
        Vi2 = Vi * Vi
        Vj2 = Vj * Vj
        cs2 = d_c0_ref[0] * d_c0_ref[0]

        etai = d_rho[d_idx]
        etaj = s_rho_fsi[s_idx]
        etaij = 2 * self.nu * (etai * etaj)/(etai + etaj)

        # This is the same as continuity acceleration times cs^2
        rhoi = d_rho[d_idx]
        rhoj = s_rho_fsi[s_idx]
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        d_ap[d_idx] += rhoi/rhoj*cs2*s_m_fsi[s_idx]*vijdotdwij

        # Viscous damping of pressure.
        xijdotdwij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]
        tmp = 1.0/d_m[d_idx]*(Vi2 + Vj2)*etaij*xijdotdwij/(R2IJ + EPS)
        d_ap[d_idx] += tmp*(d_p[d_idx] - s_p_fsi[s_idx])


class RigidFluidCouplingScheme(Scheme):
    def __init__(self, fluids, boundaries, rigid_bodies, dim, rho0, p0, c0, h,
                 nu, kr=1e5, kf=1e5, en=0.5, fric_coeff=0.5, gamma=7.0,
                 gx=0.0, gy=0.0, gz=0.0, alpha=0.1, beta=0.0,
                 kernel_choice="1", kernel_factor=3, edac_alpha=0.5):
        if boundaries is None:
            self.boundaries = []
        else:
            self.boundaries = boundaries

        if fluids is None:
            self.fluids = []
        else:
            self.fluids = fluids

        if rigid_bodies is None:
            self.rigid_bodies = []
        else:
            self.rigid_bodies = rigid_bodies

        # fluids parameters
        self.edac = False
        self.edac_alpha = edac_alpha
        self.h = h
        self.art_nu = 0.
        self.nu = nu

        self.dim = dim

        self.kernel = QuinticSpline

        self.rho0 = rho0
        self.p0 = p0
        self.c0 = c0
        self.gamma = gamma

        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.kr = kr
        self.kf = kf
        self.fric_coeff = fric_coeff
        self.en = en

        self.fluid_alpha = alpha
        self.beta = beta

        self.solver = None

    def add_user_options(self, group):
        group.add_argument("--kr-stiffness", action="store",
                           dest="kr", default=1e5,
                           type=float,
                           help="Repulsive spring stiffness")

        group.add_argument("--kf-stiffness", action="store",
                           dest="kf", default=1e3,
                           type=float,
                           help="Tangential spring stiffness")

        group.add_argument("--fric-coeff", action="store",
                           dest="fric_coeff", default=0.5,
                           type=float,
                           help="Friction coefficient")

        group.add_argument("--fluid-alpha", action="store",
                           dest="fluid_alpha", default=0.5,
                           type=float,
                           help="Artificial viscosity")

        add_bool_argument(group, 'edac', dest='edac', default=True,
                          help='Use pressure evolution equation EDAC')

    def consume_user_options(self, options):
        _vars = ['kr', 'kf', 'fric_coeff', 'fluid_alpha', 'edac']
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def attributes_changed(self):
        self.edac_nu = self.fluid_alpha * self.h * self.c0 / 8

    def get_equations(self):
        # elastic solid equations
        from pysph.sph.equation import Group
        from pysph.sph.wc.basic import (MomentumEquation, TaitEOS)
        from pysph.sph.wc.transport_velocity import (
            SetWallVelocity,
            MomentumEquationArtificialViscosity)
        from pysph.sph.wc.viscosity import (LaminarViscosity)
        from pysph.sph.basic_equations import (MonaghanArtificialViscosity,
                                               XSPHCorrection,
                                               VelocityGradient2D)
        from pysph.sph.wc.edac import (SolidWallPressureBC)

        all = self.fluids + self.boundaries + self.rigid_bodies
        edac_nu = self.edac_nu

        stage1 = []

        eqs = []
        for fluid in self.fluids:
            eqs.append(ContinuityEquation(dest=fluid,
                                          sources=self.fluids+self.boundaries), )
            if self.edac == True:
                eqs.append(
                    EDACEquation(
                        dest=fluid, sources=self.fluids+self.boundaries, nu=edac_nu), )

        if len(self.rigid_bodies) > 0:
            for fluid in self.fluids:
                eqs.append(
                    ContinuityEquationFSI(dest=fluid,
                                          sources=self.rigid_bodies), )
                if self.edac == True:
                    eqs.append(
                        EDACEquationFSI(dest=fluid,
                                        sources=self.rigid_bodies, nu=edac_nu), )

        stage1.append(Group(equations=eqs, real=False))

        # ==============================
        # Stage 2 equations
        # ==============================
        stage2 = []
        g2 = []

        if len(self.fluids) > 0:
            tmp = []
            if self.edac == False:
                for fluid in self.fluids:
                    tmp.append(
                        TaitEOS(dest=fluid, sources=None, rho0=self.rho0, c0=self.c0,
                                gamma=self.gamma))

                stage2.append(Group(equations=tmp, real=False))

        if len(self.fluids) > 0:
            tmp = []
            for solid in self.boundaries:
                tmp.append(
                    SetWallVelocity(dest=solid,
                                    sources=self.fluids))
                tmp.append(
                    SolidWallPressureBC(dest=solid,
                                        sources=self.fluids,
                                        gx=self.gx,
                                        gy=self.gy,
                                        gz=self.gz))
                tmp.append(
                    ClampWallPressure(dest=solid, sources=None))

            for solid in self.rigid_bodies:
                tmp.append(
                    SetWallVelocity(dest=solid,
                                    sources=self.fluids))
                tmp.append(
                    SolidWallPressureBCFSI(dest=solid,
                                           sources=self.fluids,
                                           gx=self.gx,
                                           gy=self.gy,
                                           gz=self.gz))
                # tmp.append(
                #     ClampWallPressureFSI(dest=solid, sources=None))

            if len(tmp) > 0:
                stage2.append(Group(equations=tmp, real=False))

        if len(self.fluids) > 0:
            for name in self.fluids:
                alpha = self.fluid_alpha
                g2.append(
                    MomentumEquationPressureGradient(dest=name,
                                                     sources=self.fluids+self.boundaries,
                                                     gx=self.gx,
                                                     gy=self.gy,
                                                     gz=self.gz))

                if abs(alpha) > 1e-14:
                    eq = MomentumEquationArtificialViscosity(dest=name,
                                                             sources=self.fluids,
                                                             c0=self.c0,
                                                             alpha=self.fluid_alpha)
                    g2.insert(-1, eq)

                if len(self.rigid_bodies) > 0:
                    g2.append(
                        ForceOnFluidDuetoRigidBody(dest=name,
                                                   sources=self.rigid_bodies))
            stage2.append(Group(equations=g2))

        #######################
        # Handle rigid bodies #
        #######################
        if len(self.rigid_bodies) > 0:
            g5 = []
            for name in self.rigid_bodies:
                g5.append(
                    ComputeContactForceNormals(dest=name,
                                               sources=self.rigid_bodies+self.boundaries))

            stage2.append(Group(equations=g5, real=False))

            g5 = []
            for name in self.rigid_bodies:
                g5.append(
                    ComputeContactForceDistanceAndClosestPoint(
                        dest=name, sources=self.rigid_bodies+self.boundaries))
            stage2.append(Group(equations=g5, real=False))

        if len(self.rigid_bodies) > 0:
            g5 = []
            for name in self.rigid_bodies:
                g5.append(
                    BodyForce(dest=name,
                              sources=None,
                              gx=self.gx,
                              gy=self.gy,
                              gz=self.gz))
            stage2.append(Group(equations=g5, real=False))

            g5 = []
            for name in self.rigid_bodies:
                g5.append(
                    ComputeContactForce(dest=name,
                                        sources=None,
                                        kr=self.kr,
                                        kf=self.kf,
                                        fric_coeff=self.fric_coeff))

            # add the force due to fluid
            if len(self.fluids) > 0:
                for name in self.rigid_bodies:
                    g5.append(ForceOnRigidBodyDuetoFluid(dest=name,
                                                         sources=self.fluids))

            stage2.append(Group(equations=g5, real=False))

            # computation of total force and torque at center of mass
            g6 = []
            for name in self.rigid_bodies:
                g6.append(SumUpExternalForces(dest=name, sources=None))

            stage2.append(Group(equations=g6, real=False))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self,
                         kernel=None,
                         integrator_cls=None,
                         extra_steppers=None,
                         **kw):
        from pysph.base.kernels import QuinticSpline
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = QuinticSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        fluidstep = GTVFFluidStep()
        bodystep = GTVFRigidBody3DStep()
        integrator_cls = GTVFIntegrator

        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = fluidstep

        for body in self.rigid_bodies:
            if body not in steppers:
                steppers[body] = bodystep

        cls = integrator_cls
        integrator = cls(**steppers)

        self.solver = Solver(dim=self.dim,
                             integrator=integrator,
                             kernel=kernel,
                             **kw)

    def setup_properties(self, particles, clean=True):
        from pysph.examples.solid_mech.impact import add_properties

        pas = dict([(p.name, p) for p in particles])

        for rigid_body in self.rigid_bodies:
            pa = pas[rigid_body]

            # properties to find the find on the rigid body by
            # Mofidi, Drescher, Emden, Teschner
            add_properties_stride(pa, pa.total_no_bodies[0],
                                  'contact_force_normal_x',
                                  'contact_force_normal_y',
                                  'contact_force_normal_z',
                                  'contact_force_normal_wij',

                                  'contact_force_normal_tmp_x',
                                  'contact_force_normal_tmp_y',
                                  'contact_force_normal_tmp_z',

                                  'contact_force_dist_tmp',
                                  'contact_force_dist',

                                  'overlap',
                                  'ft_x',
                                  'ft_y',
                                  'ft_z',
                                  'fn_x',
                                  'fn_y',
                                  'fn_z',
                                  'delta_lt_x',
                                  'delta_lt_y',
                                  'delta_lt_z',
                                  'vx_source',
                                  'vy_source',
                                  'vz_source',
                                  'x_source',
                                  'y_source',
                                  'z_source',
                                  'ti_x',
                                  'ti_y',
                                  'ti_z',
                                  'closest_point_dist_to_source')

            add_properties(pa, 'fx', 'fy', 'fz', 'dx0', 'dy0', 'dz0')

            nb = int(np.max(pa.body_id) + 1)

            # dem_id = props.pop('dem_id', None)

            consts = {
                'total_mass':
                np.zeros(nb, dtype=float),
                'xcm':
                np.zeros(3 * nb, dtype=float),
                'xcm0':
                np.zeros(3 * nb, dtype=float),
                'R': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb,
                'R0': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb,
                # moment of inertia izz (this is only for 2d)
                'izz':
                np.zeros(nb, dtype=float),
                # moment of inertia inverse in body frame
                'inertia_tensor_body_frame':
                np.zeros(9 * nb, dtype=float),
                # moment of inertia inverse in body frame
                'inertia_tensor_inverse_body_frame':
                np.zeros(9 * nb, dtype=float),
                # moment of inertia inverse in body frame
                'inertia_tensor_global_frame':
                np.zeros(9 * nb, dtype=float),
                # moment of inertia inverse in body frame
                'inertia_tensor_inverse_global_frame':
                np.zeros(9 * nb, dtype=float),
                # total force at the center of mass
                'force':
                np.zeros(3 * nb, dtype=float),
                # torque about the center of mass
                'torque':
                np.zeros(3 * nb, dtype=float),
                # velocity, acceleration of CM.
                'vcm':
                np.zeros(3 * nb, dtype=float),
                'vcm0':
                np.zeros(3 * nb, dtype=float),
                # angular momentum
                'ang_mom':
                np.zeros(3 * nb, dtype=float),
                'ang_mom0':
                np.zeros(3 * nb, dtype=float),
                # angular velocity in global frame
                'omega':
                np.zeros(3 * nb, dtype=float),
                'omega0':
                np.zeros(3 * nb, dtype=float),
                'nb':
                nb
            }

            for key, elem in consts.items():
                pa.add_constant(key, elem)

            pa.add_constant('min_dem_id', min(pa.dem_id))
            pa.add_constant('max_dem_id', max(pa.dem_id))

            eta = np.zeros(pa.nb[0]*pa.total_no_bodies[0] * 1,
                           dtype=float)
            pa.add_constant('eta', eta)

            pa.add_property(name='dem_id_source', stride=pa.total_no_bodies[0],
                            type='int')

            # compute the properties of the body
            set_total_mass(pa)
            set_center_of_mass(pa)

            # this function will compute
            # inertia_tensor_body_frame
            # inertia_tensor_inverse_body_frame
            # inertia_tensor_global_frame
            # inertia_tensor_inverse_global_frame
            # of the rigid body
            set_moment_of_inertia_and_its_inverse(pa)

            set_body_frame_position_vectors(pa)

            ####################################################
            # compute the boundary particles of the rigid body #
            ####################################################
            add_boundary_identification_properties(pa)
            # make sure your rho is not zero
            equations = get_boundary_identification_etvf_equations([pa.name],
                                                                   [pa.name])
            # print(equations)

            sph_eval = SPHEvaluator(arrays=[pa],
                                    equations=equations,
                                    dim=self.dim,
                                    kernel=QuinticSpline(dim=self.dim))

            sph_eval.evaluate(dt=0.1)

            # make normals of particle other than boundary particle as zero
            # for i in range(len(pa.x)):
            #     if pa.is_boundary[i] == 0:
            #         pa.normal[3 * i] = 0.
            #         pa.normal[3 * i + 1] = 0.
            #         pa.normal[3 * i + 2] = 0.

            # normal vectors in terms of body frame
            set_body_frame_normal_vectors(pa)

            # Adami boundary conditions. SetWallVelocity
            add_properties(pa, 'rho_fsi', 'm_fsi', 'p_fsi')
            add_properties(pa, 'ug', 'vf', 'vg', 'wg', 'uf', 'wf', 'wij')
            pa.add_property('wij')

            pa.set_output_arrays([
                'x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'normal',
                'is_boundary', 'fz', 'm', 'body_id', 'h'
            ])

        for boundary in self.boundaries:
            pa = pas[boundary]
            # Adami boundary conditions. SetWallVelocity
            add_properties(pa, 'ug', 'vf', 'vg', 'wg', 'uf', 'wf', 'wij')
            # pa.add_property('m_fluid')

            ####################################################
            # compute the boundary particles of the rigid body #
            ####################################################
            add_boundary_identification_properties(pa)
            # make sure your rho is not zero
            equations = get_boundary_identification_etvf_equations([pa.name],
                                                                   [pa.name])

            sph_eval = SPHEvaluator(arrays=[pa],
                                    equations=equations,
                                    dim=self.dim,
                                    kernel=QuinticSpline(dim=self.dim))

            sph_eval.evaluate(dt=0.1)

        for fluid in self.fluids:
            pa = pas[fluid]

            add_properties(pa, 'rho0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0',
                           'arho', 'vol', 'cs', 'ap')

            if 'c0_ref' not in pa.constants:
                pa.add_constant('c0_ref', self.c0)

            pa.vol[:] = pa.m[:] / pa.rho[:]

            pa.cs[:] = self.c0
            pa.add_output_arrays(['p'])

    def _set_particle_velocities(self, pa):
        for i in range(max(pa.body_id) + 1):
            fltr = np.where(pa.body_id == i)
            bid = i
            i9 = 9 * bid
            i3 = 3 * bid

            for j in fltr:
                dx = (pa.R[i9 + 0] * pa.dx0[j] + pa.R[i9 + 1] * pa.dy0[j] +
                      pa.R[i9 + 2] * pa.dz0[j])
                dy = (pa.R[i9 + 3] * pa.dx0[j] + pa.R[i9 + 4] * pa.dy0[j] +
                      pa.R[i9 + 5] * pa.dz0[j])
                dz = (pa.R[i9 + 6] * pa.dx0[j] + pa.R[i9 + 7] * pa.dy0[j] +
                      pa.R[i9 + 8] * pa.dz0[j])

                du = pa.omega[i3 + 1] * dz - pa.omega[i3 + 2] * dy
                dv = pa.omega[i3 + 2] * dx - pa.omega[i3 + 0] * dz
                dw = pa.omega[i3 + 0] * dy - pa.omega[i3 + 1] * dx

                pa.u[j] = pa.vcm[i3 + 0] + du
                pa.v[j] = pa.vcm[i3 + 1] + dv
                pa.w[j] = pa.vcm[i3 + 2] + dw

    def set_linear_velocity(self, pa, linear_vel):
        pa.vcm[:] = linear_vel

        self._set_particle_velocities(pa)

    def set_angular_velocity(self, pa, angular_vel):
        pa.omega[:] = angular_vel[:]

        # set the angular momentum
        for i in range(max(pa.body_id) + 1):
            i9 = 9 * i
            i3 = 3 * i
            pa.ang_mom[i3:i3 + 3] = np.matmul(
                pa.inertia_tensor_global_frame[i9:i9 + 9].reshape(3, 3),
                pa.omega[i3:i3 + 3])[:]

        self._set_particle_velocities(pa)

    def _get_edac_nu(self):
        if self.art_nu > 0:
            nu = self.art_nu
            print(self.art_nu)
            print("Using artificial viscosity for EDAC with nu = %s" % nu)
        else:
            nu = self.nu
            print("Using real viscosity for EDAC with nu = %s" % self.nu)
        return nu

    def get_solver(self):
        return self.solver
