"""Papers for reference:

1. Smoothed particle hydrodynamics modeling of granular column collapse
https://doi.org/10.1007/s10035-016-0684-3 for the benchmarks (2d column
collapse)

"""
import numpy as np

from pysph.sph.scheme import add_bool_argument

from pysph.sph.equation import Equation
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.integrator import EPECIntegrator
from pysph.base.kernels import (CubicSpline, WendlandQuintic, QuinticSpline)
from textwrap import dedent
from compyle.api import declare

from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep

from pysph.base.kernels import (CubicSpline, WendlandQuintic, QuinticSpline,
                                WendlandQuinticC4, Gaussian, SuperGaussian)

from rigid_body_common import (set_total_mass, set_center_of_mass,
                               set_moment_of_inertia_izz,
                               set_body_frame_position_vectors, BodyForce,
                               SumUpExternalForces)
from numpy import (sqrt, log)


class LVCDisplacement(Equation):
    """
    linearViscoelasticContactModelWithCoulombFriction

    From Luding 2008 paper
    """
    def __init__(self, dest, sources):
        super(LVCDisplacement, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_u, d_v, d_w, d_wx, d_wy, d_wz, d_fx, d_fy,
             d_fz, d_tng_idx, d_tng_idx_dem_id, d_tng_x, d_tng_y, d_tng_z,
             d_total_tng_contacts, d_dem_id, d_max_tng_contacts_limit, d_torx,
             d_tory, d_torz, XIJ, RIJ, d_rad_s, d_kn, d_kt, d_alpha, d_mu,
             s_idx, s_m, s_u, s_v, s_w, s_wx, s_wy, s_wz, s_rad_s, s_dem_id,
             dt, t):
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
        overlap = -1.

        # check the particles are not on top of each other.
        if RIJ > 0:
            overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

        # ---------- force computation starts ------------
        # if particles are overlapping
        if overlap > 0:
            # normal vector (nij) passing from d_idx to s_idx, i.e., i to j
            rinv = 1.0 / RIJ
            # in PySPH: XIJ[0] = d_x[d_idx] - s_x[s_idx]
            nx = XIJ[0] * rinv
            ny = XIJ[1] * rinv
            nz = XIJ[2] * rinv

            # ---- Relative velocity computation (Eq 2.9) ----
            # relative velocity of particle d_idx w.r.t particle s_idx at
            # contact point. The velocity difference provided by PySPH is
            # only between translational velocities, but we need to
            # consider rotational velocities also.
            # Distance till contact point
            a_i = d_rad_s[d_idx] - overlap / 2.
            a_j = s_rad_s[s_idx] - overlap / 2.

            # velocity of particle i at the contact point
            vi_x = d_u[d_idx] + (d_wy[d_idx] * nz - d_wz[d_idx] * ny) * a_i
            vi_y = d_v[d_idx] + (d_wz[d_idx] * nx - d_wx[d_idx] * nz) * a_i
            vi_z = d_w[d_idx] + (d_wx[d_idx] * ny - d_wy[d_idx] * nx) * a_i

            # just flip the normal and compute the angular velocity
            # contribution
            vj_x = s_u[s_idx] + (-s_wy[s_idx] * nz + s_wz[s_idx] * ny) * a_j
            vj_y = s_v[s_idx] + (-s_wz[s_idx] * nx + s_wx[s_idx] * nz) * a_j
            vj_z = s_w[s_idx] + (-s_wx[s_idx] * ny + s_wy[s_idx] * nx) * a_j

            # Now the relative velocity of particle i w.r.t j at the contact
            # point is
            vij_x = vi_x - vj_x
            vij_y = vi_y - vj_y
            vij_z = vi_z - vj_z

            # normal velocity magnitude
            vij_dot_nij = vij_x * nx + vij_y * ny + vij_z * nz
            vn_x = vij_dot_nij * nx
            vn_y = vij_dot_nij * ny
            vn_z = vij_dot_nij * nz

            # tangential velocity
            vt_x = vij_x - vn_x
            vt_y = vij_y - vn_y
            vt_z = vij_z - vn_z
            # magnitude of the tangential velocity
            # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

            m_eff = d_m[d_idx] * s_m[s_idx] / (d_m[d_idx] + s_m[s_idx])
            eta_n = d_alpha[s_dem_id[s_idx]] * sqrt(m_eff)

            ############################
            # normal force computation #
            ############################
            fn = d_kn[s_dem_id[s_idx]] * overlap + eta_n * - vij_dot_nij
            fn_x = fn * nx
            fn_y = fn * ny
            fn_z = fn * nz

            #################################
            # tangential force computation  #
            #################################
            # total number of contacts of particle i in destination
            tot_ctcs = d_total_tng_contacts[d_idx]

            # d_idx has a range of tracking indices with sources
            # starting index is p
            p = d_idx * d_max_tng_contacts_limit[0]
            # ending index is q -1
            q1 = p + tot_ctcs

            # check if the particle is in the tracking list
            # if so, then save the location at found_at
            found = 0
            for j in range(p, q1):
                if s_idx == d_tng_idx[j]:
                    if s_dem_id[s_idx] == d_tng_idx_dem_id[j]:
                        found_at = j
                        found = 1
                        break
            # if the particle is not been tracked then assign an index in
            # tracking history.
            ft_x = 0.
            ft_y = 0.
            ft_z = 0.

            if found == 0:
                found_at = q1
                d_tng_idx[found_at] = s_idx
                d_total_tng_contacts[d_idx] += 1
                d_tng_idx_dem_id[found_at] = s_dem_id[s_idx]

            # implies we are tracking the particle
            else:
                ####################################################
                # rotate the tangential force to the current plane #
                ####################################################
                # project tangential spring on on the current plane normal
                tng_dot_nij = (d_tng_x[found_at] * nx +
                               d_tng_y[found_at] * ny + d_tng_z[found_at] * nz)

                d_tng_x[found_at] = d_tng_x[found_at] - tng_dot_nij * nx
                d_tng_y[found_at] = d_tng_y[found_at] - tng_dot_nij * ny
                d_tng_z[found_at] = d_tng_z[found_at] - tng_dot_nij * nz

                # compute the tangential force
                kt = d_kt[s_dem_id[s_idx]]
                kt_1 = 1. / kt
                ft_x = -kt * d_tng_x[found_at] - eta_n * vt_x
                ft_y = -kt * d_tng_y[found_at] - eta_n * vt_y
                ft_z = -kt * d_tng_z[found_at] - eta_n * vt_z

                ft_magn = (ft_x*ft_x + ft_y*ft_y + ft_z*ft_z)**0.5

                tx = 0.
                ty = 0.
                tz = 0.
                if ft_magn > 1e-12:
                    tx = ft_x / ft_magn
                    ty = ft_y / ft_magn
                    tz = ft_z / ft_magn

                # adjust for Colomb friction
                mu = d_mu[s_dem_id[s_idx]]
                fn_mu = mu * fn
                if ft_magn > fn_mu:
                    # adjust the tangential force
                    ft_x = fn_mu * tx
                    ft_y = fn_mu * ty
                    ft_z = fn_mu * tz

                    # and adjust the spring length
                    d_tng_x[found_at] = - kt_1 * (fn_mu * tx + eta_n * vt_x)
                    d_tng_y[found_at] = - kt_1 * (fn_mu * ty + eta_n * vt_y)
                    d_tng_z[found_at] = - kt_1 * (fn_mu * tz + eta_n * vt_z)
                else:
                    d_tng_x[found_at] += vt_x * dt
                    d_tng_y[found_at] += vt_y * dt
                    d_tng_z[found_at] += vt_z * dt

            d_fx[d_idx] += fn_x + ft_x
            d_fy[d_idx] += fn_y + ft_y
            d_fz[d_idx] += fn_z + ft_z

            # torque = n cross F
            d_torx[d_idx] += (ny * ft_z - nz * ft_y) * a_i
            d_tory[d_idx] += (nz * ft_x - nx * ft_z) * a_i
            d_torz[d_idx] += (nx * ft_y - ny * ft_x) * a_i


class UpdateTangentialContactsLVCDisplacement(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_z, d_rad_s,
                        d_total_tng_contacts, d_tng_idx,
                        d_max_tng_contacts_limit, d_tng_x, d_tng_y, d_tng_z,
                        d_tng_idx_dem_id, s_x, s_y, s_z, s_rad_s, s_dem_id):
        p = declare('int')
        count = declare('int')
        k = declare('int')
        xij = declare('matrix(3)')
        last_idx_tmp = declare('int')
        sidx = declare('int')
        dem_id = declare('int')
        rij = 0.0

        idx_total_ctcs = declare('int')
        idx_total_ctcs = d_total_tng_contacts[d_idx]
        # particle idx contacts has range of indices
        # and the first index would be
        p = d_idx * d_max_tng_contacts_limit[0]
        last_idx_tmp = p + idx_total_ctcs - 1
        k = p
        count = 0

        # loop over all the contacts of particle d_idx
        while count < idx_total_ctcs:
            # The index of the particle with which
            # d_idx in contact is
            sidx = d_tng_idx[k]
            # get the dem id of the particle
            dem_id = d_tng_idx_dem_id[k]

            if sidx == -1:
                break
            else:
                if dem_id == s_dem_id[sidx]:
                    xij[0] = d_x[d_idx] - s_x[sidx]
                    xij[1] = d_y[d_idx] - s_y[sidx]
                    xij[2] = d_z[d_idx] - s_z[sidx]
                    rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] +
                               xij[2] * xij[2])

                    overlap = d_rad_s[d_idx] + s_rad_s[sidx] - rij

                    if overlap <= 0.:
                        # if the swap index is the current index then
                        # simply make it to null contact.
                        if k == last_idx_tmp:
                            d_tng_idx[k] = -1
                            d_tng_idx_dem_id[k] = -1
                            d_tng_x[k] = 0.
                            d_tng_y[k] = 0.
                            d_tng_z[k] = 0.
                        else:
                            # swap the current tracking index with the final
                            # contact index
                            d_tng_idx[k] = d_tng_idx[last_idx_tmp]
                            d_tng_idx[last_idx_tmp] = -1

                            # swap tangential x displacement
                            d_tng_x[k] = d_tng_x[last_idx_tmp]
                            d_tng_x[last_idx_tmp] = 0.

                            # swap tangential y displacement
                            d_tng_y[k] = d_tng_y[last_idx_tmp]
                            d_tng_y[last_idx_tmp] = 0.

                            # swap tangential z displacement
                            d_tng_z[k] = d_tng_z[last_idx_tmp]
                            d_tng_z[last_idx_tmp] = 0.

                            # swap tangential idx dem id
                            d_tng_idx_dem_id[k] = d_tng_idx_dem_id[
                                last_idx_tmp]
                            d_tng_idx_dem_id[last_idx_tmp] = -1

                            # decrease the last_idx_tmp, since we swapped it to
                            # -1
                            last_idx_tmp -= 1

                        # decrement the total contacts of the particle
                        d_total_tng_contacts[d_idx] -= 1
                    else:
                        k = k + 1
                else:
                    k = k + 1
                count += 1


class LVCForce(Equation):
    """
    linearViscoelasticContactModelWithCoulombFriction
    """
    def __init__(self, dest, sources, kn=1e8, mu=0.5, en=0.8):
        self.kn = kn
        self.kt = 2. / 7. * kn
        self.kt_1 = 1. / self.kt
        self.en = en
        self.et = 0.5 * self.en
        self.mu = mu
        tmp = log(en)
        self.alpha = 2. * sqrt(kn) * abs(tmp) / (sqrt(np.pi**2. + tmp**2.))
        super(LVCForce, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_u, d_v, d_w, d_wx, d_wy, d_wz, d_fx, d_fy,
             d_fz, d_tng_idx, d_tng_idx_dem_id, d_tng_fx, d_tng_fy, d_tng_fz,
             d_total_tng_contacts, d_dem_id, d_max_tng_contacts_limit, d_torx,
             d_tory, d_torz, XIJ, RIJ, d_rad_s, s_idx, s_m, s_u, s_v, s_w,
             s_wx, s_wy, s_wz, s_rad_s, s_dem_id, dt, t):
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
        overlap = -1.

        # check the particles are not on top of each other.
        if RIJ > 0:
            overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

        # ---------- force computation starts ------------
        # if particles are overlapping
        if overlap > 0:
            # normal vector (nij) passing from d_idx to s_idx, i.e., i to j
            rinv = 1.0 / RIJ
            # in PySPH: XIJ[0] = d_x[d_idx] - s_x[s_idx]
            nx = XIJ[0] * rinv
            ny = XIJ[1] * rinv
            nz = XIJ[2] * rinv

            # ---- Relative velocity computation (Eq 2.9) ----
            # relative velocity of particle d_idx w.r.t particle s_idx at
            # contact point. The velocity difference provided by PySPH is
            # only between translational velocities, but we need to
            # consider rotational velocities also.
            # Distance till contact point
            a_i = d_rad_s[d_idx] - overlap / 2.
            a_j = s_rad_s[s_idx] - overlap / 2.

            # velocity of particle i at the contact point
            vi_x = d_u[d_idx] + (d_wy[d_idx] * nz - d_wz[d_idx] * ny) * a_i
            vi_y = d_v[d_idx] + (d_wz[d_idx] * nx - d_wx[d_idx] * nz) * a_i
            vi_z = d_w[d_idx] + (d_wx[d_idx] * ny - d_wy[d_idx] * nx) * a_i

            # just flip the normal and compute the angular velocity
            # contribution
            vj_x = s_u[s_idx] + (-s_wy[s_idx] * nz + s_wz[s_idx] * ny) * a_j
            vj_y = s_v[s_idx] + (-s_wz[s_idx] * nx + s_wx[s_idx] * nz) * a_j
            vj_z = s_w[s_idx] + (-s_wx[s_idx] * ny + s_wy[s_idx] * nx) * a_j

            # Now the relative velocity of particle j w.r.t i at the contact
            # point is
            vr_x = vj_x - vi_x
            vr_y = vj_y - vi_y
            vr_z = vj_z - vi_z

            # normal velocity magnitude
            vr_dot_nij = vr_x * nx + vr_y * ny + vr_z * nz
            vn_x = vr_dot_nij * nx
            vn_y = vr_dot_nij * ny
            vn_z = vr_dot_nij * nz

            # tangential velocity
            vt_x = vr_x - vn_x
            vt_y = vr_y - vn_y
            vt_z = vr_z - vn_z
            # magnitude of the tangential velocity
            # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

            m_eff = d_m[d_idx] * s_m[s_idx] / (d_m[d_idx] + s_m[s_idx])
            eta_n = self.alpha * sqrt(m_eff)

            ############################
            # normal force computation #
            ############################
            kn_overlap = self.kn * overlap
            fn_x = -kn_overlap * nx - eta_n * vn_x
            fn_y = -kn_overlap * ny - eta_n * vn_y
            fn_z = -kn_overlap * nz - eta_n * vn_z

            #################################
            # tangential force computation  #
            #################################
            # total number of contacts of particle i in destination
            tot_ctcs = d_total_tng_contacts[d_idx]

            # d_idx has a range of tracking indices with sources
            # starting index is p
            p = d_idx * d_max_tng_contacts_limit[0]
            # ending index is q -1
            q1 = p + tot_ctcs

            # check if the particle is in the tracking list
            # if so, then save the location at found_at
            found = 0
            for j in range(p, q1):
                if s_idx == d_tng_idx[j]:
                    if s_dem_id[s_idx] == d_tng_idx_dem_id[j]:
                        found_at = j
                        found = 1
                        break
            # if the particle is not been tracked then assign an index in
            # tracking history.
            # ft_x = 0.
            # ft_y = 0.
            # ft_z = 0.

            if found == 0:
                found_at = q1
                d_tng_idx[found_at] = s_idx
                d_total_tng_contacts[d_idx] += 1
                d_tng_idx_dem_id[found_at] = s_dem_id[s_idx]

            # # implies we are tracking the particle
            # else:
            #     ####################################################
            #     # rotate the tangential force to the current plane #
            #     ####################################################
            #     ft_magn = (d_tng_fx[found_at]**2. + d_tng_fy[found_at]**2. +
            #                d_tng_fz[found_at]**2.)**0.5
            #     ft_dot_nij = (d_tng_fx[found_at] * nx +
            #                   d_tng_fy[found_at] * ny +
            #                   d_tng_fz[found_at] * nz)
            #     # tangential force projected onto the current normal of the
            #     # contact place
            #     ft_px = d_tng_fx[found_at] - ft_dot_nij * nx
            #     ft_py = d_tng_fy[found_at] - ft_dot_nij * ny
            #     ft_pz = d_tng_fz[found_at] - ft_dot_nij * nz

            #     ftp_magn = (ft_px**2. + ft_py**2. + ft_pz**2.)**0.5
            #     if ftp_magn > 0:
            #         one_by_ftp_magn = 1. / ftp_magn

            #         tx = ft_px * one_by_ftp_magn
            #         ty = ft_py * one_by_ftp_magn
            #         tz = ft_pz * one_by_ftp_magn
            #     else:
            #         if vt_magn > 0.:
            #             tx = -vt_x / vt_magn
            #             ty = -vt_y / vt_magn
            #             tz = -vt_z / vt_magn
            #         else:
            #             tx = 0.
            #             ty = 0.
            #             tz = 0.

            #     # rescale the projection by the magnitude of the
            #     # previous tangential force, which gives the tangential
            #     # force on the current plane
            #     ft_x = ft_magn * tx
            #     ft_y = ft_magn * ty
            #     ft_z = ft_magn * tz

            #     # (*) check against Coulomb criterion
            #     # Tangential force magnitude due to displacement
            #     ftr_magn = (ft_x * ft_x + ft_y * ft_y + ft_z * ft_z)**(0.5)
            #     fn_magn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)

            #     # we have to compare with static friction, so
            #     # this mu has to be static friction coefficient
            #     fn_mu = self.mu * fn_magn

            #     if ftr_magn >= fn_mu:
            #         # rescale the tangential displacement
            #         d_tng_fx[found_at] = fn_mu * tx
            #         d_tng_fy[found_at] = fn_mu * ty
            #         d_tng_fz[found_at] = fn_mu * tz

            #         # set the tangential force to static friction
            #         # from Coulomb criterion
            #         ft_x = fn_mu * tx
            #         ft_y = fn_mu * ty
            #         ft_z = fn_mu * tz

            d_tng_fx[found_at] -= self.kt * vt_x * dt
            d_tng_fy[found_at] -= self.kt * vt_y * dt
            d_tng_fz[found_at] -= self.kt * vt_z * dt

            # check for Coloumb friction
            fn_magn = (fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)**(0.5)
            fn_mu = self.mu * fn_magn

            ft_magn = (d_tng_fx[found_at] * d_tng_fx[found_at] +
                       d_tng_fy[found_at] * d_tng_fy[found_at] +
                       d_tng_fz[found_at] * d_tng_fz[found_at])

            if ft_magn >= fn_magn:
                d_tng_fx[found_at] = fn_mu * d_tng_fx[found_at] / ft_magn
                d_tng_fy[found_at] = fn_mu * d_tng_fy[found_at] / ft_magn
                d_tng_fz[found_at] = fn_mu * d_tng_fz[found_at] / ft_magn

            d_fx[d_idx] += fn_x + d_tng_fx[found_at]
            d_fy[d_idx] += fn_y + d_tng_fy[found_at]
            d_fz[d_idx] += fn_z + d_tng_fz[found_at]

            # torque = n cross F
            d_torx[d_idx] += (ny * d_tng_fz[found_at] -
                              nz * d_tng_fy[found_at]) * a_i
            d_tory[d_idx] += (nz * d_tng_fx[found_at] -
                              nx * d_tng_fz[found_at]) * a_i
            d_torz[d_idx] += (nx * d_tng_fy[found_at] -
                              ny * d_tng_fx[found_at]) * a_i


class UpdateTangentialContactsLVCForce(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_z, d_rad_s,
                        d_total_tng_contacts, d_tng_idx,
                        d_max_tng_contacts_limit, d_tng_fx, d_tng_fy, d_tng_fz,
                        d_tng_idx_dem_id, s_x, s_y, s_z, s_rad_s, s_dem_id):
        p = declare('int')
        count = declare('int')
        k = declare('int')
        xij = declare('matrix(3)')
        last_idx_tmp = declare('int')
        sidx = declare('int')
        dem_id = declare('int')
        rij = 0.0

        idx_total_ctcs = declare('int')
        idx_total_ctcs = d_total_tng_contacts[d_idx]
        # particle idx contacts has range of indices
        # and the first index would be
        p = d_idx * d_max_tng_contacts_limit[0]
        last_idx_tmp = p + idx_total_ctcs - 1
        k = p
        count = 0

        # loop over all the contacts of particle d_idx
        while count < idx_total_ctcs:
            # The index of the particle with which
            # d_idx in contact is
            sidx = d_tng_idx[k]
            # get the dem id of the particle
            dem_id = d_tng_idx_dem_id[k]

            if sidx == -1:
                break
            else:
                if dem_id == s_dem_id[sidx]:
                    xij[0] = d_x[d_idx] - s_x[sidx]
                    xij[1] = d_y[d_idx] - s_y[sidx]
                    xij[2] = d_z[d_idx] - s_z[sidx]
                    rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] +
                               xij[2] * xij[2])

                    overlap = d_rad_s[d_idx] + s_rad_s[sidx] - rij

                    if overlap <= 0.:
                        # if the swap index is the current index then
                        # simply make it to null contact.
                        if k == last_idx_tmp:
                            d_tng_idx[k] = -1
                            d_tng_idx_dem_id[k] = -1
                            d_tng_fx[k] = 0.
                            d_tng_fy[k] = 0.
                            d_tng_fz[k] = 0.
                        else:
                            # swap the current tracking index with the final
                            # contact index
                            d_tng_idx[k] = d_tng_idx[last_idx_tmp]
                            d_tng_idx[last_idx_tmp] = -1

                            # swap tangential x displacement
                            d_tng_fx[k] = d_tng_fx[last_idx_tmp]
                            d_tng_fx[last_idx_tmp] = 0.

                            # swap tangential y displacement
                            d_tng_fy[k] = d_tng_fy[last_idx_tmp]
                            d_tng_fy[last_idx_tmp] = 0.

                            # swap tangential z displacement
                            d_tng_fz[k] = d_tng_fz[last_idx_tmp]
                            d_tng_fz[last_idx_tmp] = 0.

                            # swap tangential idx dem id
                            d_tng_idx_dem_id[k] = d_tng_idx_dem_id[
                                last_idx_tmp]
                            d_tng_idx_dem_id[last_idx_tmp] = -1

                            # decrease the last_idx_tmp, since we swapped it to
                            # -1
                            last_idx_tmp -= 1

                        # decrement the total contacts of the particle
                        d_total_tng_contacts[d_idx] -= 1
                    else:
                        k = k + 1
                else:
                    k = k + 1
                count += 1


class DEMStep(IntegratorStep):
    def stage1(self, d_idx, d_m, d_moi, d_u, d_v, d_w, d_wx, d_wy, d_wz, d_fx,
               d_fy, d_fz, d_torx, d_tory, d_torz, dt):
        dtb2 = 0.5 * dt
        m_inverse = 1. / d_m[d_idx]
        d_u[d_idx] += dtb2 * d_fx[d_idx] * m_inverse
        d_v[d_idx] += dtb2 * d_fy[d_idx] * m_inverse
        d_w[d_idx] += dtb2 * d_fz[d_idx] * m_inverse

        I_inverse = 1. / d_moi[d_idx]
        d_wx[d_idx] += dtb2 * d_torx[d_idx] * I_inverse
        d_wy[d_idx] += dtb2 * d_tory[d_idx] * I_inverse
        d_wz[d_idx] += dtb2 * d_torz[d_idx] * I_inverse

    def stage2(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z, dt):
        d_x[d_idx] += dt * d_u[d_idx]
        d_y[d_idx] += dt * d_v[d_idx]
        d_z[d_idx] += dt * d_w[d_idx]

    def stage3(self, d_idx, d_m, d_moi, d_u, d_v, d_w, d_wx, d_wy, d_wz, d_fx,
               d_fy, d_fz, d_torx, d_tory, d_torz, dt):
        dtb2 = 0.5 * dt
        m_inverse = 1. / d_m[d_idx]
        d_u[d_idx] += dtb2 * d_fx[d_idx] * m_inverse
        d_v[d_idx] += dtb2 * d_fy[d_idx] * m_inverse
        d_w[d_idx] += dtb2 * d_fz[d_idx] * m_inverse

        I_inverse = 1. / d_moi[d_idx]
        d_wx[d_idx] += dtb2 * d_torx[d_idx] * I_inverse
        d_wy[d_idx] += dtb2 * d_tory[d_idx] * I_inverse
        d_wz[d_idx] += dtb2 * d_torz[d_idx] * I_inverse


class DEMScheme(Scheme):
    def __init__(self,
                 granular_particles,
                 boundaries,
                 kn=1e5,
                 en=0.5,
                 integrator="gtvf",
                 dim=2,
                 gx=0.0,
                 gy=0.0,
                 gz=0.0,
                 kernel_choice="1",
                 kernel_factor=3,
                 contact_model="LVCDisplacement"):
        self.granular_particles = granular_particles

        if boundaries is None:
            self.boundaries = []
        else:
            self.boundaries = boundaries

        # assert(dim, 2)

        self.dim = dim

        self.kernel = CubicSpline

        self.integrator = integrator

        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.kn = kn
        self.en = en

        self.contact_model = contact_model

        self.solver = None

    def add_user_options(self, group):
        # add_bool_argument(
        #     group,
        #     'shear-stress-tvf-correction',
        #     dest='shear_stress_tvf_correction',
        #     default=True,
        #     help='Add the extra shear stress rate term arriving due to TVF')

        # add_bool_argument(group,
        #                   'edac',
        #                   dest='edac',
        #                   default=True,
        #                   help='Use pressure evolution equation EDAC')

        choices = ['LVC']
        group.add_argument("--contact-model",
                           action="store",
                           dest='contact_model',
                           default="LVCDisplacement",
                           choices=choices,
                           help="Specify what contact model to use " % choices)

    def consume_user_options(self, options):
        _vars = ['contact_model']
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def get_equations(self):
        return self._get_gtvf_equations()

    def _get_gtvf_equations(self):
        from pysph.sph.equation import Group, MultiStageEquations
        from pysph.sph.basic_equations import (ContinuityEquation,
                                               MonaghanArtificialViscosity,
                                               VelocityGradient3D,
                                               VelocityGradient2D)
        from pysph.sph.solid_mech.basic import (IsothermalEOS,
                                                HookesDeviatoricStressRate,
                                                MonaghanArtificialStress)
        all = list(set(self.granular_particles + self.boundaries))

        stage1 = []
        g1 = []

        # ------------------------
        # stage 1 equations starts
        # ------------------------
        # There will be no equations for stage 1

        # ------------------------
        # stage 2 equations starts
        # ------------------------
        stage2 = []
        g1 = []

        if self.contact_model == "LVCDisplacement":
            for granules in self.granular_particles:
                g1.append(
                    # see the previous examples and write down the sources
                    UpdateTangentialContactsLVCDisplacement(dest=granules,
                                                            sources=all))
            stage2.append(Group(equations=g1, real=False))
        elif self.contact_model == "LVCDisplacement":
            for granules in self.granular_particles:
                g1.append(
                    # see the previous examples and write down the sources
                    UpdateTangentialContactsLVCForce(dest=granules,
                                                     sources=all))
            stage2.append(Group(equations=g1, real=False))

        g2 = []
        for granules in self.granular_particles:
            g2.append(
                BodyForce(dest=granules,
                          sources=None,
                          gx=self.gx,
                          gy=self.gy,
                          gz=self.gz))

        if self.contact_model == "LVCDisplacement":
            for granules in self.granular_particles:
                g2.append(LVCDisplacement(dest=granules, sources=all))

        elif self.contact_model == "LVCDisplacement":
            for granules in self.granular_particles:
                g2.append(LVCForce(dest=granules, sources=all))

        stage2.append(Group(equations=g2, real=False))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self,
                         kernel=None,
                         integrator_cls=None,
                         extra_steppers=None,
                         **kw):
        from pysph.base.kernels import CubicSpline
        from pysph.sph.wc.gtvf import GTVFIntegrator
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        for granular_particles in self.granular_particles:
            if granular_particles not in steppers:
                steppers[granular_particles] = DEMStep()

        cls = integrator_cls if integrator_cls is not None else GTVFIntegrator
        integrator = cls(**steppers)

        self.solver = Solver(dim=self.dim,
                             integrator=integrator,
                             kernel=kernel,
                             **kw)

    def setup_properties(self, particles, clean=True):
        from pysph.examples.solid_mech.impact import add_properties

        pas = dict([(p.name, p) for p in particles])

        for particles in self.granular_particles:
            pa = pas[particles]

            add_properties(pa, 'fx', 'fy', 'fz', 'torx', 'tory', 'torz', 'wx',
                           'wy', 'wz')

            # create the array to save the tangential interaction particles
            # index and other variables
            # HERE `tng` is tangential
            limit = pa.max_tng_contacts_limit[0]
            pa.add_property('tng_idx', stride=limit, type="int")
            pa.tng_idx[:] = -1
            pa.add_property('tng_idx_dem_id', stride=limit, type="int")
            pa.tng_idx_dem_id[:] = -1

            if self.contact_model == "LVCDisplacement":
                pa.add_property('tng_x', stride=limit)
                pa.add_property('tng_y', stride=limit)
                pa.add_property('tng_z', stride=limit)
                pa.tng_x[:] = 0.
                pa.tng_y[:] = 0.
                pa.tng_z[:] = 0.

            if self.contact_model == "LVCForce":
                pa.add_property('tng_fx', stride=limit)
                pa.add_property('tng_fy', stride=limit)
                pa.add_property('tng_fz', stride=limit)
                pa.tng_fx[:] = 0.
                pa.tng_fy[:] = 0.
                pa.tng_fz[:] = 0.

            pa.add_property('total_tng_contacts', type="int")
            pa.total_tng_contacts[:] = 0

            pa.set_output_arrays(
                ['x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'fz', 'm', 'moi'])

    def get_solver(self):
        return self.solver
