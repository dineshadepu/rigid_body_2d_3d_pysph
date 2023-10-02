from pysph.base.utils import get_particle_array
from rigid_body_common import setup_damping_coefficient
import numpy as np
from math import sin, pi, log, pi

M_PI = pi


def create_particle_array(name, x, y, body_id, dem_id, total_mass, total_no_bodies):
    pa = get_particle_array(x=x, y=y, name=name)

    pa.add_property('body_id', type='int', data=body_id)
    pa.add_property('dem_id', type='int', data=dem_id)
    pa.add_constant('total_no_bodies', [total_no_bodies])
    pa.add_constant('min_dem_id', min(pa.dem_id))
    pa.add_constant('max_dem_id', max(pa.dem_id))
    pa.add_constant('total_mass', total_mass)

    nb = int(np.max(pa.body_id) + 1)
    # print("hereeee")
    # print(nb)
    pa.add_constant('nb', nb)

    eta = np.zeros(pa.nb[0]*pa.total_no_bodies[0] * 1,
                   dtype=float)
    pa.add_constant('eta', eta)

    return pa


def test_single_rigid_body():
    # =========================
    # create the particle array
    # =========================
    x = [1., 2.]
    body_id = np.ones_like(x, dtype=int) * 0
    dem_id = np.ones_like(x, dtype=int) * 0
    total_mass = np.array([2.])
    pa = create_particle_array(x=[1., 2.], y=[0., 0.], body_id=body_id,
                               dem_id=dem_id, total_mass=total_mass,
                               total_no_bodies=1, name="body1")

    coeff_of_rest = np.array([0.8])
    pa.add_constant('coeff_of_rest', coeff_of_rest)
    # =========================
    # create the particle array
    # =========================

    setup_damping_coefficient(pa, [pa], boundaries=[])

    # test the properties

    t1 = 2. * 2.
    t2 = 2. + 2.
    m_star = t1 / t2
    t1 = log(0.8)
    t2 = t1**2. + M_PI**2.
    tmp = (m_star / t2)**0.5
    expected = -2. * t1 * tmp
    np.testing.assert_array_almost_equal(pa.eta, [expected])


def test_single_particle_array_with_2_rigid_bodies():
    x = [1., 2., 3., 4.]
    body_id = [0, 0, 1, 1]
    dem_id = [0, 0, 1, 1]
    total_mass = np.array([2., 2.])
    pa = create_particle_array(x=x, y=0, body_id=body_id,
                               dem_id=dem_id, total_mass=total_mass,
                               total_no_bodies=2, name="body1")

    coeff_of_rest = np.array([1., 0.8, 0.8, 1.0])
    pa.add_constant('coeff_of_rest', coeff_of_rest)

    setup_damping_coefficient(pa, [pa], boundaries=[])

    # test the properties
    t1 = 2. * 2.
    t2 = 2. + 2.
    m_star = t1 / t2
    t1 = log(0.8)
    t2 = t1**2. + M_PI**2.
    tmp = (m_star / t2)**0.5
    t3 = -2. * t1 * tmp

    expected = [0., t3, t3, 0.]
    np.testing.assert_array_almost_equal(pa.eta, expected)


def test_single_particle_array_with_2_rigid_bodies_different_mass():
    x = [1., 2., 3., 4.]
    body_id = [0, 0, 1, 1]
    dem_id = [0, 0, 1, 1]
    total_mass = np.array([1., 2.])
    pa = create_particle_array(x=x, y=0, body_id=body_id,
                               dem_id=dem_id, total_mass=total_mass,
                               total_no_bodies=2, name="body1")

    coeff_of_rest = np.array([1., 0.8, 0.8, 1.0])

    pa.add_constant('coeff_of_rest', coeff_of_rest)

    setup_damping_coefficient(pa, [pa], boundaries=[])

    # test the properties
    t1 = 1. * 2.
    t2 = 1. + 2.
    m_star = t1 / t2
    t1 = log(0.8)
    t2 = t1**2. + M_PI**2.
    tmp = (m_star / t2)**0.5
    t3 = -2. * t1 * tmp

    expected = [0., t3, t3, 0.]
    np.testing.assert_array_almost_equal(pa.eta, expected)


def test_single_particle_array_with_5_rigid_bodies():
    x = np.linspace(0., 1., 10)
    body_id = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    dem_id = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    total_mass = np.array([2., 2., 2., 2., 2.])
    pa = create_particle_array(x=x, y=0., body_id=body_id,
                               dem_id=dem_id, total_mass=total_mass,
                               total_no_bodies=5, name="body1")

    coeff_of_rest = np.array([1., 0.8, 0.8, 0.8, 0.8,
                              0.8, 1., 0.8, 0.8, 0.8,
                              0.8, 0.8, 1., 0.8, 0.8,
                              0.8, 0.8, 0.8, 1., 0.8,
                              0.8, 0.8, 0.8, 0.8, 1.])
    pa.add_constant('coeff_of_rest', coeff_of_rest)

    setup_damping_coefficient(pa, [pa], boundaries=[])

    # test the properties
    t1 = 2. * 2.
    t2 = 2. + 2.
    m_star = t1 / t2
    t1 = log(0.8)
    t2 = t1**2. + M_PI**2.
    tmp = (m_star / t2)**0.5
    t3 = -2. * t1 * tmp

    expected = np.array([0., t3, t3, t3, t3,
                         t3, 0., t3, t3, t3,
                         t3, t3, 0., t3, t3,
                         t3, t3, t3, 0., t3,
                         t3, t3, t3, t3, 0.])
    np.testing.assert_array_almost_equal(pa.eta, expected)


def test_two_particle_array_with_1_rigid_body_1_rigid_body():
    x = np.array(1.)
    body_id = [0]
    dem_id = [0]
    total_mass = np.array([2.])
    body1 = create_particle_array(x=x, y=0., body_id=body_id,
                               dem_id=dem_id, total_mass=total_mass,
                               total_no_bodies=2, name="body1")

    coeff_of_rest = np.array([1., 0.8])
    body1.add_constant('coeff_of_rest', coeff_of_rest)

    x = np.array(1.)
    body_id = [0]
    dem_id = [1]
    total_mass = np.array([2.])
    body2 = create_particle_array(x=x, y=0., body_id=body_id,
                               dem_id=dem_id, total_mass=total_mass,
                               total_no_bodies=2, name="body2")

    coeff_of_rest = np.array([0.8, 1.0])
    body2.add_constant('coeff_of_rest', coeff_of_rest)

    setup_damping_coefficient(body1, [body1, body2], boundaries=[])
    setup_damping_coefficient(body2, [body1, body2], boundaries=[])

    # test the properties
    t1 = 2. * 2.
    t2 = 2. + 2.
    m_star = t1 / t2
    t1 = log(0.8)
    t2 = t1**2. + M_PI**2.
    tmp = (m_star / t2)**0.5
    t3 = -2. * t1 * tmp

    expected = np.array([0., t3])
    np.testing.assert_array_almost_equal(body1.eta, expected)

    expected = np.array([t3, 0.])
    np.testing.assert_array_almost_equal(body2.eta, expected)


def test_two_particle_array_with_1_rigid_body_1_boundary():
    x = np.array(1.)
    body_id = [0]
    dem_id = [0]
    total_mass = np.array([2.])
    body1 = create_particle_array(x=x, y=0., body_id=body_id,
                               dem_id=dem_id, total_mass=total_mass,
                               total_no_bodies=2, name="body1")

    coeff_of_rest = np.array([1.0, 0.8])
    body1.add_constant('coeff_of_rest', coeff_of_rest)

    x = np.array(1.)
    body_id = [0]
    dem_id = [1]
    total_mass = np.array([2.])
    body2 = create_particle_array(x=x, y=0., body_id=body_id,
                               dem_id=dem_id, total_mass=total_mass,
                               total_no_bodies=2, name="body2")

    setup_damping_coefficient(body1, [body1], boundaries=[body2])

    # test the properties
    m_star = 2.
    t1 = log(0.8)
    t2 = t1**2. + M_PI**2.
    tmp = (m_star / t2)**0.5
    t3 = -2. * t1 * tmp

    expected = np.array([0., t3])
    np.testing.assert_array_almost_equal(body1.eta, expected)


def test_three_particle_array_with_1_boundary_3_rigid_bodies_1_boundary():
    # create the boundary
    x = np.array(1.)
    body_id = [0]
    dem_id = [0]
    total_mass = np.array([0.])
    boundary_1 = create_particle_array(x=x, y=0., body_id=body_id,
                                       dem_id=dem_id, total_mass=total_mass,
                                       total_no_bodies=5, name="booundary_1")

    # create the rigid bodies
    x = np.linspace(0., 1., 10)
    body_id = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    dem_id = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    total_mass = np.array([2., 2., 2.])
    body1 = create_particle_array(x=x, y=0., body_id=body_id,
                                  dem_id=dem_id, total_mass=total_mass,
                                  total_no_bodies=5, name="body1")

    coeff_of_rest = np.array([0.8, 1., 0.8, 0.8, 0.8,
                              0.8, 0.8, 1.0, 0.8, 0.8,
                              0.8, 0.8, 0.8, 1.0, 0.8])
    body1.add_constant('coeff_of_rest', coeff_of_rest)

    # create the boundary
    x = np.array(1.)
    body_id = [0]
    dem_id = [4]
    total_mass = np.array([0.])
    boundary_2 = create_particle_array(x=x, y=0., body_id=body_id,
                                       dem_id=dem_id, total_mass=total_mass,
                                       total_no_bodies=5, name="booundary_2")

    # setup the damping parameters
    setup_damping_coefficient(body1, [body1], boundaries=[boundary_1, boundary_2])

    # ================================
    # test the properties
    # ================================
    t1 = 2. * 2.
    t2 = 2. + 2.
    m_star = t1 / t2
    t1 = log(0.8)
    t2 = t1**2. + M_PI**2.
    tmp = (m_star / t2)**0.5
    t3 = -2. * t1 * tmp

    # boundary damping parameter
    m_star = 2.
    t1 = log(0.8)
    t2 = t1**2. + M_PI**2.
    tmp = (m_star / t2)**0.5
    b_t3 = -2. * t1 * tmp

    expected = np.array([b_t3, 0., t3, t3, b_t3,
                         b_t3, t3, 0.0, t3, b_t3,
                         b_t3, t3, t3, 0.0, b_t3])

    np.testing.assert_array_almost_equal(body1.eta, expected)


def test_three_particle_array_with_1_boundary_1_rigid_bodies_3_rigid_body_1():
    # create the boundary
    x = np.array(1.)
    body_id = [0]
    dem_id = [0]
    total_mass = np.array([0.])
    boundary_1 = create_particle_array(x=x, y=0., body_id=body_id,
                                       dem_id=dem_id, total_mass=total_mass,
                                       total_no_bodies=5, name="booundary_1")

    # =========================
    # create the rigid bodies
    # =========================
    x = np.linspace(0., 1., 10)
    body_id = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    dem_id = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    total_mass = np.array([2., 2., 2.])
    body1 = create_particle_array(x=x, y=0., body_id=body_id,
                                  dem_id=dem_id, total_mass=total_mass,
                                  total_no_bodies=5, name="body1")

    coeff_of_rest = np.array([0.8, 1., 0.8, 0.8, 0.8,
                              0.8, 0.8, 1.0, 0.8, 0.8,
                              0.8, 0.8, 0.8, 1.0, 0.8])
    body1.add_constant('coeff_of_rest', coeff_of_rest)

    # =========================
    # create the particle array
    # =========================
    x = [1., 2.]
    body_id = np.ones_like(x, dtype=int) * 0
    dem_id = np.ones_like(x, dtype=int) * 4
    total_mass = np.array([2.])
    body2 = create_particle_array(x=[1., 2.], y=[0., 0.], body_id=body_id,
                               dem_id=dem_id, total_mass=total_mass,
                               total_no_bodies=5, name="body2")

    coeff_of_rest = np.array([0.8, 0.8, 0.8, 0.8, 1.0])
    body2.add_constant('coeff_of_rest', coeff_of_rest)
    # =========================
    # create the particle array
    # =========================

    # setup the damping parameters
    setup_damping_coefficient(body1, [body1, body2], boundaries=[boundary_1])
    setup_damping_coefficient(body2, [body1, body2], boundaries=[boundary_1])

    # ================================
    # test the properties
    # ================================
    t1 = 2. * 2.
    t2 = 2. + 2.
    m_star = t1 / t2
    t1 = log(0.8)
    t2 = t1**2. + M_PI**2.
    tmp = (m_star / t2)**0.5
    t3 = -2. * t1 * tmp

    # boundary damping parameter
    m_star = 2.
    t1 = log(0.8)
    t2 = t1**2. + M_PI**2.
    tmp = (m_star / t2)**0.5
    b_t3 = -2. * t1 * tmp

    expected = np.array([b_t3, 0., t3, t3, t3,
                         b_t3, t3, 0.0, t3, t3,
                         b_t3, t3, t3, 0.0, t3])

    np.testing.assert_array_almost_equal(body1.eta, expected)

    # body 2 eta
    expected = np.array([b_t3, t3, t3, t3, 0.])

    np.testing.assert_array_almost_equal(body2.eta, expected)


if __name__ == '__main__':
    # single particle array tests
    test_single_rigid_body()
    test_single_particle_array_with_2_rigid_bodies()
    test_single_particle_array_with_2_rigid_bodies_different_mass()
    test_single_particle_array_with_5_rigid_bodies()

    # two particle array tests
    test_two_particle_array_with_1_rigid_body_1_rigid_body()
    test_two_particle_array_with_1_rigid_body_1_boundary()

    # two particle array tests
    test_three_particle_array_with_1_boundary_3_rigid_bodies_1_boundary()
    test_three_particle_array_with_1_boundary_1_rigid_bodies_3_rigid_body_1()
