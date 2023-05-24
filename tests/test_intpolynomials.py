from unittest import TestCase

import numpy as np
from mpmath import mpf, mp, almosteq, mpc, fabs, workdps, extradps

from intpolynomials import IntPolynomial, IntPolynomialArray
from intpolynomials.intpolynomials import IntPolynomialIter

mp.dps = 28

polys = (
    ((3, -7, 15),            2),
    ((4, 0, -6, 2, 1, 0, 4), 6),
    ((1,),                   0),
    ((1, 0),                 0),
    ((1, 0, 0),              0),
    ((0, 1),                 1),
    ((0, 0, 1),              2),
    ((0,),                  -1),
    ((0, 0),                -1),
    ((0, 0, 0),             -1)
)

roots = (
    ((3, -7, 15),            ((mpc(real='0.233333333333333333333333333333322' , imag='-0.381517438075319901301447809606401'), mpf('0.447213595499957939281834733746247'), 1), (mpc(real='0.233333333333333333333333333333322' , imag='0.381517438075319901301447809606401'), mpf('0.447213595499957939281834733746247'), 1))),
    ((4, 0, -6, 2, 1, 0, 4), ((mpc(real='0.0754273014906132950504682827333017', imag='1.26176387237674875696813711590777'  ), mpf('1.26401635568742709187391777189851' ), 1), (mpc(real='0.0754273014906132950504682827333017', imag='-1.26176387237674875696813711590777'), mpf('1.26401635568742709187391777189851' ), 1), (mpc(real='-0.89113640939210882744009701860007', imag='0.121903354901129783733530969831347'), mpf('0.899435672007960216227049103045863'), 1), (mpc(real='-0.89113640939210882744009701860007', imag='-0.121903354901129783733530969831347'), mpf('0.899435672007960216227049103045863'), 1), (mpc(real='0.815709107901495532389628735866842', imag='0.329068730049742973233002497731467'), mpf('0.879583752584143800163345954721996'), 1), (mpc(real='0.815709107901495532389628735866842', imag='-0.329068730049742973233002497731467'), mpf('0.879583752584143800163345954721996'), 1))),
    ((1,),                   ValueError),
    ((1, 0),                 ValueError),
    ((1, 0, 0),              ValueError),
    ((0, 1),                 ((mpf('0.0'),  mpf('0.0'), 1),)),
    ((0, 0, 1),              ((mpf('0.0'),  mpf('0.0'), 2),)),
    ((0, 0, 0, 1),           ((mpf('0.0'),  mpf('0.0'), 3),)),
    ((1, 6, 12, 8),          ((mpf('-0.5'), mpf('0.5'), 3),)),
    ((1, 3, 3, 1),           ((mpf('-1.0'), mpf('1.0'), 3),)),
    ((0,),                   ValueError),
    ((0, 0),                 ValueError),
    ((0, 0, 0),              ValueError),
    ((2, -2),                ((mpf('1.0'),  mpf('1.0'), 1),)),
    ((1, 0, 0, 1),           ((mpf('-1.0'), mpf('1.0'), 1), (mpc(real='0.5', imag='-0.866025403784438646763723170752918'), mpf('1.0'), 1), (mpc(real='0.5', imag='0.866025403784438646763723170752918'), mpf('1.0'), 1))),
    ((2, 0, 0, 2),           ((mpf('-1.0'), mpf('1.0'), 1), (mpc(real='0.5', imag='-0.866025403784438646763723170752918'), mpf('1.0'), 1), (mpc(real='0.5', imag='0.866025403784438646763723170752918'), mpf('1.0'), 1)))
)

points = (
    (mpf('3.08220700148448822512509619073' ), ((mpf('123.924550989608582424124326664911'), mpf('85.4662100445346467537528857218176' )), (mpf('3525.31193302820527627737682762379'), mpf('6813.19774725399818947421085007337' )), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('3.08220700148448822512509619073' ), mpf('1')), (mpf('9.5'                               ), mpf('6.16441400296897645025019238145424' )), (mpf('0'), mpf('0')), (mpf('0'), mpf('0')), (mpf('0'), mpf('0')))),
    (mpf('-3.08220700148448822512509619073'), ((mpf('167.075449010391417575875673335089'), mpf('-99.4662100445346467537528857218176')), (mpf('3408.18806697179472372262317237621'), mpf('-6699.19774725399818947421085007337')), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('-3.08220700148448822512509619073'), mpf('1')), (mpf('9.5'                               ), mpf('-6.16441400296897645025019238145424')), (mpf('0'), mpf('0')), (mpf('0'), mpf('0')), (mpf('0'), mpf('0')))),
    (mpf('3.14159265358979323846264338328' ), ((mpf('129.05291744121182661327886131518' ), mpf('87.24777960769379715387930149838'   )), (mpf('3949.76079228928367399526379143213'), mpf('7490.01605613141279185915192670849' )), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('3.14159265358979323846264338328' ), mpf('1')), (mpf('9.86960440108935861883449099987572'), mpf('6.28318530717958647692528676655867' )), (mpf('0'), mpf('0')), (mpf('0'), mpf('0')), (mpf('0'), mpf('0')))),
    (mpf('-3.14159265358979323846264338328'), ((mpf('173.035214591468931951755868681091'), mpf('-101.24777960769379715387930149838' )), (mpf('3825.73568556808439329335853116373'), mpf('-7371.58080331834048843313803471039')), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('-3.14159265358979323846264338328'), mpf('1')), (mpf('9.86960440108935861883449099987572'), mpf('-6.28318530717958647692528676655867')), (mpf('0'), mpf('0')), (mpf('0'), mpf('0')), (mpf('0'), mpf('0')))),

    (1,           ((mpf('11'), mpf('23' )), (mpf('5'  ), mpf('22'  )), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('1' ), mpf('1')), (mpf('1'), mpf('2' )), (mpf('0'), mpf('0')), (mpf('0'), mpf('0')), (mpf('0'), mpf('0')))),
    (-1,          ((mpf('25'), mpf('-37')), (mpf('1'  ), mpf('-10' )), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('-1'), mpf('1')), (mpf('1'), mpf('-2')), (mpf('0'), mpf('0')), (mpf('0'), mpf('0')), (mpf('0'), mpf('0')))),
    (0,           ((mpf('3' ), mpf('-7 ')), (mpf('4'  ), mpf('0'   )), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('0' ), mpf('1')), (mpf('0'), mpf('0' )), (mpf('0'), mpf('0')), (mpf('0'), mpf('0')), (mpf('0'), mpf('0')))),
    (2,           ((mpf('49'), mpf('53' )), (mpf('268'), mpf('800' )), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('2' ), mpf('1')), (mpf('4'), mpf('4' )), (mpf('0'), mpf('0')), (mpf('0'), mpf('0')), (mpf('0'), mpf('0')))),
    (-2,          ((mpf('77'), mpf('-67')), (mpf('236'), mpf('-752')), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('1'), mpf('0')), (mpf('-2'), mpf('1')), (mpf('4'), mpf('-4')), (mpf('0'), mpf('0')), (mpf('0'), mpf('0')), (mpf('0'), mpf('0'))))
)

divide = (
    ((((0,), 1), ((0,), 1)), ZeroDivisionError),
    ((((1,), 1), ((0,), 1)), ZeroDivisionError),
    ((((1,), 1), ((0,), 2)), ZeroDivisionError),
    ((((1, 1), 1), ((0,), 1)), ZeroDivisionError),
    ((((1, 1), 1), ((0,), 2)), ZeroDivisionError),
    ((((1,), 1), ((1,), 1)), (((1,), 1), ((0,), 1))),
    ((((1,), 2), ((1,), 1)), (((1,), 2), ((0,), 1))),
    ((((1,), 2), ((1,), 2)), (((1,), 1), ((0,), 1))),
    ((((1,), 2), ((1,), 2)), (((1,), 1), ((0,), 1))),
    ((((1,), 3), ((1,), 2)), (((2,), 3), ((0,), 1))),
    ((((5,), 3), ((1,), 2)), (((10,), 3), ((0,), 1))),
    ((((3,), 3), ((1,), 2)), (((2,), 1), ((0,), 1))),
    ((((3,), 6), ((1,), 2)), (((1,), 1), ((0,), 1))),
    ((((3,), 6), ((1,), 2)), (((1,), 1), ((0,), 1))),
    ((((8,), 6), ((3,), 4)), (((16,), 9), ((0,), 1))),
    ((((0, 1), 1), ((0, 1), 1)), (((1,), 1), ((0,), 1))),
    ((((0, 1), 2), ((0, 1), 1)), (((1,), 2), ((0,), 1))),
    ((((0, 1), 2), ((0, 1), 2)), (((1,), 1), ((0,), 1))),
    ((((0, 1), 2), ((0, 1), 2)), (((1,), 1), ((0,), 1))),
    ((((0, 1), 3), ((0, 1), 2)), (((2,), 3), ((0,), 1))),
    ((((0, 5), 3), ((0, 1), 2)), (((10,), 3), ((0,), 1))),
    ((((0, 3), 3), ((0, 1), 2)), (((2,), 1), ((0,), 1))),
    ((((0, 3), 6), ((0, 1), 2)), (((1,), 1), ((0,), 1))),
    ((((0, 3), 6), ((0, 1), 2)), (((1,), 1), ((0,), 1))),
    ((((0, 8), 6), ((0, 3), 4)), (((16,), 9), ((0,), 1))),
    ((((-7, 0, 0, 3), 2), ((-7, 0, 0, 3), 2)), (((1,), 1), ((0,), 1))),
    ((((-7, 0, 0, 3), 3), ((-7, 0, 0, 3), 2)), (((2,), 3), ((0,), 1))),
    ((((-21, 3, 0, 0, -3, 4), 17), ((-21, 3, 0, 0, -3, 4), 17)), (((1,), 1), ((0,), 1))),
    ((((-21, 3, 0, 0, -3, 4), 17), ((-7, 0, 0, 3), 2)), (((0, -6, 8), 51), ((-63, -12, 28), 51))),
    ((((-7, 0, 0, 3), 2), ((-21, 3, 0, 0, -3, 4), 17)), (((0,), 1), ((-7, 0, 0, 3), 2))),
    ((((0, 11, 0, 0, 0, 0, 0, 1), 3), ((-21, 3, 0, 0, -3, 4), 17)), (((153, 204, 272), 192), ((189, 929, 300, -48, 27), 192))),
    ((((-21, 3, 0, 0, -3, 4), 17), ((0, 11, 0, 0, 0, 0, 0, 1), 3)), (((0,), 1), ((-21, 3, 0, 0, -3, 4), 17))),
    ((((0, 11, 0, 0, 0, 0, 0, 1), 3), ((-7, 0, 0, 3), 2)), (((0, 14, 0, 0, 6), 27), ((0, 148), 27))),
    ((((-7, 0, 0, 3), 2), ((0, 11, 0, 0, 0, 0, 0, 1), 3)), (((0,), 1), ((-7, 0, 0, 3), 2))),
    ((((8, 6, 0, 0, 2), 6), ((-7, 0, 0, 3), 2)), (((0, 2), 9), ((12, 16), 9)))
)

gcds = (
    ((((0,), 1), ((0,), 1)), ZeroDivisionError),
    ((((1,), 1), ((0,), 1)), ((1,), 1)),
    ((((1,), 1), ((0,), 2)), ((1,), 1)),
    ((((1, 1), 1), ((0,), 1)), ((1, 1), 1)),
    ((((1, 1), 1), ((0,), 2)), ((1, 1), 1)),
    ((((1,), 1), ((1,), 1)), ((1,), 1)),
    ((((1,), 2), ((1,), 1)), ((1,), 1)),
    ((((1,), 2), ((1,), 2)), ((1,), 1)),
    ((((1,), 2), ((1,), 2)), ((1,), 1)),
    ((((1,), 3), ((1,), 2)), ((1,), 1)),
    ((((5,), 3), ((1,), 2)), ((1,), 1)),
    ((((3,), 3), ((1,), 2)), ((1,), 1)),
    ((((3,), 6), ((1,), 2)), ((1,), 1)),
    ((((3,), 6), ((1,), 2)), ((1,), 1)),
    ((((8,), 6), ((3,), 4)), ((1,), 1)),
    ((((0, 1), 1), ((0, 1), 1)), ((0, 1), 1)),
    ((((0, 1), 2), ((0, 1), 1)), ((0, 1), 1)),
    ((((0, 1), 2), ((0, 1), 2)), ((0, 1), 1)),
    ((((0, 1), 2), ((0, 1), 2)), ((0, 1), 1)),
    ((((0, 1), 3), ((0, 1), 2)), ((0, 1), 1)),
    ((((0, 5), 3), ((0, 1), 2)), ((0, 1), 1)),
    ((((0, 3), 3), ((0, 1), 2)), ((0, 1), 1)),
    ((((0, 3), 6), ((0, 1), 2)), ((0, 1), 1)),
    ((((0, 3), 6), ((0, 1), 2)), ((0, 1), 1)),
    ((((0, 8), 6), ((0, 3), 4)), ((0, 1), 1)),
    ((((-7, 0, 0, 3), 2), ((-7, 0, 0, 3), 2)), ((-7, 0, 0, 3), 1)),
    ((((-7, 0, 0, 3), 3), ((-7, 0, 0, 3), 2)), ((-7, 0, 0, 3), 1)),
    ((((-21, 3, 0, 0, -3, 4), 17), ((-21, 3, 0, 0, -3, 4), 17)), ((-21, 3, 0, 0, -3, 4), 1)),
    ((((-21, 3, 0, 0, -3, 4), 17), ((-7, 0, 0, 3), 2)), ((1,), 1)),
    ((((-7, 0, 0, 3), 17), ((-21, 3, 0, 0, -3, 4), 17)), ((1,), 1)),
    ((((0, 11, 0, 0, 0, 0, 0, 1), 3), ((-21, 3, 0, 0, -3, 4), 17)), ((1,), 1)),
    ((((-21, 3, 0, 0, -3, 4), 17), ((0, 11, 0, 0, 0, 0, 0, 1), 3)), ((1,), 1)),
    ((((0, 11, 0, 0, 0, 0, 0, 1), 3), ((-7, 0, 0, 3), 2)), ((1,), 1)),
    ((((-7, 0, 0, 3), 2), ((0, 11, 0, 0, 0, 0, 0, 1), 3)), ((1,), 1)),
    ((((8, 6, 0, 0, 2), 6), ((-7, 0, 0, 3), 2)), ((1,), 1)),
    ((((1, 1, 1, 0, 0, 1, 1, 1), 1), ((1, 1, 1, 0, 0, 0, 1, 1, 1), 1)), ((1, 1, 1), 1)),
    ((((2, 2, -2, -2, -2, 2), 27), ((-1, 0, 0, -2, 0, 0, 3), 32)), ((-1, 0, 0, 1), 1))
)

class Test_IntPolynomialArray(TestCase):

    def test_eval(self):

        array = IntPolynomialArray(6)
        array.empty(len(polys) * len(points))
        i = 0

        for x, Ys in points:

            for (p, d), (y, yp) in zip(polys, Ys):

                poly1 = IntPolynomial(len(p) - 1)
                poly1.set(p)

                self.assertTrue(almosteq(
                    y,
                    poly1(x)
                ))

                y2, yp2 = poly1(x, calc_deriv=True)

                self.assertTrue(almosteq(
                    y,
                    y2
                ))

                self.assertTrue(almosteq(
                    yp,
                    yp2
                ))

                self.assertEqual(
                    d,
                    poly1.deg()
                )

                array.append(poly1)
                i += 1

                self.assertEqual(
                    i,
                    len(array)
                )

                poly2 = array[i-1]

                try:
                    self.assertTrue(almosteq(
                        y,
                        poly2(x)
                    ))

                except AssertionError:
                    raise

                y2, yp2 = poly2(x, calc_deriv=True)

                self.assertTrue(almosteq(
                    y,
                    y2
                ))

                self.assertTrue(almosteq(
                    yp,
                    yp2
                ))

                self.assertEqual(
                    d,
                    poly2.deg()
                )

                try:
                    self.assertEqual(
                        poly1,
                        poly2
                    )

                except Exception:
                    raise

    def test_roots(self):

        array = IntPolynomialArray(6)
        array.empty(len(roots))
        i = 0

        for poly, rts in roots:

            p = IntPolynomial(len(poly) - 1)
            p.set(poly)

            for _ in range(2):

                if isinstance(rts, type):

                    try:
                        with self.assertRaises(rts):
                            roots_and_roots_abs = p.roots(ret_abs=True)

                    except AssertionError:
                        print(poly, rts)
                        print(roots_and_roots_abs)
                        raise

                else:

                    roots_and_roots_abs_and_mult = p.roots(ret_abs=True)

                    try:
                        self.assertEqual(
                            p.deg(),
                            sum(m for _, _, m in roots_and_roots_abs_and_mult)
                        )

                    except:
                        raise

                    indices = []

                    for r, ra, m in roots_and_roots_abs_and_mult:

                        self.assertTrue(almosteq(
                            fabs(r),
                            ra
                        ))

                        with extradps(p.deg()):
                            y = p(r)

                        try:
                            with extradps(-1):
                                self.assertTrue(almosteq(
                                    mpf('0.0'),
                                    fabs(y)
                                ))

                        except AssertionError:

                            # while not almosteq(0.0, fabs(p(r))):
                            #     mp.dps -= 1
                            print(fabs(y), mp.dps, str(p), p.deg())
                            raise


                        for n, (r2, ra2, m2) in enumerate(rts):

                            if almosteq(r2, r) and almosteq(ra2, ra) and m == m2 and n not in indices:

                                indices.append(n)
                                break

                        else:
                            self.fail()

                if _ == 0:

                    array.append(p)
                    i += 1
                    p = array[i-1]

        max_deg = 4
        max_sum_abs_coef = 11

        for deg in range(1, max_deg + 1):

            for sum_abs_coef in range(1, max_sum_abs_coef):

                for p in IntPolynomialIter(deg, sum_abs_coef, False):

                    try:
                        rts = p.roots(True)

                    except mp.NoConvergence:
                        self.fail()

                    else:

                        for r, r_abs, m in rts:

                            self.assertTrue(almosteq(
                                fabs(r),
                                r_abs
                            ))

                            with extradps(p.deg()):
                                y = p(r)

                            try:
                                with extradps(-1):
                                    self.assertTrue(almosteq(
                                        0.,
                                        fabs(y)
                                    ))

                            except:
                                print(r, p, p.deg(), mp.dps, fabs(y))
                                raise

                            pp = p

                            for d in range(1, m):

                                pp = pp.deriv()

                                with extradps(pp.deg()):
                                    y = pp(r)

                                self.assertTrue(almosteq(
                                    0.0,
                                    fabs(y)
                                ))


    def test_divide(self):

        for ((top, top_lcd), (bot, bot_lcd)), ans in divide:

            top = IntPolynomial(len(top) - 1).set(top)
            top.set_lcd(top_lcd)
            bot = IntPolynomial(len(bot) - 1).set(bot)
            bot.set_lcd(bot_lcd)

            if isinstance(ans, type):

                with self.assertRaises(ans):
                    top.divide(bot)

            else:

                (quo, quo_lcd), (rem, rem_lcd) = ans
                quo = IntPolynomial(len(quo) - 1).set(quo)
                quo.set_lcd(quo_lcd)
                rem = IntPolynomial(len(rem) - 1).set(rem)
                rem.set_lcd(rem_lcd)

                try:
                    self.assertEqual(
                        (quo, rem),
                        top.divide(bot)
                    )
                except:
                    raise

    def test_gcd(self):

        for nxt in gcds:

            try:
                ((a, a_lcd), (b, b_lcd)), ans = nxt
            except:
                raise

            a = IntPolynomial(len(a) - 1).set(a)
            a.set_lcd(a_lcd)
            b = IntPolynomial(len(b) - 1).set(b)
            b.set_lcd(b_lcd)

            if isinstance(ans, type):

                try:
                    with self.assertRaises(ans):
                        a.gcd(b)

                except:
                    raise

            else:

                g, g_lcd = ans
                g = IntPolynomial(len(g) - 1).set(g)
                g.set_lcd(g_lcd)

                try:
                    self.assertEqual(
                        g,
                        a.gcd(b)
                    )
                except:
                    raise

partitions = (
    ((0, 0, True),  None),
    ((0, 0, False), None),
    ((0, 1, True),  ((1,),)),
    ((0, 1, False), ((1,), (-1,))),
    ((0, 2, True),  None),
    ((0, 2, False), ((2,), (-2,))),
    ((0, 3, True),  None),
    ((0, 3, False), ((3,), (-3,))),
    ((1, 0, True),  None),
    ((1, 0, False), None),
    ((1, 1, True),  ((0, 1),)),
    ((1, 1, False), ((0, 1), (0, -1))),
    ((1, 2, True),  ((1, 1), (-1, 1))),
    ((1, 2, False), ((1, 1), (-1, 1), (1, -1), (-1, -1), (0, 2), (0, -2))),
    ((1, 3, True),  ((2, 1), (-2, 1))),
    ((1, 3, False), ((2, 1), (-2, 1), (2, -1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2), (0, 3), (0, -3))),
    ((2, 0, True),  None),
    ((2, 0, False), None),
    ((2, 1, True),  ((0, 0, 1),)),
    ((2, 1, False), ((0, 0, 1), (0, 0, -1))),
    ((2, 2, True),  ((1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1))),
    ((2, 2, False), ((1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1), (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1), (0, 0, 2), (0, 0, -2))),
    ((2, 3, True),  ((2, 0, 1), (-2, 0, 1), (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1), (0, 2, 1), (0, -2, 1))),
    ((2, 3, False), ((2, 0, 1), (-2, 0, 1), (2, 0, -1), (-2, 0, -1), (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1), (1, 1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1), (0, 2, 1), (0, -2, 1), (0, 2, -1), (0, -2, -1), (1, 0, 2), (-1, 0, 2), (1, 0, -2), (-1, 0, -2), (0, 1, 2), (0, -1, 2), (0, 1, -2), (0, -1, -2), (0, 0, 3), (0, 0, -3))),
    ((3, 0, True),  None),
    ((3, 0, False), None),
    # ((1, 3), ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0))),
    # ((2, 3), ((2, 0, 0, 0), (1, 1, 0, 0), (0, 2, 0, 0), (1, 0, 1, 0), (0, 0, 2, 0), (1, 0, 0, 1), (0, 0, 0, 2))),
    # ((3, 3), ((4, 0, 0, 0), (3, 1, 0, 0), (2, 2, 0, 0), (1, 3, 0, 0), (0, 4, 0, 0), (3, 0, 1, 0), (2, 1, 1, 0), (1, 2, 1, 0), (0, 3, 1, 0), (2, 0, 2, 0), (1, 1, 2, 0), (0, 2, 2, 0), (1, 0, 3, 0), (0, 1, 3, 0), (0, 0, 4, 0), (3, 0, 0, 1), (2, 1, 0, 1), (1, 2, 0, 1), (0, 3, 0, 1), (2, 0, 1, 1), (1, 1, 1, 1), (0, 2, 1, 1), (1, 0, 2, 1), (0, 1, 2, 1), (0, 0, 3, 1), (2, 0, 0, 2), (1, 1, 0, 2), (0, 2, 0, 2), (1, 0, 1, 2), (0, 1, 1, 2), (0, 0, 2, 2), (1, 0, 0, 3), (0, 1, 0, 3), (0, 0, 1, 3), (0, 0, 0, 4)))
)

class Test_IntPolynomialIter(TestCase):

    def test___next__(self):

        for (deg, sum_abs_coefs, monic), ans in partitions:

            if ans is None:

                with self.assertRaises(ValueError):
                    IntPolynomialIter(deg, sum_abs_coefs, monic)

            else:

                num_calc = 0

                for calc, exp in zip(IntPolynomialIter(deg, sum_abs_coefs, monic), ans):

                    _exp = IntPolynomial(deg)
                    _exp.set(exp)

                    try:

                        num_calc += 1
                        self.assertEqual(
                            _exp,
                            calc
                        )

                    except:
                        raise

                self.assertEqual(
                    len(ans),
                    num_calc
                )

                for i, last_coefs in enumerate(ans):

                    last_coefs = np.array(last_coefs, dtype = np.int64)
                    last_coefs = IntPolynomial(len(last_coefs) - 1).set(last_coefs)
                    num_calc = 0

                    for calc, exp in zip(IntPolynomialIter(deg, sum_abs_coefs, monic, last_coefs), ans[i + 1 : ]):

                        num_calc += 1

                        try:
                            self.assertTrue(np.all(
                                np.array(exp, dtype = np.int64) ==
                                calc.get_ndarray()
                            ))
                        except:
                            raise

                    self.assertEqual(
                        len(ans) - i - 1,
                        num_calc
                    )


















