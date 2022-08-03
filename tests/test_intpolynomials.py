from unittest import TestCase

from mpmath import mpf, mp, almosteq

from intpolynomials import Int_Polynomial, Int_Polynomial_Array

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

xs = (
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

class Test_Int_Polynomial_Array(TestCase):

    def test_important_stuff(self):

        array = Int_Polynomial_Array(6)
        array.empty(len(polys) * len(xs))
        i = 0

        for x, Ys in xs:

            for (p, d), (y, yp) in zip(polys, Ys):

                poly1 = Int_Polynomial(len(p) - 1)
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




























