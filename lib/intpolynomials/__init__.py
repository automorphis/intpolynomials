
from intpolynomials.intpolynomials import IntPolynomial, IntPolynomialArray, IntPolynomialIter
from intpolynomials.registers import IntPolynomialRegister

def get_include():

    from pathlib import Path
    return Path(__file__).parent / "intpolynomials.pxd"

