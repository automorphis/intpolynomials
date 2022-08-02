
from intpolynomials.intpolynomials import Int_Polynomial, Int_Polynomial_Array, zero_poly
from intpolynomials.registers import Int_Polynomial_Register

def get_include():

    from pathlib import Path
    return Path(__file__).parent / "intpolynomials.pxd"

