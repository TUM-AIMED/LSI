
from laplace.curvature import AsdlGGN, BackPackGGN, AsdlHessian, AsdlEF, BackPackEF



def laplace_backed_helper(key):
    if key == "AsdlGGN":
        return AsdlGGN
    elif key == "BackPackGGN":
        return BackPackGGN
    elif key == "AsdlHessian":
        return AsdlHessian
    elif key == "AsdlEF":
        return AsdlEF
    elif key == "BackPackEF":
        return BackPackEF
    else:
        raise Exception(f"Key {key} not a valid laplace backend class (AsdlGGN, BackPackGGN, AsdlHessian)")
    
def representation_helper(key):
    if key == "full":
        return "full"
    elif key == "kron":
        return "kron"
    elif key == "lowrank":
        return "lowrank"
    elif key == "diag":
        return "diag"
    else:
        raise Exception(f"Key {key} not a valid laplace representation (full, kron, lowrank, diag)")