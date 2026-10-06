#!/usr/bin/env python3
"""
Simulatore del Type Checker per le Forme Simboliche e gli Effetti di JANUS.
Dimostra l'unificazione delle forme tensoriali e la composizione degli effetti.
"""
from typing import Dict, List, Tuple, Optional

class TensorType:
    def __init__(self, dtype: str, shape: List[str]):
        self.dtype = dtype
        self.shape = shape # es. ['B', '128', 'D']

    def __repr__(self):
        return f"tens[{self.dtype}, {', '.join(self.shape)}]"

def unify_matmul(lhs: TensorType, rhs: TensorType) -> Tuple[TensorType, Dict[str, str]]:
    """
    Unifica lhs [..., M, K] e rhs [..., K, N] producendo [..., M, N].
    Risolve le variabili simboliche.
    """
    if len(lhs.shape) < 2 or len(rhs.shape) < 2:
        raise ValueError(f"Matmul richiede tensori di rango >= 2, ottenuti {lhs} e {rhs}")
    
    k1 = lhs.shape[-1]
    k2 = rhs.shape[-2]
    
    bindings = {}
    if k1 != k2:
        # Se uno è simbolico e l'altro concreto
        if not k1.isdigit() and k2.isdigit():
            bindings[k1] = k2
        elif k1.isdigit() and not k2.isdigit():
            bindings[k2] = k1
        elif not k1.isdigit() and not k2.isdigit():
            bindings[k1] = k2
        else:
            raise TypeError(f"Dimensione interna incompatibile: {k1} != {k2}")
            
    out_shape = lhs.shape[:-1] + [rhs.shape[-1]]
    return TensorType(lhs.dtype, out_shape), bindings

# Test rapido di unificazione
x = TensorType("f32", ["B", "S", "D_in"])
w = TensorType("f32", ["D_in", "D_out"])
out, env = unify_matmul(x, w)
print("Unificazione Matmul di JANUS:")
print(f"  Input X: {x}")
print(f"  Pesi W:  {w}")
print(f"  Output:  {out}")
print(f"  Env:     {env}")
