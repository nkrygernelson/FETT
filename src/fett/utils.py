from typing import List, Tuple
from pymatgen.core.composition import Composition

def get_atomic_number_from_ele(ele):
    return Composition(ele).elements[0].Z

def formula_to_set_representation(formula: str, max_elements: int = 10) -> Tuple[List[int], List[float]]:
    """
    Convert a chemical formula to a set representation with elements and fractional weights.
    Returns None if formula is invalid.
    """
    try:
        comp = Composition(formula)
        elements = []
        weights = []
        
        # Get total number of atoms
        total_atoms = comp.num_atoms
        
        # Extract elements and their fractional amounts
        for element, amount in comp.items():
            elements.append(element.Z)  # Atomic number
            weights.append(amount / total_atoms)  # Fractional amount
        
        # Sort by atomic number
        sorted_pairs = sorted(zip(elements, weights), key=lambda x: x[0])
        elements, weights = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        # Pad to max_elements
        padding_needed = max_elements - len(elements)
        if padding_needed > 0:
            elements = list(elements) + [0] * padding_needed  # 0 for padding
            weights = list(weights) + [0.0] * padding_needed
        elif padding_needed < 0:
            # Truncate
            elements = list(elements)[:max_elements]
            weights = list(weights)[:max_elements]
            # Renormalize weights
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]
            
        return list(elements), list(weights)
    except Exception:
        # Return None for invalid formulas so caller can handle (e.g. drop row)
        return None
