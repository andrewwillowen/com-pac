# Theta Calculations

!!! abstract
    The method for calculating these theta values is derived from [Demaison, et al. J. Phys. Chem. A 2011, 115, 14078–14091](dx.doi.org/10.1021/jp2063595).

For planar molecules, `com-pac` calculates three estimates of the angle ($\theta$)
between the **parent species' principal axes** and an **isotopologue's principal axes**.
These are labelled $\theta_7$, $\theta_8$, and $\theta_9$, corresponding to the 
three equations described in the introduction of Demaison, et al. 

## Two coordinate frames

The inertia matrix of a molecule depends on the coordinate system in which the
atomic positions are expressed. `com-pac` works with two frames:

- **Parent frame**: the principal axes frame of the first (parent) isotopologue.
- **Isotopologue frame**: the principal axes frame specific to each isotopologue.

In the parent frame, only the parent isotopologue is expected to have a fully diagonal
interia matrix.
The other isotopologues when represented in the parent frame will almost certainly
have non-zero off-diagonal elements.
When the inertia matrix for an isotopologue is calculated while in its respective frame,
only then is the inertia matrix guaranteed to be fully diagonal.

In principle, the difference of an isotopologue's inertia matrix in the parent frame
and in its own frame is correlated with degree of rotation between the two frames.
The $\theta$ values are supposed to quantify that rotation.

!!! note
    This calculation is only performed for planar molecules, i.e. those for which
    the $I^e_{ac}$ and $I^e_{bc}$ elements of the parent inertia matrix are zero.
    It's not clear from the theory how a corresponding calculation would be done 
    for non-planar molecules.

## The three estimates

!!! warning
    The following calculations are based on the developer's interpretation of the
    paper and may be incorrect!

For each isotopologue, the following quantities are used:

- $I^e_{aa}$, $I^e_{bb}$, $I^e_{ab}$ — elements of the isotopologue's inertia
  matrix evaluated in the **parent frame**
- $I^e_a$, $I^e_b$ — principal moments of inertia from the isotopologue's **own
  principal axes frame** (i.e., the diagonal elements of its diagonalised inertia
  matrix)

All three $\theta$ values are reported in degrees.

### $\theta_7$

The following equation is derived from Equation 7 of Demaison et al.

$$
\theta_7 = \frac{1}{2} \arctan\!\left(\frac{2\, I^e_{ab}}{I^e_{aa} - I^e_{bb}}\right)
$$

### $\theta_8$

The following equation is derived from Equation 8 of Demaison et al.

$$
\theta_8 = \frac{1}{2} \arccos\!\left(\frac{I^e_{aa} - I^e_{bb}}{I^e_a - I^e_b}\right)
$$

### $\theta_9$

The following equation is derived from Equation 9 of Demaison et al.

$$
\theta_9 = \frac{1}{2} \arcsin\!\left(\frac{2\, I^e_{ab}}{I^e_a - I^e_b}\right)
$$

where $I^e_{ab}$ is the off-diagonal element of the inertia matrix evaluated in the
**parent frame**.

!!! note
    It's implied that $I^e_{ab}$ is the value for the isotopologue's inertia when
    in the **parent frame**, since in the **isotopologue frame** the value is guaranteed
    to be zero.
    
