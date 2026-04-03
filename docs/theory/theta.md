# Theta Calculations

For planar molecules, `com-pac` calculates three estimates of the angle ($\theta$)
between the **parent species' principal axes** and an **isotopologue's principal axes**.
These are labelled $\theta_7$, $\theta_8$, and $\theta_9$.

## Two coordinate frames

The inertia matrix of a molecule depends on the coordinate system in which the
atomic positions are expressed. `com-pac` works with two frames:

- **Parent frame**: the principal axes frame of the first (parent) isotopologue.
  All isotopologue coordinates are expressed in this frame.
- **Isotopologue frame**: the principal axes frame specific to each isotopologue,
  where the inertia matrix is diagonal and the two diagonal elements are the
  principal moments of inertia $I^e_a$ and $I^e_b$.

When an isotopologue's masses are assigned to atomic positions in the *parent frame*,
the resulting inertia matrix is generally not diagonal — the off-diagonal element
$I^e_{ab}$ becomes non-zero, reflecting the small rotation between the two frames.
The $\theta$ values quantify that rotation.

!!! note
    This calculation is only performed for planar molecules, i.e. those for which
    the $I^e_{ac}$ and $I^e_{bc}$ elements of the parent inertia matrix are zero.

## The three estimates

For each isotopologue, the following quantities are used:

- $I^e_{aa}$, $I^e_{bb}$, $I^e_{ab}$ — elements of the isotopologue's inertia
  matrix evaluated in the **parent frame**
- $I^e_a$, $I^e_b$ — principal moments of inertia from the isotopologue's **own
  principal axes frame** (i.e., the diagonal elements of its diagonalised inertia
  matrix)

### $\theta_7$

$$
\theta_7 = \frac{1}{2} \arctan\!\left(\frac{2\, I^e_{ab}}{I^e_{aa} - I^e_{bb}}\right)
$$

### $\theta_8$

$$
\theta_8 = \frac{1}{2} \arccos\!\left(\frac{I^e_{aa} - I^e_{bb}}{I^e_a - I^e_b}\right)
$$

### $\theta_9$

$$
\theta_9 = \frac{1}{2} \arcsin\!\left(\frac{2\, I^e_{ab}}{I^e_a - I^e_b}\right)
$$

where $I^e_{ab}$ is the off-diagonal element of the inertia matrix evaluated in the
**parent frame**.

All three $\theta$ values are reported in degrees.
