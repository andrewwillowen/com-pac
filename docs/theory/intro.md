# Introduction

The math used in the `com-pac` package is based on the following description
from the textbook "Microwave Molecular Spectra" by Walter Gordy and Robert L. Cook, 
page 12:

!!! quote

    The classical angular momentum of a rigid system of particles 

    $$
    \mathbf{P} = \mathbf{I} \cdot \omega
    $$

    where $\omega$ is the angular velocity and $\mathbf{I}$ is the moment of
    inertia tensor which in dyadic notation is written as

    $$
    \begin{split}
      \mathbf{I} = & I_{xx} \mathbf{i} \mathbf{i} + I_{xy} \mathbf{i} \mathbf{j} + I_{xz} \mathbf{i} \mathbf{k} \\
                 & + I_{yx} \mathbf{j} \mathbf{i} + I_{yy} \mathbf{j} \mathbf{j} + I_{yz} \mathbf{j} \mathbf{k} \\
                 & + I_{zx} \mathbf{k} \mathbf{i} + I_{zy} \mathbf{k} \mathbf{j} + I_{zz} \mathbf{k} \mathbf{k} \\
    \end{split}
    $$

    with

    $$
    \begin{split}
      & I_{xx} = \sum{m (y^2 + z^2)} \\
      & I_{yy} = \sum{m (z^2 + x^2)} \\
      & I_{zz} = \sum{m (x^2 + y^2)} \\
      & I_{xy} = I_{yx} = - \sum{m x y} \\
      & I_{zx} = I_{xz} = - \sum{m x z} \\
      & I_{yz} = I_{zy} = - \sum{m y z} \\
    \end{split}
    $$ 

    in which $m$ is the mass of a particular particle and $x, y, z$ are its positional
    coordinates relative to a rectangular coordinate system fixed in the body and with
    its origin at the center of gravity of the body. The summation is taken over
    all the particles of the body. The origin of the coordinate system is chosen
    at the center of mass because this choice allows the total kinetic energy to be
    written as ... [to be continued]

