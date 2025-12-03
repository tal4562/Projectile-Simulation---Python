# Projectile-Simulation---Python
Small python code for a projectile with wind and drag.


## Installation

To run the code, you need to install the following Python libraries. You can install them using pip:

```bash
pip install matplotlib numpy
```
This project also requires Tkinter. Some systems do not include it by default, so you may need to install it manually.


## GUI Explanation ##

<img width="800" height="763" alt="Proj_GUI" src="https://github.com/user-attachments/assets/c5839b70-4a55-4957-9278-8e4f1f24b519" />

Below the figure lie the projectile sliders and buttons.

Those include:

- **Launch** — sets a new projectile.
- **Reset** — clears the figure.
- **Delete Last** and **Delete First** — remove the last or first projectile path.
- **Initial Velocity**
- **Launch Angle**
- **Initial Location**
- **Mass**
- **Radius**

On the right of the figure are the controls for the simulation parameters:

- **Wind Velocity**
- **Animation Time**
- **Drag Coefficient**
- **Time Step** — optional; unless the user toggles it, a default step size is used.
- **Number of Steps** — optional; unless the user toggles it, an estimated maximum number of steps is used.
Drag is modeled using the standard quadratic drag equation:

F_drag = -0.5 * ρ * A * C_d * norm(v) * v

In this simulation, the drag coefficient (C_d) is defined to include
all aerodynamic constants **except** the negative sign and the area A.
This allows the user to change the projectile radius (which affects A)
without needing to modify the drag coefficient itself.
The negative sign is added inside the code to ensure that drag
always acts opposite to the direction of motion.


where:
- rho is the air density
- A is the cross-sectional area of the projectile
- C_d is the drag coefficient
- v is the velocity vector
- The negative sign means drag always opposes motion

The negative sign ensures that drag always acts opposite to the direction of motion.

The equations of motion are integrated using a classical **Runge–Kutta 4 (RK4)** solver for improved stability and accuracy compared to simple Euler integration. Despite using RK4, the simulation does not require extremely small time steps; moderate values provide smooth and stable trajectories.

Upon launching a projectile an option to plot acceleration, velocity and drag versus time will pop.









