import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from collections import deque

# ============== Constants and globals ==============
g = -9.81  # acceleration due to gravity [m/s^2]
projectiles = deque()  # Stores dictionaries: {'line': artist, 'dot': artist, 'anim': animation_obj}


# ======================================== Physics ==================================
# 1: interpolates ground
def ground_intersection(x_prev, y_prev, t_prev, vx, vy, dt, y0=0):
    """
    Computes the exact ground-hit point (y = y0) within a time step.
    Returns: (x_hit, y_hit, t_hit, fraction)
    """
    y_next = y_prev + vy * dt

    # Fraction of dt required to reach the ground
    # α = (y0 - y_prev) / (y_next - y_prev)
    t_fraction = (y0 - y_prev) / (y_next - y_prev)

    # Clamp to [0,1] for safety
    t_fraction = max(0.0, min(1.0, t_fraction))

    x_hit = x_prev + vx * dt * t_fraction
    y_hit = y0
    t_hit = t_prev + dt * t_fraction

    return x_hit, y_hit, t_hit


# 2: Finds the trajectory of the projectile with RK4
def calc_trajectory(v0: float, theta: float, x0: float, y0: float, mass: float, wind_x: float, wind_y: float,
                    drag_coe: float, dt: float, N_max: int):
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)
    g = -9.81

    # Initial state vector: [x, y, vx, vy]
    state = np.array([x0, y0, vx0, vy0])

    # Pre-allocate arrays
    x = np.zeros(N_max)
    y = np.zeros(N_max)

    v = np.zeros(N_max)
    vx = np.zeros(N_max)
    vy = np.zeros(N_max)

    a = np.zeros(N_max)
    axx = np.zeros(N_max)
    ay = np.zeros(N_max)

    F_drag_ar = np.zeros(N_max)
    F_drag_ar_x = np.zeros(N_max)
    F_drag_ar_y = np.zeros(N_max)

    # set up initial conditions
    x[0], y[0] = x0, y0
    v[0], vx[0], vy[0] = v0, vx0, vx0

    # inverse mass for multiplication
    mass_inv = 1.0 / mass

    # --- RK4 HELPER FUNCTION ---
    # This function returns the derivative dy/dt, where y is the state vector [x, y, vx, vy]
    def derivatives(s: np.ndarray, wind_x, wind_y, drag_coe, mass_inv):
        # s = [x, y, vx, vy]
        vx_c, vy_c = s[2], s[3]

        # Relative velocity
        v_rel_x = vx_c - wind_x
        v_rel_y = vy_c - wind_y
        v_rel = np.sqrt(v_rel_x * v_rel_x + v_rel_y * v_rel_y)

        # Drag force
        # F_drag_x/y includes drag_coe and is applied opposite to v_rel
        F_drag_x = drag_coe * v_rel * v_rel_x
        F_drag_y = drag_coe * v_rel * v_rel_y

        # Accelerations (F = ma -> a = F/m)
        # ax = F_drag_x / mass
        # ay = g + F_drag_y / mass
        dvx_dt = F_drag_x * mass_inv
        dvy_dt = g + F_drag_y * mass_inv

        # Return derivatives: [dx/dt, dy/dt, dvx/dt, dvy/dt]
        return np.array([vx_c, vy_c, dvx_dt, dvy_dt])

    # 2. RK4 Integration Loop
    for i in range(1, N_max):

        # --- RK4 STEP ---
        # k1
        k1 = dt * derivatives(state, wind_x, wind_y, drag_coe, mass_inv)
        # k2
        k2 = dt * derivatives(state + k1 / 2, wind_x, wind_y, drag_coe, mass_inv)
        # k3
        k3 = dt * derivatives(state + k2 / 2, wind_x, wind_y, drag_coe, mass_inv)
        # k4
        k4 = dt * derivatives(state + k3, wind_x, wind_y, drag_coe, mass_inv)

        # New state vector
        state_new = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Extract new values
        x_new, y_new, vx_new, vy_new = state_new

        # --- Store and Check ---
        x[i], y[i] = x_new, y_new
        v[i], vx[i], vy[i] = np.sqrt(vx_new * vx_new + vy_new * vy_new), vx_new, vy_new

        # Calculate/Store secondary data (angles)
        v_rel_x = vx_new - wind_x
        v_rel_y = vy_new - wind_y

        F_drag_ar_x[i] = drag_coe * np.sqrt(v_rel_x ** 2 + v_rel_y ** 2) * v_rel_x
        F_drag_ar_y[i] = drag_coe * np.sqrt(v_rel_x ** 2 + v_rel_y ** 2) * v_rel_y
        F_drag_ar[i] = np.sqrt(F_drag_ar_x[i] * F_drag_ar_x[i] + F_drag_ar_y[i] * F_drag_ar_y[i])

        axx[i] = F_drag_ar_x[i] * mass_inv
        ay[i] = g + F_drag_ar_y[i] * mass_inv
        a[i] = np.sqrt(axx[i] * axx[i] + ay[i] * ay[i])

        # Tests if we have passed ground, assume ground is always at y = 0
        if y[i] < 0:
            # Interpolate to find precise intersection
            x_hit, y_hit, t_hit = ground_intersection(x[i - 1], y[i - 1], y[i], state[2], state[3], dt)

            x[i] = x_hit
            y[i] = y_hit

            # Trim arrays to the final index i
            x = x[:i + 1]
            y = y[:i + 1]

            v = v[:i + 1]
            vx = vx[:i + 1]
            vy = vy[:i + 1]

            a = a[:i + 1]
            axx = axx[:i + 1]
            ay = ay[:i + 1]

            F_drag_ar_x = F_drag_ar_x[:i + 1]
            F_drag_ar_y = F_drag_ar_y[:i + 1]
            F_drag_ar = F_drag_ar[:i + 1]
            break

        # Update state for next iteration
        state = state_new

        # Safety check for stability
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print(f"Warning: RK4 instability detected at step {i}. Stopping.")
            # Trim the arrays to the last stable point
            x = x[:i]
            y = y[:i]

            v = v[:i]
            vx = vx[:i]
            vy = vy[:i]

            a = a[:i]
            axx = axx[:i]
            ay = ay[:i]

            F_drag_ar_x = F_drag_ar_x[:i]
            F_drag_ar_y = F_drag_ar_y[:i]
            F_drag_ar = F_drag_ar[:i]
            break

    return x, y, v, vx, vy, a, axx, ay, F_drag_ar, F_drag_ar_y, F_drag_ar_x


# ============== Slider,Check boxes and Buttons ===============
# 1. creates a slider on the right side of the figure
def add_slider_right(text, row, col, frm, to, default, res=0.1):
    tk.Label(wind_frame, text=text).grid(row=row, column=col, sticky="e", padx=10)
    slider = tk.Scale(wind_frame, to=to, from_=frm, orient='horizontal', resolution=res)
    slider.set(default)
    slider.grid(row=row, column=col + 1, sticky="ew", padx=10, pady=5)
    return slider


# 2. creates a slider below the figure
def add_slider_below(text, row, col, frm, to, default):
    tk.Label(root, text=text).grid(row=row, column=col, sticky="e", padx=10)
    slider = tk.Scale(root, to=frm, from_=to, orient='horizontal', resolution=0.1)
    slider.set(default)
    slider.grid(row=row, column=col + 1, sticky="ew", padx=10, pady=5)
    return slider


# 3. adds a checkbox
def add_checkbox(text, container, row, col, boolean_var, columnspan):
    cb = tk.Checkbutton(container, text=text, variable=boolean_var)
    cb.grid(row=row, column=col, sticky="w", padx=10, pady=2, columnspan=columnspan)
    return cb


# 4. reset function
def reset_simulation():
    # Stop animations
    for p in projectiles:
        if p['anim'] and hasattr(p['anim'], 'event_source') and p['anim'].event_source:
            p['anim'].event_source.stop()
        p['line'].remove()
        p['dot'].remove()

    projectiles.clear()

    # Reset Axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    canvas.draw()


# 5. deletes the last projectile from the plot
def delete_last_proj():
    if projectiles:
        # Get last projectile
        p = projectiles.pop()

        # Stop animation if running
        if p['anim'] and hasattr(p['anim'], 'event_source') and p['anim'].event_source:
            p['anim'].event_source.stop()

        # Remove artists from plot
        p['line'].remove()
        p['dot'].remove()

        canvas.draw()


# 6. deletes the first projectile from the plot
def delete_first_proj():
    if projectiles:
        # Get first projectile
        p = projectiles.popleft()

        # Stop animation if running
        if p['anim'] and hasattr(p['anim'], 'event_source') and p['anim'].event_source:
            p['anim'].event_source.stop()

        # Remove artists from plot
        p['line'].remove()
        p['dot'].remove()

        canvas.draw()


# ========================================= Animation =================================
def animate(frame, x_array, y_array, line_artist, dot_artist):
    # Get the data up to the current frame
    current_x = x_array[:frame + 1]
    current_y = y_array[:frame + 1]

    line_artist.set_data(current_x, current_y)

    if len(current_x) > 0:
        dot_artist.set_data([current_x[-1]], [current_y[-1]])

    return line_artist, dot_artist


# ============================== Animation set up ===============================
def start_simulation():
    # Gather parameters
    v0 = vel1.get()
    theta = np.radians(ang1.get())
    m = mass1.get()
    x0 = x01.get()
    y0 = y01.get()
    r = r1.get() / 100.0
    wind_x_vel = wind_x.get()
    wind_y_vel = wind_y.get()
    dr_co = drag_coef.get()
    animation_duration = sim_time.get()

    # get number of steps and dt from the user, else tries to estimate
    def safe_float(x, default=0.0):
        try:
            return float(x)
        except ValueError:
            return default

    dt = safe_float(time_step.get()) * 1e-6 if dt_enable.get() else 0.001
    # Estimate max steps based on initial vertical velocity
    try:
        t_max_estimate = 2 * (v0 * np.sin(theta)) / abs(g)
    except ZeroDivisionError:
        t_max_estimate = 100.0  # Safety value

    # Set N based on the smallest time step (dt) and ensure a minimum duration
    N_max = int(t_max_estimate / dt * 1.5) + 200  # add buffer
    N = int(N_steps.get()) if N_enable.get() else N_max

    A = np.pi * r * r
    # Simplified drag coef calc, this allows to play with the radius
    actual_drag = -dr_co * A
    # Calculate Physics
    x, y, v, vx, vy, a, axx, ay, F, Fx, Fy = calc_trajectory(v0, theta, x0, y0, m, wind_x_vel, wind_y_vel, actual_drag,
                                                             dt, N)

    # --- AUTO SCALING ---
    # Get current plot boundaries
    curr_xmin, curr_xmax = ax.get_xlim()
    curr_ymin, curr_ymax = ax.get_ylim()

    # Get extremes of the new projectile path
    path_min_x = np.min(x)
    path_max_x = np.max(x)
    path_min_y = np.min(y)
    path_max_y = np.max(y)

    # 1. Calculate New X Limits
    # Expand LEFT if new path goes further left than current view
    # (We multiply by 1.1 to add padding. If negative, -10 * 1.1 = -11, which creates space)
    new_left = min(curr_xmin, path_min_x * 1.1 if path_min_x < 0 else path_min_x)

    # Expand RIGHT if new path goes further right
    new_right = max(curr_xmax, path_max_x * 1.1)

    # 2. Calculate New Y Limits
    new_bottom = min(curr_ymin, path_min_y * 1.1 if path_min_y < 0 else path_min_y)

    # Expand TOP
    new_top = max(curr_ymax, path_max_y * 1.1)

    # Apply the new limits
    ax.set_xlim(new_left, new_right)
    ax.set_ylim(new_bottom, new_top)

    # --- CREATE NEW ARTISTS ---
    # We create unique line and dot objects for this specific launch
    # Pick a random color or cycle colors if you wish
    color = next(ax._get_lines.prop_cycler)['color']

    new_line, = ax.plot([], [], '--', linewidth=2, color=color, label="Path")
    new_dot, = ax.plot([], [], 'o', markersize=6, color=color, zorder=5)

    # --- ANIMATION SETUP ---
    target_fps = 40
    total_frames = int(animation_duration * target_fps)

    # Ensure we don't exceed data length
    if total_frames > len(x):
        total_frames = len(x)

    indices = np.linspace(0, len(x) - 1, total_frames).astype(int)
    x_anim = x[indices]
    y_anim = y[indices]
    interval = 1000 / target_fps

    # Create the animation object
    new_anim = FuncAnimation(
        fig, animate, frames=len(x_anim),
        fargs=(x_anim, y_anim, new_line, new_dot),
        interval=interval, blit=True, repeat=False)

    # --- STORE DATA ---
    # this ensures proper deletion
    proj_data = {"line": new_line, "dot": new_dot, "anim": new_anim, "x": x, "y": y}
    projectiles.append(proj_data)

    # plots acceleration vs time
    def on_plot_a():
        fig2, axes = plt.subplots(3, 1, figsize=(6, 8))
        t = dt * np.arange(len(a))

        # --- |a| plot ---
        axes[0].plot(t[1:], a[1:])
        axes[0].set_title("|Acceleration| vs Time")
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("|a|")
        axes[0].grid(True)

        # --- ax plot ---
        axes[1].plot(t[1:], axx[1:])
        axes[1].set_title("Ax vs Time")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Ax")
        axes[1].grid(True)

        # --- ay plot ---
        axes[2].plot(t[1:], ay[1:])
        axes[2].set_title("Ay vs Time")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_ylabel("Ay")
        axes[2].grid(True)

        fig2.tight_layout()
        fig2.show()

    tk.Button(wind_frame, text="a(t)", command=on_plot_a).grid(column=0, row=11, sticky="w")

    # plots velocity vs time
    def on_plot_v():
        fig3, axes = plt.subplots(3, 1, figsize=(6, 8))
        t = dt * np.arange(len(v))

        # --- |v| plot ---
        axes[0].plot(t[1:], v[1:])
        axes[0].set_title("|Velocity| vs Time")
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("|v| [m/s]")
        axes[0].grid(True)

        # --- vx plot ---
        axes[1].plot(t[1:], vx[1:])
        axes[1].set_title("vx vs Time")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("vx [m/s]")
        axes[1].grid(True)

        # --- vy plot ---
        axes[2].plot(t[1:], vy[1:])
        axes[2].set_title("vy vs Time")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_ylabel("vy [m/s]")
        axes[2].grid(True)

        fig3.tight_layout()
        fig3.show()

    tk.Button(wind_frame, text="v(t)", command=on_plot_v).grid(column=0, row=13, sticky="w")

    # plots drag vs time
    def on_plot_F():
        fig4, axes = plt.subplots(3, 1, figsize=(6, 8))
        t = dt * np.arange(len(v))

        # --- |F| plot ---
        axes[0].plot(t[1:], F[1:])
        axes[0].set_title("|Drag| vs Time")
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("|F| [N]")
        axes[0].grid(True)

        # --- Fx plot ---
        axes[1].plot(t[1:], Fx[1:])
        axes[1].set_title("Fx vs Time")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Fx [N]")
        axes[1].grid(True)

        # --- Fy plot ---
        axes[2].plot(t[1:], Fy[1:])
        axes[2].set_title("Fy vs Time")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_ylabel("Fy [N]")
        axes[2].grid(True)

        fig4.tight_layout()
        fig4.show()

    tk.Button(wind_frame, text="F(t)", command=on_plot_F).grid(column=0, row=14, sticky="w")

    canvas.draw()


# ============================================== GUI set up =====================================
# --- TKINTER SETUP ---
root = tk.Tk()
root.title("Projectile Simulation")

# --- Matplotlib Figure ---
fig, ax = plt.subplots(figsize=(8, 5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=0, column=0, columnspan=4, pady=10, padx=10)

ax.set_xlabel("X position [m]")
ax.set_ylabel("Y position [m]")
ax.set_title("Projectile Motion")
ax.grid(True)
ax.set_xlim(0, 50)
ax.set_ylim(0, 20)

# Configure Grid weights
root.grid_columnconfigure(0, weight=0)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=0)
root.grid_columnconfigure(3, weight=1)
root.grid_columnconfigure(4, weight=0)

# ============================================= right side =============================
# --- WIND SLIDER FRAME ---
wind_frame = tk.Frame(root)
wind_frame.grid(row=0, column=4, sticky='n', padx=10, pady=10)
wind_frame.grid_columnconfigure(0, weight=1)

# booleans for user
dt_enable = tk.BooleanVar(value=False)
N_enable = tk.BooleanVar(value=False)

# Sliders for the parameters
tk.Label(wind_frame, text="Simulation Parameters", font=("Arial", 10, "bold")).grid(row=3, column=0, pady=5)
wind_x = add_slider_right('Wind x [m/s]', 4, 0, -20, 20, 0)
wind_y = add_slider_right('Wind y [m/s]', 4, 2, -20, 20, 0)
drag_coef = add_slider_right('Drag Coef', 5, 0, 0, 0.9, 0, 0.01)
sim_time = add_slider_right('Anim Time [s]', 5, 2, 0.1, 10, 1)

# dt set up
dt_box = add_checkbox('Time Step [μs]', wind_frame, 6, 0, dt_enable, 1)
time_step = tk.Entry(wind_frame, width=6)
time_step.grid(column=1, row=6, sticky="w", padx=(2, 10))
time_step.insert(0, "1000")

# number of steps set up
add_checkbox('Number of Steps', wind_frame, 7, 0, N_enable, 1)
N_steps = tk.Entry(wind_frame, width=6)
N_steps.grid(column=1, row=7, sticky="w", padx=(2, 10))
N_steps.insert(0, "10000")
tk.Label(wind_frame, text="Plot Functions", font=("Arial", 10, "bold")).grid(row=8, column=0, pady=5)

# --- BUTTONS ---
# launch projectile
launch_first = tk.Button(root, text="Launch", command=start_simulation, font=("Arial", 10, "bold"))
launch_first.grid(row=1, column=0, pady=10)
# Reset all
reset = tk.Button(root, text="Reset All", command=reset_simulation)
reset.grid(row=1, column=1, pady=10)
# delete last projectile
delete_last = tk.Button(root, text="Delete Last", command=delete_last_proj)
delete_last.grid(row=1, column=2, pady=10)
# delete first projectile
delete_first = tk.Button(root, text="Delete First", command=delete_first_proj)
delete_first.grid(row=1, column=3, pady=10)

# --- PARAMETERS ---
# sliders below the figure
tk.Label(root, text="Projectile Parameters", font=("Arial", 10, "bold")).grid(row=2, column=1, padx=10)
vel1 = add_slider_below("Initial Velocity [m/s]", 3, 0, 100, 1, 20)
ang1 = add_slider_below("Launch Angle [deg]", 4, 0, 89, 1, 45)
mass1 = add_slider_below("Mass [kg]", 5, 0, 100, 0.1, 1)
x01 = add_slider_below("x0 [m]", 3, 2, 20, -20, 0)
y01 = add_slider_below("y0 [m]", 4, 2, 20, -20, 0)
r1 = add_slider_below("radius [cm]", 5, 2, 200, 0.1, 1)

root.mainloop()
