import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ---------------------------------------
# Function and gradient
# ---------------------------------------
def f(x):
    return x**2

def grad_f(x):
    return 2*x

# ---------------------------------------
# Example point and descent direction
# ---------------------------------------
xk = 4
gk = grad_f(xk)
dk = -gk  # steepest descent

alpha0 = 0.25  # initial step
c0 = 0.7       # Wolfe parameter

# ---------------------------------------
# Plot setup
# ---------------------------------------
xs = np.linspace(-1, 5, 500)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)

ax.plot(xs, f(xs), label="f(x)")

curr_point, = ax.plot(xk, f(xk), 'bo', label="current point xk")
new_point, = ax.plot([], [], 'ro', label="new point xk+αdk")

# Directional derivative lines
dir_deriv_line = ax.plot([], [], 'g--', label="Directional derivative at xk")[0]
dir_deriv_scaled_line = ax.plot([], [], 'orange', linestyle='--', label="c * directional derivative at xk")[0]
dir_deriv_new_line = ax.plot([], [], 'm:', label="Directional derivative at new point")[0]

ax.legend()
ax.set_title("Wolfe Curvature Condition Example f(x)=x^2")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")

# ---------------------------------------
# Sliders
# ---------------------------------------
ax_alpha = plt.axes([0.2, 0.2, 0.65, 0.03])
alpha_slider = Slider(ax_alpha, "alpha", 0, 0.6, valinit=alpha0, valstep=0.01)

ax_c = plt.axes([0.2, 0.1, 0.65, 0.03])
c_slider = Slider(ax_c, "c", 0, 1, valinit=c0, valstep=0.01)

# ---------------------------------------
# Update function
# ---------------------------------------
def update(val):
    alpha = alpha_slider.val
    c = c_slider.val

    # New point
    x_new = xk + alpha * dk
    y_new = f(x_new)
    new_point.set_data(x_new, y_new)

    # Directional derivative at current point
    slope_xk = gk * dk
    y_dir_line = f(xk) + slope_xk * (xs - xk) / dk
    dir_deriv_line.set_data(xs, y_dir_line)

    # Scaled directional derivative at current point (c * slope)
    y_scaled_line = f(xk) + c * slope_xk * (xs - xk) / dk
    dir_deriv_scaled_line.set_data(xs, y_scaled_line)

    # Directional derivative at new point
    slope_new = grad_f(x_new) * dk
    y_new_line = f(x_new) + slope_new * (xs - x_new) / dk
    dir_deriv_new_line.set_data(xs, y_new_line)

    # Print status
    print(f"x_new={x_new:.3f}, dir_deriv_new={slope_new:.3f}, c*dir_deriv_xk={c*slope_xk:.3f}, Wolfe OK? {slope_new >= c*slope_xk}")

    fig.canvas.draw_idle()

alpha_slider.on_changed(update)
c_slider.on_changed(update)

update(None)
plt.show()