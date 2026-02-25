import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ---------------------------------------
# Function
# ---------------------------------------
def f(x):
    return x**2

def grad_f(x):
    return 2*x

# ---------------------------------------
# Example point
# ---------------------------------------
xk = 4
gk = grad_f(xk)
dk = -gk            # steepest descent = -8

alpha0 = 0.25       # gives x_new = 2
eta0 = 0.5

# ---------------------------------------
# Plot setup
# ---------------------------------------
xs = np.linspace(-1, 5, 500)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)

ax.plot(xs, f(xs), label="f(x)")

curr_point, = ax.plot(xk, f(xk), 'bo', label="current point (4,16)")
new_point, = ax.plot([], [], 'ro', label="new point")
rhs_point, = ax.plot([], [], 'mo', label="Armijo RHS point")  # purple point

# Tangent line (full gradient)
tangent_line, = ax.plot([], [], 'g--', label="tangent at xk")

# Orange tilted line (scaled by eta)
eta_line, = ax.plot([], [], 'orange', linestyle='--', label="Armijo slope line")

ax.legend()
ax.set_title("Armijo Condition Example f(x)=x^2, 4 → 2")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")

# ---------------------------------------
# Sliders
# ---------------------------------------
ax_alpha = plt.axes([0.2, 0.2, 0.65, 0.03])
alpha_slider = Slider(ax_alpha, "alpha", 0, 0.6,
                      valinit=alpha0, valstep=0.01)

ax_eta = plt.axes([0.2, 0.1, 0.65, 0.03])
eta_slider = Slider(ax_eta, "eta", 0, 1,
                    valinit=eta0, valstep=1e-4)

# ---------------------------------------
# Update function
# ---------------------------------------
def update(val):
    alpha = alpha_slider.val
    eta = eta_slider.val

    # New point (red)
    x_new = xk + alpha * dk
    y_new = f(x_new)
    new_point.set_data(x_new, y_new)

    # Purple RHS point moves along tilted orange line
    y_rhs = f(xk) + eta * gk * (x_new - xk)
    rhs_point.set_data(x_new, y_rhs)

    # Tangent line (green, full slope)
    y_tangent = f(xk) + gk * (xs - xk)
    tangent_line.set_data(xs, y_tangent)

    # Orange tilted line (for visualization of Armijo slope)
    y_eta_line = f(xk) + eta * gk * (xs - xk)
    eta_line.set_data(xs, y_eta_line)

    # Optional: print info
    print(f"x_new={x_new:.3f}, f(new)={y_new:.3f}, RHS={y_rhs:.3f}, Armijo OK? {y_new <= y_rhs}")

    fig.canvas.draw_idle()

alpha_slider.on_changed(update)
eta_slider.on_changed(update)

update(None)
plt.show()