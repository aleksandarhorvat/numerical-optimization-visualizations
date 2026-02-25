import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Example function (strongly convex)
def f(x, a):
    return 0.5*a*x**2 + np.sin(x) + 3

def second_derivative(x, a):
    return a - np.sin(x)

# Domain
X = np.linspace(-5, 5, 400)

a0 = 2.0   # controls base curvature
m0 = 0.5   # strong convexity parameter

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,7))
plt.subplots_adjust(bottom=0.25)

# Initial plots
line_f, = ax1.plot(X, f(X, a0), label='f(x)')
ax1.set_title('Function f(x)')
ax1.grid(True)
ax1.legend()

line_dd, = ax2.plot(X, second_derivative(X, a0), label="f''(x)")
line_m, = ax2.plot(X, np.ones_like(X)*m0, 'r--', label='m (lower bound)')
ax2.set_title('Hessian eigenvalue (1D = second derivative)')
ax2.grid(True)
ax2.legend()

# Sliders
ax_a = plt.axes([0.25, 0.1, 0.5, 0.03])
ax_m = plt.axes([0.25, 0.05, 0.5, 0.03])

slider_a = Slider(ax_a, 'Curvature a', 0.1, 5.0, valinit=a0)
slider_m = Slider(ax_m, 'm', 0.0, 3.0, valinit=m0)

def update(val):
    a = slider_a.val
    m = slider_m.val
    
    line_f.set_ydata(f(X, a))
    line_dd.set_ydata(second_derivative(X, a))
    line_m.set_ydata(np.ones_like(X)*m)
    
    # Check condition visually
    if np.all(second_derivative(X, a) >= m):
        ax2.set_title("Condition satisfied: f''(x) ≥ m → Strongly Convex ✅")
    else:
        ax2.set_title("Condition fails somewhere → Not strongly convex ❌")
    
    fig.canvas.draw_idle()

slider_a.on_changed(update)
slider_m.on_changed(update)

plt.show()
