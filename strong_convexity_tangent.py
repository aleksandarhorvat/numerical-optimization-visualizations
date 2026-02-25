import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Strongly convex function
def f(x):
    return x**2

def grad_f(x):
    return 2*x

m = 0.5     # strong convexity parameter
x0 = 1    # point where tangent is taken

fig, ax = plt.subplots(figsize=(8,5))
plt.subplots_adjust(bottom=0.25)

X = np.linspace(-1, 5, 400)
ax.plot(X, f(X), label='f(x)', linewidth=2)

# Tangent line at x0
tangent_line, = ax.plot([], [], 'b--', label='Tangent line at x')
strong_bound_curve, = ax.plot([], [], 'm-', linewidth=2, label='Strong convexity lower bound')
y_point, = ax.plot([], [], 'ko', label='f(y)')

ax.scatter([x0], [f(x0)], color='red', zorder=5)
ax.text(x0, f(x0)+0.5, 'x', color='red')

ax.legend()
ax.grid(True)

# Slider for y
ax_y = plt.axes([0.25, 0.1, 0.5, 0.03])
slider_y = Slider(ax_y, 'y', -1.0, 5.0, valinit=4.0)

def update(val):
    y = slider_y.val
    
    # Tangent line
    T = f(x0) + grad_f(x0)*(X - x0)
    tangent_line.set_data(X, T)

    # Strong convexity lower bound curve
    strong_curve = f(x0) + grad_f(x0)*(X - x0) + (m/2)*(X - x0)**2
    strong_bound_curve.set_data(X, strong_curve)

    # Point y on function
    y_point.set_data([y], [f(y)])

    ax.set_title(f'y = {y:.2f}')
    fig.canvas.draw_idle()

slider_y.on_changed(update)
update(3.0)

plt.show()
