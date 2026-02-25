import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def f(x):
    return x**2

x = 1.0
y = 4.0
m = 1
lambd0 = 0.5

fig, ax = plt.subplots(figsize=(8,5))
plt.subplots_adjust(bottom=0.25)

X = np.linspace(min(x,y)-1, max(x,y)+1, 400)
ax.plot(X, f(X), label='f(x)', linewidth=2)
ax.scatter([x,y], [f(x),f(y)], color='black')

lhs_point, = ax.plot([], [], 'ro', label='LHS f(λx+(1−λ)y)')
rhs_curve, = ax.plot([], [], 'g-', linewidth=2, label='Strong RHS curve')
rhs_line_regular, = ax.plot([], [], 'b--', label='Convex combo RHS')
dip_line, = ax.plot([], [], 'm-', linewidth=3, label='Strong convexity dip')

ax.legend()
ax.grid(True)

ax_lambda = plt.axes([0.25, 0.1, 0.5, 0.03])
slider_lambda = Slider(ax_lambda, 'λ', 0.0, 1.0, valinit=lambd0)

# --- Precompute full strong convex curve ---
lam_vals = np.linspace(0, 1, 200)
x_curve = lam_vals*x + (1-lam_vals)*y
rhs_regular_curve = lam_vals*f(x) + (1-lam_vals)*f(y)
dip_curve = (m/2)*lam_vals*(1-lam_vals)*(x-y)**2
rhs_strong_curve = rhs_regular_curve - dip_curve
rhs_curve.set_data(x_curve, rhs_strong_curve)

def update(val):
    lambd = slider_lambda.val

    lhs_x = lambd*x + (1-lambd)*y
    lhs_y = f(lhs_x)

    rhs_regular = lambd*f(x) + (1-lambd)*f(y)
    dip = (m/2)*lambd*(1-lambd)*(x-y)**2
    rhs_strong = rhs_regular - dip

    lhs_point.set_data([lhs_x], [lhs_y])
    rhs_line_regular.set_data([X[0], X[-1]], [rhs_regular, rhs_regular])
    dip_line.set_data([lhs_x, lhs_x], [rhs_strong, rhs_regular])

    ax.set_title(f'λ={lambd:.2f}   Dip={dip:.3f}')
    fig.canvas.draw_idle()

slider_lambda.on_changed(update)
update(lambd0)

plt.show()
