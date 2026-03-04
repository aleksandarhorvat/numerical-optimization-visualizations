"""
Steepest Descent Visualizer
============================
Edit the PARAMETERS section, then run:  python steepest_descent.py
Terminal shows matrix A, function, factor, and per-step values.
Plot shows 2D level curves + trajectory with a step slider.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ─── PARAMETERS ────────────────────────────────────────────────
M        = 9.0        # largest eigenvalue
m        = 1.0        # smallest eigenvalue
b        = np.array([0.0, 0.0])   # b vector in f(x) = 1/2 x^T A x + b^T x + c
c        = 0.0                    # constant in f(x)
x0       = np.array([1.0, 1.0])   # starting point
max_iter = 20

# backtracking parameters
eta      = 0.5    # sufficient decrease parameter (Armijo condition)
beta     = 0.8    # step size reduction factor
alpha0   = 1.0    # initial step size guess
# ───────────────────────────────────────────────────────────────

# ─── matrix A (diagonal with eigenvalues M, m) ─────────────────
A = np.array([[M, 0.0],
              [0.0, m]])

# ─── function, gradient ────────────────────────────────────────
def f(x):
    return 0.5 * x @ A @ x + b @ x + c

def grad(x):
    return A @ x + b

# ─── exact minimizer x* = -A^{-1} b ───────────────────────────
x_star = -np.linalg.solve(A, b)

# ─── convergence factor ────────────────────────────────────────
factor = ((M - m) / (M + m))**2

# ─── terminal output: setup info ───────────────────────────────
print("=" * 55)
print("  STEEPEST DESCENT SETUP")
print("=" * 55)
print(f"\nMatrix A (diagonal, eigenvalues m={m}, M={M}):")
print(f"  A = [[{A[0,0]:.2f},  {A[0,1]:.2f}],")
print(f"       [{A[1,0]:.2f},  {A[1,1]:.2f}]]")
print(f"\nFunction:")
print(f"  f(x) = 1/2 * x^T A x + b^T x + c")
print(f"       = 1/2 * ({M:.2f}*x1^2 + {m:.2f}*x2^2)"
      + (f" + ({b[0]:.2f})*x1 + ({b[1]:.2f})*x2" if np.any(b != 0) else "")
      + (f" + {c:.2f}" if c != 0 else ""))
print(f"\nMinimizer x* = {x_star}")
print(f"f(x*)        = {f(x_star):.6f}")
print(f"\nConvergence factor ((M-m)/(M+m))^2:")
print(f"  = (({M}-{m})/({M}+{m}))^2 = ({M-m}/{M+m})^2 = {factor:.6f}")
print(f"\nStarting point x0 = {x0}")
print("=" * 55)

# ─── backtracking line search ──────────────────────────────────
def backtracking(x, d):
    a = alpha0
    fx = f(x)
    gd = grad(x) @ d
    while f(x + a * d) > fx + eta * a * gd:
        a *= beta
    return a

# ─── run steepest descent with backtracking ────────────────────
xs      = [x0.copy()]
alphas  = []

for _ in range(max_iter):
    g = grad(xs[-1])
    if np.linalg.norm(g) < 1e-12:
        break
    d = -g
    a = backtracking(xs[-1], d)
    alphas.append(a)
    xs.append(xs[-1] + a * d)

xs      = np.array(xs)
N       = len(xs)
f_vals  = np.array([f(x) for x in xs])
f_star  = f(x_star)
f_errs  = f_vals - f_star
dists   = np.linalg.norm(xs - x_star, axis=1)

# ─── precompute ratios ─────────────────────────────────────────
f_ratios = np.where(f_errs[:-1] > 1e-15, f_errs[1:] / f_errs[:-1], np.nan)
d_ratios = np.where(dists[:-1]  > 1e-15, dists[1:]  / dists[:-1],  np.nan)

# ─── 2D level curve plot ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
plt.subplots_adjust(bottom=0.2)

span = max(1.5, float(np.max(np.abs(xs))) + 0.5)
u    = np.linspace(-span, span, 300)
U, V = np.meshgrid(u, u)
Z    = np.array([[f(np.array([U[i,j], V[i,j]])) for j in range(U.shape[1])]
                 for i in range(U.shape[0])])

levels = np.percentile(Z[Z > f_star + 1e-6], np.linspace(1, 96, 16))
cs = ax.contour(U, V, Z, levels=levels, cmap='viridis', alpha=0.5)
ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')

ax.plot(*x_star, '*', color='cyan', ms=14, zorder=6, label=f'x* = {x_star}')

# ─── eigenvectors ──────────────────────────────────────────────
eigenvalues, eigenvectors = np.linalg.eigh(A)
arrow_len = span * 0.45

for i in range(2):
    lam = eigenvalues[i]
    vec = eigenvectors[:, i]
    color = '#ff4444' if lam == max(eigenvalues) else '#44aaff'
    label = f'eigvec λ={lam:.2f} ({"steep" if lam == max(eigenvalues) else "gentle"})'
    # draw arrow in both directions through x*
    ax.annotate('', xy=x_star + arrow_len * vec,
                xytext=x_star - arrow_len * vec,
                arrowprops=dict(arrowstyle='<->', color=color, lw=2))
    # label at tip
    ax.text(*(x_star + (arrow_len + 0.1) * vec), label,
            color=color, fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor=color))

ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title(f'Steepest Descent (backtracking)  |  M={M}, m={m}  |  factor={factor:.4f}')
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

# trajectory artists
full_path, = ax.plot(xs[:, 0], xs[:, 1], 'o--',
                     color='gray', lw=1, ms=3, alpha=0.3, zorder=2)
traj_line, = ax.plot([], [], 'o-', color='#f7806a', lw=2, ms=5, zorder=4)
curr_dot,  = ax.plot([], [], 'o',  color='red',     ms=10, zorder=5)

# ─── terminal print for current step ───────────────────────────
def print_step(k):
    print(f"\n--- Step k = {k} ---")
    print(f"  x_k                  = [{xs[k,0]:.6f}, {xs[k,1]:.6f}]")
    print(f"  f(x_k)               = {f_vals[k]:.6f}")
    print(f"  f(x_k) - f(x*)       = {f_errs[k]:.6f}")
    print(f"  ||x_k - x*||         = {dists[k]:.6f}")
    if k + 1 < N:
        print(f"\n  f(x_k+1) - f(x*)     = {f_errs[k+1]:.6f}")
        print(f"  ||x_k+1 - x*||       = {dists[k+1]:.6f}")
        if not np.isnan(f_ratios[k]):
            print(f"\n  f-ratio  f(k+1)/f(k) = {f_ratios[k]:.6f}  (should be ~{factor:.4f})")
        if not np.isnan(d_ratios[k]):
            print(f"  dist-ratio d(k+1)/d(k)= {d_ratios[k]:.6f}  (changes every step)")
        if k > 0 and not np.isnan(d_ratios[k]) and not np.isnan(d_ratios[k-1]):
            print(f"  prev dist-ratio       = {d_ratios[k-1]:.6f}")
            print(f"  delta dist-ratio      = {abs(d_ratios[k]-d_ratios[k-1]):.6f}"
                  + ("  <-- oscillating" if abs(d_ratios[k]-d_ratios[k-1]) > 0.01 else ""))
        theor_bound = factor * f_errs[k]
        holds = f_errs[k+1] <= theor_bound + 1e-12
        print(f"\n  Theorem 6.1 check:")
        print(f"    f(x_k+1) - f(x*)        = {f_errs[k+1]:.6f}")
        print(f"    factor x [f(x_k)-f(x*)] = {theor_bound:.6f}")
        print(f"    inequality holds?        = {'YES' if holds else 'NO'}")
    else:
        print("  (last step)")

# ─── slider update ─────────────────────────────────────────────
def update(val):
    k = int(slider.val)
    traj_line.set_data(xs[:k+1, 0], xs[:k+1, 1])
    curr_dot.set_data([xs[k, 0]], [xs[k, 1]])
    print_step(k)
    fig.canvas.draw_idle()

ax_sl  = fig.add_axes([0.15, 0.07, 0.7, 0.03])
slider = Slider(ax_sl, 'Step k', 0, N - 1, valinit=0, valstep=1)
slider.on_changed(update)

print_step(0)
plt.show()
