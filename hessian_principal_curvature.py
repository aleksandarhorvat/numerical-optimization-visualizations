import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------- Function, gradient, Hessian ----------
def f(x, y):
    return x**2 + y**2

def grad_f(x, y):
    return np.array([2*x, 2*y])

H = np.array([[2, 0],
              [0, 2]])  # Hessian matrix

# Display Hessian as column vectors
print("Hessian matrix:")
print(H)
print("\nHessian as column vectors:")
for i, col in enumerate(H.T):
    print(f"Column {i+1}: {col}")

# -------- Eigenanalysis of Hessian (Principal Curvatures) ----------
eigenvalues, eigenvectors = np.linalg.eig(H)
print("\n" + "="*60)
print("HESSIAN EIGENVECTORS & EIGENVALUES (Principal Curvatures)")
print("="*60)
print("\nIf Gradient tells us WHERE the function increases...")
print("Then Hessian eigenvectors tell us HOW the function CURVES!")
print("\nIn each principal direction:")
print("- POSITIVE eigenvalue = Function curves UP (convex)")
print("- NEGATIVE eigenvalue = Function curves DOWN (concave)")
print("- MAGNITUDE = Strength of curvature\n")

for i in range(len(eigenvalues)):
    direction = eigenvectors[:, i]
    curvature = eigenvalues[i]
    curve_type = "⬆ CONVEX (curves UP)" if curvature > 0 else "⬇ CONCAVE (curves DOWN)"
    print(f"Direction {i+1}: {direction} | Curvature: {curvature:.1f} | {curve_type}")

# -------- Point of expansion ----------
x0, y0 = 0.5, 0.5
z0 = f(x0, y0)
g0 = grad_f(x0, y0)

# -------- Grid ----------
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# -------- Tangent plane (uses gradient) ----------
T = z0 + g0[0]*(X - x0) + g0[1]*(Y - y0)

# -------- Quadratic approximation (uses Hessian) ----------
dx = X - x0
dy = Y - y0
Q = z0 + g0[0]*dx + g0[1]*dy + 0.5*(H[0,0]*dx**2 + 2*H[0,1]*dx*dy + H[1,1]*dy**2)

# -------- Plot ----------
fig = plt.figure(figsize=(14,10))

# Main 3D plot
ax = fig.add_subplot(121, projection='3d')

# Surface
ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')

# Tangent plane
ax.plot_surface(X, Y, T, alpha=0.3, color='black', linewidth=0.5)

# Quadratic approx
ax.plot_surface(X, Y, Q,  alpha=0.4, cmap='autumn')

# Gradient vector (direction of steepest increase)
ax.quiver(x0, y0, z0, g0[0], g0[1], 0, length=0.8, normalize=True, color='blue', arrow_length_ratio=0.3, linewidth=2.5, label='Gradient\n(steepest increase)')

# Hessian eigenvectors (principal curvature directions)
for i in range(len(eigenvalues)):
    eig_vector = eigenvectors[:, i]
    eig_value = eigenvalues[i]
    color = 'red' if eig_value > 0 else 'orange'  # Red for convex, orange for concave
    ax.quiver(x0, y0, z0, eig_vector[0], eig_vector[1], 0, 
              length=0.6, normalize=True, color=color, arrow_length_ratio=0.3, 
              linewidth=2, label=f'Principal Dir {i+1}\n(curvature={eig_value:.0f})')

# Point
ax.scatter(x0, y0, z0, s=100, color='black', zorder=5)

ax.set_title("Gradient vs Hessian Principal Curvature Directions", fontsize=12, fontweight='bold')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x,y)")
ax.legend(loc='upper left', fontsize=9)
ax.view_init(elev=25, azim=45)

# 2D plot showing curvature along principal directions
ax2 = fig.add_subplot(122)

# Sample along principal directions
t_range = np.linspace(-1.5, 1.5, 200)

for i in range(len(eigenvalues)):
    eig_vector = eigenvectors[:, i]
    eig_value = eigenvalues[i]
    
    # Points along this direction
    x_line = x0 + t_range * eig_vector[0]
    y_line = y0 + t_range * eig_vector[1]
    z_line = f(x_line, y_line)
    
    # Quadratic approximation along this direction
    z_quad = z0 + g0[0]*(x_line - x0) + g0[1]*(y_line - y0) + 0.5*eig_value*t_range**2
    
    color = 'red' if eig_value > 0 else 'orange'
    label = f'Dir {i+1}: {"Convex" if eig_value > 0 else "Concave"} (λ={eig_value:.0f})'
    
    ax2.plot(t_range, z_line, linestyle='-', linewidth=2, color=color, label=label, alpha=0.7)
    ax2.plot(t_range, z_quad, linestyle='--', linewidth=2, color=color, alpha=0.5)

ax2.axvline(0, color='black', linestyle=':', alpha=0.5, label='Expansion point')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel("Distance along principal direction (t)", fontsize=10)
ax2.set_ylabel("Function value", fontsize=10)
ax2.set_title("Function behavior along Principal Curvature Directions\n(solid=actual, dashed=quadratic approx)", fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.show()