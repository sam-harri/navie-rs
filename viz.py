import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import os
from matplotlib import cm, ticker

# Configuration
JSON_FILE_PATH = "output/wide.json"
SAVE_GIF = True
GIF_FILENAME = "output/wide.gif"
ANIMATION_INTERVAL = 50
CMAP = 'viridis'
VMIN = 0.0
VMAX = 1.0
CONTOUR_LEVELS = 20
SHOW_PRESSURE_CONTOURS = False

print(f"Loading data from: {JSON_FILE_PATH}")
try:
    with open(JSON_FILE_PATH, 'r') as f:
        results = json.load(f)
except FileNotFoundError:
    print(f"Error: JSON file not found at {JSON_FILE_PATH}"); exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {JSON_FILE_PATH}"); exit()

metadata = results['metadata']
data = results['data']
if not data: print("Error: No timestep data found in the JSON file."); exit()

# Extract Metadata
nx = metadata['nx']; ny = metadata['ny']; dx = metadata['dx']; dy = metadata['dy']
re = metadata['re']; dt = metadata['dt']
print(f"Grid: {nx}x{ny}, dx={dx:.4f}, dy={dy:.4f}, Re={re}, dt={dt}")
print(f"Color range fixed to: [{VMIN}, {VMAX}]")

# Create Grid Coordinates (Cell Centers)
xce = np.linspace(0.5 * dx, (nx - 0.5) * dx, nx)
yce = np.linspace(0.5 * dy, (ny - 0.5) * dy, ny)
Xce, Yce = np.meshgrid(xce, yce, indexing='ij')

fig, ax = plt.subplots(figsize=(7, 7))

def get_fields(timestep_idx):
    step_data = data[timestep_idx]
    step = step_data['step']; time = step_data['time']
    u = np.array(step_data['u_centers']).reshape((nx, ny), order='F').T
    v = np.array(step_data['v_centers']).reshape((nx, ny), order='F').T
    p = np.array(step_data['p_centers']).reshape((nx, ny), order='F').T
    magnitude = np.sqrt(u**2 + v**2)
    magnitude = np.clip(magnitude, VMIN, VMAX)
    return step, time, u, v, p, magnitude

# Initial Plot
step0, time0, u0, v0, p0, mag0 = get_fields(0)

levels = np.linspace(VMIN, VMAX, CONTOUR_LEVELS)
cont = ax.contourf(Xce.T, Yce.T, mag0, levels=levels, cmap=CMAP)
cbar = fig.colorbar(cont, ax=ax, label='Velocity Magnitude')
cbar.set_ticks(np.linspace(VMIN, VMAX, 11))

if SHOW_PRESSURE_CONTOURS:
    p_cont = ax.contour(Xce.T, Yce.T, p0, levels=10, colors='white', alpha=0.5, linewidths=0.5)

ax.set_title(f"Step: {step0}, Time: {time0:.3f} s")
ax.set_xlabel("X"); ax.set_ylabel("Y")
ax.set_xlim(0, nx * dx); ax.set_ylim(0, ny * dy)
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()

def update(frame_idx):
    step, time, u, v, p, magnitude = get_fields(frame_idx)
    ax.clear()
    
    cont = ax.contourf(Xce.T, Yce.T, magnitude, levels=levels, cmap=CMAP)
    
    if SHOW_PRESSURE_CONTOURS:
        p_cont = ax.contour(Xce.T, Yce.T, p, levels=10, colors='white', alpha=0.5, linewidths=0.5)
    
    ax.set_title(f"Step: {step}, Time: {time:.3f} s")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_xlim(0, nx * dx); ax.set_ylim(0, ny * dy)
    ax.set_aspect('equal', adjustable='box')
    
    cbar.update_normal(cont)
    cbar.set_ticks(np.linspace(VMIN, VMAX, 11))

    print(f"  Processed frame {frame_idx+1}/{len(data)} (Step: {step})")
    return cont, ax.title

print("Creating animation...")
ani = animation.FuncAnimation(fig, update, frames=len(data),
                              interval=ANIMATION_INTERVAL, blit=False)

if SAVE_GIF:
    output_dir = os.path.dirname(GIF_FILENAME)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"Saving animation to {GIF_FILENAME}...")
    try:
        ani.save(GIF_FILENAME, writer='imageio', fps=1000/ANIMATION_INTERVAL)
        print("Animation saved successfully.")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Trying to show plot instead...")
        plt.show()
else:
    print("Displaying animation...")
    plt.show()