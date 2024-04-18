import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib.animation import FuncAnimation

def three_body_problem(t, y):
    # Simplificación del problema de tres cuerpos con masas y condiciones iniciales establecidas
    m1, m2, m3 = 1.0, 1.0, 1.0  # Masas
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = y
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)

    ax1 = m2 * (x2 - x1) / r12**3 + m3 * (x3 - x1) / r13**3
    ay1 = m2 * (y2 - y1) / r12**3 + m3 * (y3 - y1) / r13**3
    ax2 = m1 * (x1 - x2) / r12**3 + m3 * (x3 - x2) / r23**3
    ay2 = m1 * (y1 - y2) / r12**3 + m3 * (y3 - y2) / r23**3
    ax3 = m1 * (x1 - x3) / r13**3 + m2 * (x2 - x3) / r23**3
    ay3 = m1 * (y1 - y3) / r13**3 + m2 * (y2 - y3) / r23**3

    return [vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3]

def update(frame, y0, t_span):
    sol = solve_ivp(three_body_problem, [0, frame/10], y0, t_eval=[0, frame/10], rtol=1e-5)
    line1.set_data(sol.y[0], sol.y[1])
    line2.set_data(sol.y[2], sol.y[3])
    line3.set_data(sol.y[4], sol.y[5])
    return line1, line2, line3

def start_animation():
    global ani
    ani = FuncAnimation(fig, update, frames=300, fargs=(y0, t_span), blit=True, repeat=False)
    canvas.draw()

root = tk.Tk()
root.title("Three Body Problem Simulation")

fig = Figure()
ax = fig.add_subplot(111)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

y0 = [-0.97000436, 0.24308753, 0.97000436, -0.24308753, 0, 0, -0.466203685, 0.43236573, -0.466203685, 0.43236573, 0.93240737, -0.86473146]
t_span = [0, 10]

line1, = ax.plot([], [], 'r-')
line2, = ax.plot([], [], 'g-')
line3, = ax.plot([], [], 'b-')

ani = None
start_animation()

root.mainloop()
