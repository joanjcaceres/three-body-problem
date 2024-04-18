import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from multiprocessing import Pool
import imageio
import os

def three_body_problem(t, y):
    #hola
    m1, m2, m3 = 1.01, 0.9, 1  # Masas de los cuerpos
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = y
    r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2)
    r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
    
    dx1dt = vx1
    dy1dt = vy1
    dx2dt = vx2
    dy2dt = vy2
    dx3dt = vx3
    dy3dt = vy3
    
    dvx1dt = -m2 * (x1 - x2) / r12**3 - m3 * (x1 - x3) / r13**3
    dvy1dt = -m2 * (y1 - y2) / r12**3 - m3 * (y1 - y3) / r13**3
    dvx2dt = -m1 * (x2 - x1) / r12**3 - m3 * (x2 - x3) / r23**3
    dvy2dt = -m1 * (y2 - y1) / r12**3 - m3 * (y2 - y3) / r23**3
    dvx3dt = -m1 * (x3 - x1) / r13**3 - m2 * (x3 - x2) / r23**3
    dvy3dt = -m1 * (y3 - y1) / r13**3 - m2 * (y3 - y2) / r23**3

    return [dx1dt, dy1dt, dx2dt, dy2dt, dx3dt, dy3dt, dvx1dt, dvy1dt, dvx2dt, dvy2dt, dvx3dt, dvy3dt]

def plot_frame(args):
    fig, ax = plt.subplots()
    i, solution, max_trace_length = args
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    # Determinar el rango de índices para trazar
    start_idx = max(0, i - max_trace_length)
    end_idx = i + 1
    
    # Dibujar la trayectoria con desvanecimiento
    for j in range(start_idx, end_idx):
        alfa = 1 - (i - j) / max_trace_length  # Alfa se reduce con la edad del segmento
        ax.plot(solution.y[0, j:j+2], solution.y[1, j:j+2], 'r-', alpha=alfa)
        ax.plot(solution.y[2, j:j+2], solution.y[3, j:j+2], 'g-', alpha=alfa)
        ax.plot(solution.y[4, j:j+2], solution.y[5, j:j+2], 'b-', alpha=alfa)
    
    filename = f'frame_{i:04d}.png'
    fig.savefig(filename)
    plt.close(fig)
    return filename

def main():
    y0 = [0.97000436, -0.24308753, -0.97000436, 0.24308753, 0, 0, 0.466203685, 0.43236573, 0.466203685, 0.43236573, -0.93240737, -0.86473146]
    t_span = [0, 50]
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    solution = solve_ivp(three_body_problem, t_span, y0, t_eval=t_eval, rtol=1e-5)
    max_trace_length = 100  # Mantener solo las últimas 100 posiciones para desvanecimiento
    
    with Pool() as pool:  # Dejar libre un núcleo
        filenames = pool.map(plot_frame, [(i, solution, max_trace_length) for i in range(len(t_eval))])

    with imageio.get_writer('three_body_problem_fade.gif', mode='I', duration=0.05) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            os.remove(filename)

if __name__ == '__main__':
    main()