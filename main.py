import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import pandas as pd
import time

from instances import get_instance_data, generate_random_instance


def generate_distinct_colors(num_colors):
    colors = []
    np.random.seed(9)
    for i in range(num_colors):
        hue = np.random.rand()  # Random hue
        saturation = np.random.uniform(0.5, 0.8)  # Random saturation between 0.5 and 1.0
        lightness = np.random.uniform(0.1, 0.8)  # Random lightness between 0.4 and 0.8
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append((r, g, b))
    return colors


def solve_instance(instance_data):
    # Dados da instância
    n_jobs, n_machines = instance_data["jobs_machines"]
    J = [x + 1 for x in range(n_jobs)]
    M = [x + 1 for x in range(n_machines)]
    p = {}
    for j, job_data in enumerate(instance_data["data"]):
        for i in range(int(len(job_data) / 2)):
            p[(job_data[i * 2] + 1, j + 1)] = job_data[i * 2 + 1]
    # Criação do modelo
    model = gp.Model("JobShopScheduling")
    # Variáveis de decisão
    x = model.addVars(M, J, vtype=GRB.CONTINUOUS, name="x")
    z = model.addVars(M, J, J, vtype=GRB.BINARY, name="z")
    C = model.addVar(vtype=GRB.CONTINUOUS, name="C")
    # Função objetivo: minimizar o makespan
    model.setObjective(C, GRB.MINIMIZE)
    p_list = list(p.items())
    # Restrições
    V = 10000  # Grande número
    for j in J:
        for m in M:
            # Non-negativity
            model.addConstr(x[m, j] >= 0)

            # Makespan
            model.addConstr(x[m, j] + p[m, j] <= C)

        for k in range(n_machines):
            if k != 0:
                machine = p_list[((j - 1) * n_machines) + k][0][0]
                prev_machine = p_list[((j - 1) * n_machines) + k - 1][0][0]
                model.addConstr(x[machine, j] + p[machine, j] <= x[prev_machine, j])
    for m in M:
        for j in J:
            for k in J:
                if j != k:
                    # Sequenciamento entre máquinas
                    model.addConstr(x[m, j] + p[m, j] <= x[m, k] + V * (1 - z[m, j, k]))
                    model.addConstr(z[m, j, k] + z[m, k, j] == 1)
    # Otimizar o modelo
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    # Exibir resultados
    if model.status == GRB.OPTIMAL:
        # print time to solve n jobs with m machines
        for m in M:
            for j in J:
                print(f"Start time of job {j} on machine {m}: {x[m, j].X}")
        print(f"Optimal makespan: {C.X}")
    tasks = {}
    for j in J:
        tasks[f"Job {j}"] = []
        for m in M:
            newJob = (m, x[m, j].X, p[m, j])
            tasks[f"Job {j}"].append(newJob)
    fig, ax = plt.subplots()
    # Definindo cores para cada tarefa
    colors = generate_distinct_colors(len(J))
    # Plotando as barras do gráfico
    for i, (task, intervals) in enumerate(tasks.items()):
        for machine, start, duration in intervals:
            ax.broken_barh([(start, duration)], (machine - 0.5, 1), facecolors=(colors[i % len(colors)]), label=task)
    # Ajustando a legenda para não repetir labels
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())
    # Ajustando os eixos
    ax.set_ylim(0.5, len(M) + 0.5)
    ax.set_xlim(0, C.X)
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_yticks(np.arange(1, len(M) + 0.5, 1))
    plt.show()
    df = pd.DataFrame(tasks)
    print(df)
    return end_time - start_time


problem_sizes = [(11,11), (12,12), (13,13), (14,14), (15,15), (16,16), (17,17), (18,18), (19,19), (20,20)]

for size in problem_sizes:
    total_time = 0
    for i in range(3):
        instance = generate_random_instance(size[0], size[1])
        print(f"Instance: {instance['jobs_machines']}")
        total_time += solve_instance(instance)
    average_time_in_seconds = total_time / 3
    print(f"Average time to solve instance with {size[0]} jobs and {size[1]} machines: {average_time_in_seconds:.2f} seconds")