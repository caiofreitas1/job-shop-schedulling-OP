import gurobipy as gp
from gurobipy import GRB

# Conjuntos e parâmetros (exemplo)
J = [0, 1, 2]  # Conjunto de jobs
M = [0, 1]  # Conjunto de máquinas
p = {
    (0, 0): 3, (0, 1): 2, (0, 2): 2,
    (1, 0): 2, (1, 1): 1, (1, 2): 4
}  # Tempo de processamento

# Criação do modelo
model = gp.Model("JobShopScheduling")

# Variáveis de decisão
x = model.addVars(M, J, vtype=GRB.CONTINUOUS, name="x")
z = model.addVars(M, J, J, vtype=GRB.BINARY, name="z")
C = model.addVar(vtype=GRB.CONTINUOUS, name="C")

# Função objetivo: minimizar o makespan
model.setObjective(C, GRB.MINIMIZE)

# Restrições
V = 1000  # Grande número
for j in J:
    for m in M:
        # Non-negativity
        model.addConstr(x[m, j] >= 0)

        # Makespan
        model.addConstr(x[m, j] + p[m, j] <= C)

    for h in range(1, len(M)):
        # Sequenciamento dentro de uma máquina
        model.addConstr(x[m, j] + p[m, j] <= x[m, j] + x[h, j])

for m in M:
    for j in J:
        for k in J:
            if j != k:
                # Sequenciamento entre máquinas
                model.addConstr(x[m, j] + p[m, j] <= x[m, k] + V * (1 - z[m, j, k]))
                model.addConstr(z[m, j, k] + z[m, k, j] == 1)

# Otimizar o modelo
model.optimize()

# Exibir resultados
if model.status == GRB.OPTIMAL:
    print(f"Optimal makespan: {C.X}")
    for m in M:
        for j in J:
            print(f"Start time of job {j} on machine {m}: {x[m, j].X}")
