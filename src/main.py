"""
Prova de Conceito - Aprendizado por Reforço
Ambiente: Flappy Bird (use_lidar=False)
Objetivo: Definir um MDP e aplicar a Equação de Bellman para estimar V(s) e Q(s,a)
Autor: <seu nome>
"""

import gymnasium as gym
import flappy_bird_gymnasium
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# ----------------------------
# 1. DEFINIÇÃO DO MDP
# ----------------------------

env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)

A = [0, 1]      # Ações possíveis
gamma = 0.99      # Fator de desconto
alpha = 0.1      # Taxa de aprendizado (Q-learning)
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay_rate = 0.9995
epsilon = epsilon_start

EPISODES = 20000
MAX_STEPS = 1000

V = {}
Q = {}
returns_count = {}  # contador de visitas (s,a) para média incremental

# novo: pasta para salvar imagens
images_dir = os.path.join(os.path.dirname(__file__), "..", "images")
images_dir = os.path.abspath(images_dir)
os.makedirs(images_dir, exist_ok=True)

scores = []
avg_rewards = []
steps_per_episode = []
mean_V = []

best_ep = -1
best_score = -1
best_steps = 0
best_total_reward = -np.inf
best_episode = None

# ----------------------------
# 2. COLETA DE EXPERIÊNCIAS E ATUALIZAÇÃO - MONTE CARLO FIRST-VISIT
# ----------------------------

for ep in range(EPISODES):
    obs, _ = env.reset()
    state = tuple(np.round(obs, 1))
    episode = []  # lista de (state, action, reward)
    total_reward = 0
    episode_score = 0
    episode_steps = 0

    if epsilon > epsilon_end:
        epsilon *= epsilon_decay_rate

    for step in range(MAX_STEPS):
        # Política ε-greedy baseada em Q atual
        if random.random() < epsilon:
            action = random.choice(A)
        else:
            q0 = Q.get((state, 0), 0.0)
            q1 = Q.get((state, 1), 0.0)
            action = 1 if q1 > q0 else 0

        new_obs, reward, terminated, truncated, _ = env.step(action)
        new_state = tuple(np.round(new_obs, 1))
        done = terminated or truncated

        episode.append((state, action, reward))

        total_reward += reward
        episode_steps += 1
        if reward >= 0.99:
            episode_score += 1

        if done:
            break
        state = new_state

    # Após o episódio: First-Visit Monte Carlo para Q(s,a)
    T = len(episode)
    for t, (s_t, a_t, _) in enumerate(episode):
        # verificar se esta é a primeira visita de (s_t,a_t)
        first_visit = True
        for prev in episode[:t]:
            if prev[0] == s_t and prev[1] == a_t:
                first_visit = False
                break
        if not first_visit:
            continue

        # calcular retorno G_t = sum_{k=t}^{T-1} gamma^{k-t} * r_{k}
        G = 0.0
        pow_gamma = 1.0
        for k in range(t, T):
            G += pow_gamma * episode[k][2]
            pow_gamma *= gamma

        # inicializar contadores se necessário
        if (s_t, a_t) not in returns_count:
            returns_count[(s_t, a_t)] = 0
        if (s_t, a_t) not in Q:
            Q[(s_t, a_t)] = 0.0

        # atualização incremental da média (primeira visita)
        returns_count[(s_t, a_t)] += 1
        N = returns_count[(s_t, a_t)]
        Q[(s_t, a_t)] += (G - Q[(s_t, a_t)]) / N

    # atualizar V(s) como max_a Q(s,a) para cada estado visto no episódio
    for (s, _, _) in episode:
        V[s] = max(Q.get((s, a), 0.0) for a in A)

    # Atualiza métricas
    scores.append(episode_score)
    avg_rewards.append(total_reward)
    steps_per_episode.append(episode_steps)
    mean_V.append(np.mean(list(V.values())) if len(V) > 0 else 0.0)

    # novo: atualizar melhor episódio com base na recompensa total
    if total_reward > best_total_reward:
        best_ep = ep
        best_total_reward = total_reward
        best_steps = episode_steps
        best_episode = episode.copy()

env.close()

# ----------------------------
# 4. RESULTADOS TEXTUAIS
# ----------------------------

print("\n=== Função de Valor dos Estados (V) ===")
for i, (s, v) in enumerate(list(V.items())[:5]):
    print(f"Estado {i+1}: V(s) = {v:.3f}")

print("\n=== Função de Valor Estado-Ação (Q) ===")
for i, ((s, a), q) in enumerate(list(Q.items())[:5]):
    print(f"Q(s,a={a}) = {q:.3f}")

print(f"\nMelhor episódio (por recompensa total): {best_ep} | Recompensa: {best_total_reward:.3f} | Steps: {best_steps}")
print("\nCálculo finalizado com sucesso!")

# ----------------------------
# 5. GRÁFICOS DE ANÁLISE
# ----------------------------

plt.figure()
plt.plot(range(len(scores)), scores)
plt.xlabel("Episódio")
plt.ylabel("Canos Passados")
plt.title("Desempenho por Episódio (Flappy Bird)")
plt.grid(True)
plt.savefig(os.path.join(images_dir, "scores_per_episode.png"))
plt.close()

plt.figure()
plt.plot(range(len(avg_rewards)), avg_rewards)
plt.xlabel("Episódio")
plt.ylabel("Recompensa Total")
plt.title("Evolução da Recompensa Média")
plt.grid(True)
plt.savefig(os.path.join(images_dir, "avg_rewards.png"))
plt.close()

plt.figure()
plt.plot(range(len(mean_V)), mean_V)
plt.xlabel("Episódio")
plt.ylabel("Média de V(s)")
plt.title("Convergência da Função de Valor dos Estados")
plt.grid(True)
plt.savefig(os.path.join(images_dir, "mean_V_convergence.png"))
plt.close()

Q0 = [q for (s, a), q in Q.items() if a == 0]
Q1 = [q for (s, a), q in Q.items() if a == 1]

plt.figure()
plt.bar(["Não bater (0)", "Bater (1)"], [np.mean(Q0) if Q0 else 0.0, np.mean(Q1) if Q1 else 0.0], color=["#77aaff", "#ff7777"])
plt.title("Comparação Média de Q(s,a) entre Ações")
plt.ylabel("Valor Médio de Q(s,a)")
plt.savefig(os.path.join(images_dir, "mean_Q_comparison.png"))
plt.close()

plt.figure()
plt.plot(range(len(steps_per_episode)), steps_per_episode)
plt.xlabel("Episódio")
plt.ylabel("Steps até o fim")
plt.title("Tempo de Sobrevivência por Episódio")
plt.grid(True)
plt.savefig(os.path.join(images_dir, "steps_per_episode.png"))
plt.close()

# substitui teste ao vivo por relatório e gráfico do melhor episódio
if best_episode is not None:
    rewards_best = [r for (_, _, r) in best_episode]
    actions_best = [a for (_, a, _) in best_episode]

    print("\n=== Melhor Episódio (detalhes por passo) ===")
    print(f"Episódio: {best_ep} | Recompensa total: {best_total_reward:.3f} | Steps: {best_steps}")
    print(f"Ações (primeiros 50): {actions_best[:50]}")
    print(f'Recompensas (primeiros 50): {rewards_best[:50]}')

    plt.figure(figsize=(8,4))
    plt.plot(range(len(rewards_best)), rewards_best, marker='o', markersize=3)
    plt.xlabel("Passo do Episódio")
    plt.ylabel("Recompensa")
    plt.title("Recompensas por Passo - Melhor Episódio")
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, "best_episode_rewards.png"))
    plt.close()
else:
    print("Nenhum episódio armazenado como 'melhor episodio'.")
