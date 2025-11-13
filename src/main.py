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
import time

# ----------------------------
# 1. DEFINIÇÃO DO MDP
# ----------------------------

env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)

A = [0, 1]                      # Ações possíveis
gamma = 0.99                    # Fator de desconto
alpha = 0.1                     # Taxa de aprendizado (Q-learning)
epsilon_start = 1.0             # Começa explorando 100%
epsilon_end = 0.01              # Termina explorando apenas 1%
epsilon_decay_rate = 0.9995     # Quão rápido epsilon cai (ajuste esse valor)
epsilon = epsilon_start

EPISODES = 20000
MAX_STEPS = 1000

V = {}
Q = {}

# pasta para salvar imagens (vai ficar em ../images relativo a src/)
images_dir = os.path.join(os.path.dirname(__file__), "..", "images")
images_dir = os.path.abspath(images_dir)
os.makedirs(images_dir, exist_ok=True)
# pasta para salvar resultados (CSV)
results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
results_dir = os.path.abspath(results_dir)
os.makedirs(results_dir, exist_ok=True)

scores = []
avg_rewards = []
steps_per_episode = []
mean_V = []

best_ep = -1
best_score = -1
best_steps = 0
best_episode_by_score = None   # guarda o episódio com maior episode_score

# ----------------------------
# 2. COLETA DE EXPERIÊNCIAS E ATUALIZAÇÃO DE BELLMAN
# ----------------------------

for ep in range(EPISODES):
    obs, _ = env.reset()
    state = tuple(np.round(obs, 1))
    episode = []  # <- ADICIONADO: registrar (state, action, reward) para playback
    total_reward = 0
    episode_score = 0
    episode_steps = 0

    if epsilon > epsilon_end:
        epsilon *= epsilon_decay_rate

    for step in range(MAX_STEPS):
        # Política ε-greedy: explora ou explota
        if random.random() < epsilon:
            action = random.choice(A)
        else:
            q0 = Q.get((state, 0), 0.0)
            q1 = Q.get((state, 1), 0.0)
            action = 1 if q1 > q0 else 0

        new_obs, reward, terminated, truncated, _ = env.step(action)
        new_state = tuple(np.round(new_obs, 1))
        done = terminated or truncated

        episode.append((state, action, reward))  # <- ADICIONADO: salvar passo

        # Inicializa valores se ainda não existem
        if state not in V:
            V[state] = 0.0
        if (state, action) not in Q:
            Q[(state, action)] = 0.0

        # ----------------------------
        # 3. EQUAÇÃO DE BELLMAN - Q-LEARNING
        # ----------------------------
        # Q(s,a) <- Q(s,a) + α [R + γ max_a' Q(s',a') - Q(s,a)]
        best_next = max(Q.get((new_state, a), 0.0) for a in A)
        td_target = reward + gamma * best_next
        td_error = td_target - Q[(state, action)]
        Q[(state, action)] += alpha * td_error
        V[state] = max(Q.get((state, a), 0.0) for a in A)

        total_reward += reward
        episode_steps += 1
        if reward >= 0.99:
            episode_score += 1

        if done:
            break
        state = new_state

    # Atualiza métricas
    scores.append(episode_score)
    avg_rewards.append(total_reward)
    steps_per_episode.append(episode_steps)
    mean_V.append(np.mean(list(V.values())))

    # atualiza melhor episódio por número de canos passados (episode_score)
    if episode_score > best_score or (episode_score == best_score and episode_steps > best_steps):
        best_ep = ep
        best_score = episode_score
        best_steps = episode_steps
        best_episode_by_score = episode.copy()

env.close()

# ----------------------------
# 4. RESULTADOS TEXTUAIS
# ----------------------------

# Salvar média de V(s) por episódio em CSV (episode,mean_V)
try:
    csv_path = os.path.join(results_dir, "mean_V_per_episode.csv")
    np.savetxt(csv_path, np.column_stack((np.arange(len(mean_V)), mean_V)), delimiter=",", header="episode,mean_V", comments='')
    print(f"Saved mean_V per episode to: {csv_path}")
except Exception as e:
    print(f"Warning: failed to save mean_V CSV: {e}")

print("\n=== Função de Valor dos Estados (V) ===")
for i, (s, v) in enumerate(list(V.items())[:5]):
    print(f"Estado {i+1}: V(s) = {v:.3f}")

print("\n=== Função de Valor Estado-Ação (Q) ===")
for i, ((s, a), q) in enumerate(list(Q.items())[:5]):
    print(f"Q(s,a={a}) = {q:.3f}")

print(f"\nMelhor episódio: {best_ep} | Canos passados: {best_score} | Steps: {best_steps}")
print("\nCálculo finalizado com sucesso!")

# ----------------------------
# 5. GRÁFICOS DE ANÁLISE (SALVANDO)
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

# salva gráfico do melhor episódio (por score) se disponível
if best_episode_by_score:
    rewards_best = [r for (_, _, r) in best_episode_by_score]
    actions_best = [a for (_, a, _) in best_episode_by_score]
    plt.figure(figsize=(8,4))
    plt.plot(range(len(rewards_best)), rewards_best, marker='o', markersize=3)
    plt.xlabel("Passo do Episódio")
    plt.ylabel("Recompensa")
    plt.title(f"Recompensas por Passo - Melhor Episódio (score={best_score})")
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, "best_episode_by_score_rewards.png"))
    plt.close()

# ----------------------------
# 6. PLAYBACK DO MELHOR EPISÓDIO (por canos passados)
# ----------------------------
PLAY_BEST = True  # coloque False para não abrir a janela ao final
if PLAY_BEST:
    if not best_episode_by_score:
        print("Nenhum episódio válido encontrado para reprodução.")
    else:
        print(f"\nReproduzindo melhor episódio (por canos): episódio {best_ep}, canos={best_score}, steps={best_steps}")
        env_play = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        obs, _ = env_play.reset()
        done = False
        # reproduz ações salvas
        for (s, a, r) in best_episode_by_score:
            if done:
                break
            _, _, terminated, truncated, _ = env_play.step(a)
            done = terminated or truncated
            time.sleep(0.02)
        env_play.close()
