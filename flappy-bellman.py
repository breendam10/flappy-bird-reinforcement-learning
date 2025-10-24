"""
Prova de Conceito - Aprendizado por Reforço
Ambiente: Flappy Bird (use_lidar=False)
Objetivo: Definir um MDP e aplicar a Equação de Bellman para estimar V(s) e Q(s,a)
Autor: Tredost
"""

import gymnasium as gym
import flappy_bird_gymnasium
import matplotlib.pyplot as plt
import numpy as np
import random
import time

# ----------------------------
# 1. DEFINIÇÃO DO MDP
# ----------------------------

env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)

A = [0, 1]      # Ações possíveis
gamma = 0.99      # Fator de desconto
alpha = 0.1      # Taxa de aprendizado (Q-learning)
epsilon_start = 1.0  # Começa explorando 100%
epsilon_end = 0.01   # Termina explorando apenas 1%
epsilon_decay_rate = 0.9995 # Quão rápido epsilon cai (ajuste esse valor)
epsilon = epsilon_start

EPISODES = 14000
MAX_STEPS = 1000

V = {}
Q = {}

scores = []
avg_rewards = []
steps_per_episode = []
mean_V = []

best_ep = -1
best_score = -1
best_steps = 0
best_actions = None
best_initial_obs = None

# ----------------------------
# 2. COLETA DE EXPERIÊNCIAS E ATUALIZAÇÃO DE BELLMAN
# ----------------------------

for ep in range(EPISODES):
    # Gera um seed por episódio para possibilitar replays determinísticos
    # Use np.random para não avançar o estado do gerador 'random' usado pela política
    ep_seed = int(np.random.randint(0, 2**31 - 1))
    obs, _ = env.reset(seed=ep_seed)
    initial_obs = obs
    state = tuple(np.round(obs, 1))
    total_reward = 0
    episode_score = 0
    episode_steps = 0
    episode_actions = []

    if epsilon > epsilon_end:
        epsilon *= epsilon_decay_rate

    for step in range(MAX_STEPS):
        if random.random() < epsilon:
            action = random.choice(A)
        else:
            q0 = Q.get((state, 0), 0.0)
            q1 = Q.get((state, 1), 0.0)
            action = 1 if q1 > q0 else 0

        episode_actions.append(action)

        new_obs, reward, terminated, truncated, _ = env.step(action)
        new_state = tuple(np.round(new_obs, 1))
        done = terminated or truncated

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

    if episode_score > best_score:
        best_ep = ep
        best_score = episode_score
        best_steps = episode_steps
        # Salva a sequência de ações (e a observação inicial) desse melhor episódio
        best_actions = episode_actions.copy()
        best_initial_obs = initial_obs
        best_seed = ep_seed

env.close()

# ----------------------------
# 5. GRÁFICOS DE ANÁLISE
# ----------------------------
'''''
plt.figure()
plt.plot(range(len(scores)), scores)
plt.xlabel("Episódio")
plt.ylabel("Canos Passados")
plt.title("Desempenho por Episódio (Flappy Bird)")
plt.grid(True)

plt.figure()
plt.plot(range(len(avg_rewards)), avg_rewards)
plt.xlabel("Episódio")
plt.ylabel("Recompensa Total")
plt.title("Evolução da Recompensa Média")
plt.grid(True)

plt.figure()
plt.plot(range(len(mean_V)), mean_V)
plt.xlabel("Episódio")
plt.ylabel("Média de V(s)")
plt.title("Convergência da Função de Valor dos Estados")
plt.grid(True)

Q0 = [q for (s, a), q in Q.items() if a == 0]
Q1 = [q for (s, a), q in Q.items() if a == 1]

plt.figure()
plt.bar(["Não bater (0)", "Bater (1)"], [np.mean(Q0), np.mean(Q1)], color=["#77aaff", "#ff7777"])
plt.title("Comparação Média de Q(s,a) entre Ações")
plt.ylabel("Valor Médio de Q(s,a)")

plt.figure()
plt.plot(range(len(steps_per_episode)), steps_per_episode)
plt.xlabel("Episódio")
plt.ylabel("Steps até o fim")
plt.title("Tempo de Sobrevivência por Episódio")
plt.grid(True)

plt.show()'''

# ----------------------------
# 6. APRENDIZAGEM POR REFORÇO ATIVA (Q-LEARNING EM AÇÃO)
# ----------------------------
"""
Em resumo:
✅ O agente começa aleatório (exploração pura)
✅ Aprende gradualmente por Bellman iterativo
✅ Converge para uma política que sobrevive e passa mais canos
"""

# Após o treinamento, reproduz a melhor run encontrada (se houver)
if best_actions is None or len(best_actions) == 0:
    print("\nNenhuma run salva como melhor episódio — não há replay disponível.")
else:
    print(f"\nReproduzindo melhor episódio #{best_ep} | Canos passados: {best_score} | Steps: {best_steps}")
    replay_env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    # Se tivermos o seed salvo, use-o para tentar reproduzir o mesmo estado inicial
    if 'best_seed' in globals():
        obs, _ = replay_env.reset(seed=best_seed)
    else:
        obs, _ = replay_env.reset()
    state = tuple(np.round(obs, 1))
    done = False
    score = 0

    # Executa a sequência de ações salva. Caso o episódio termine antes do fim da lista, paramos.
    for i, action in enumerate(best_actions):
        if done:
            break
        obs, reward, terminated, truncated, info = replay_env.step(action)
        state = tuple(np.round(obs, 1))
        done = terminated or truncated
        if reward >= 0.99:
            score += 1
        # Pequeno delay para a visualização do replay
        time.sleep(0.02)

    print(f"\n=== Replay Final ===\nCanos passados (reproduzidos): {score} | Steps executados: {i+1}")
    replay_env.close()
