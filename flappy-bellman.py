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

EPISODES = 20000
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

# ----------------------------
# 2. COLETA DE EXPERIÊNCIAS E ATUALIZAÇÃO DE BELLMAN
# ----------------------------

for ep in range(EPISODES):
    obs, _ = env.reset()
    state = tuple(np.round(obs, 1))
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

    if episode_score > best_score:
        best_ep = ep
        best_score = episode_score
        best_steps = episode_steps

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

print(f"\nMelhor episódio: {best_ep} | Canos passados: {best_score} | Steps: {best_steps}")
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

plt.show()

# ----------------------------
# 6. APRENDIZAGEM POR REFORÇO ATIVA (Q-LEARNING EM AÇÃO)
# ----------------------------
"""
Aqui o agente já utiliza Q-Learning, uma forma ativa de aprendizado baseada diretamente na
Equação de Bellman. Após muitos episódios, os valores de Q(s,a) passam a guiar as ações.

- O termo α [R + γ·max(Q(s′,a′)) − Q(s,a)] é o “erro de Bellman”, ajustando o valor aprendido.
- A política ε-greedy permite explorar o ambiente e evitar overfitting em poucas ações.
- Assim, o agente tende a bater as asas nos momentos corretos, aprendendo a sobreviver mais tempo.

Em resumo:
✅ O agente começa aleatório (exploração pura)
✅ Aprende gradualmente por Bellman iterativo
✅ Converge para uma política que sobrevive e passa mais canos
"""

# Após o treinamento, o agente pode ser testado:
test_env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
obs, _ = test_env.reset()
state = tuple(np.round(obs, 1))
done = False
score = 0

while not done:
    # Escolhe a melhor ação aprendida
    q0 = Q.get((state, 0), 0.0)
    q1 = Q.get((state, 1), 0.0)
    action = 1 if q1 > q0 else 0

    obs, reward, terminated, truncated, info = test_env.step(action)
    state = tuple(np.round(obs, 1))
    done = terminated or truncated

    if reward >= 0.99:
        score += 1

print(f"\n=== Teste Final ===\nCanos passados: {score}")
test_env.close()
