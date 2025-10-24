# flappy-bird-reinforcement-learning — Conceitos e design

Esta documentação explica os conceitos e as decisões de design por trás deste projeto — uma prova de conceito de Aprendizado por Reforço (Reinforcement Learning) aplicada ao jogo Flappy Bird. O objetivo aqui é esclarecer a ideia, os fundamentos teóricos e as escolhas implementacionais adotadas, não fornecer um tutorial passo a passo de execução.

## Visão geral do projeto

O projeto demonstra como modelar o problema do Flappy Bird como um Processo de Decisão de Markov (MDP) simples e aplicar uma versão tabular de Q-Learning — uma implementação direta da Equação de Bellman — para estimar as funções de valor V(s) e Q(s,a). O agente observa um vetor contínuo do ambiente, discrretiza esse vetor para indexar tabelas (`V` e `Q`) e aprende com atualizações online a cada passo (temporal-difference).

Embora seja uma prova de conceito, o código inclui pontos importantes para estudos: escolha de discretização, política ε-greedy, armazenamento da melhor run para replay, métricas de avaliação por episódio e visualizações básicas para analisar convergência.

## Conceitos-chave

- Processo de Decisão de Markov (MDP): formaliza o problema como estados S, ações A, recompensas R, fator de desconto γ e transições probabilísticas P(s'|s,a). O objetivo é encontrar uma política π(a|s) que maximize o retorno esperado.
- Equação de Bellman: relação recursiva entre o valor de um estado e os valores dos estados futuros. Para Q-Learning usamos a forma temporal-difference:

  Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]

  onde α é a taxa de aprendizado e γ é o fator de desconto.
- Q-Learning: algoritmo off-policy que estima a função Q (estado-ação) sem exigir um modelo do ambiente. A cada transição (s,a,r,s') atualizamos Q usando o TD-target r + γ max_a' Q(s',a').
- Exploração vs Exploração (ε-greedy): a política usada mistura exploração aleatória com exploração da melhor ação atual; ε decai ao longo do tempo (epsilon decay) para transicionar de exploração para exploração dirigida.
- Discretização de estados: quando observações são contínuas, para usar uma abordagem tabular precisamos mapear observações a estados discretos. A discretização define a capacidade de generalização do agente.

## Principais escolhas deste projeto

- Ambiente: FlappyBird-v0 (via `flappy_bird_gymnasium`) com `use_lidar=False` — observações contínuas obtidas do ambiente.
- Ações: binárias — 0 (não bater) e 1 (bater). O agente escolhe entre essas duas ações em cada passo.
- Atualização: online por step (Q-learning TD update dentro do loop de tempo). Ou seja, Q e V são atualizados imediatamente após cada transição observada.
- Discretização atual: observações contínuas são arredondadas com uma casa decimal e transformadas em tupla, usada como chave nas tabelas:

  state = tuple(np.round(obs, 1))

  Essa escolha é simples e rápida, controlando a granularidade por número de casas decimais. Menos casas → menos estados; mais casas → maior cardinalidade e sparsidade.
- Reprodutibilidade e replay: o código gera um seed por episódio (usando o gerador NumPy para não interferir com o gerador `random` da política), armazena a sequência de ações do melhor episódio e reproduz essa sequência no final — isto permite inspecionar qualitativamente o comportamento aprendido.

## Hiperparâmetros relevantes

- γ (gamma): fator de desconto (ex.: 0.99). Controla quanto o agente valoriza recompensas futuras.
- α (alpha): taxa de aprendizado (ex.: 0.1). Controla o passo de atualização do Q.
- ε (epsilon): probabilidade inicial de explorar (ex.: 1.0) com decaimento exponencial controlado por `epsilon_decay_rate` até `epsilon_end`.
- Discretização (precisão): definir quantas casas decimais usar em `np.round` é uma escolha crítica — impacta número de estados e velocidade de aprendizagem.

## Métricas e análise

O projeto coleta e plota métricas por episódio para avaliação qualitativa:

- `scores`: número de canos passados por episódio (interpretado como sucesso em curto prazo).
- `avg_rewards`: soma das recompensas recebidas no episódio (indicador de retorno bruto).
- `steps_per_episode`: sobrevivência (tempo) por episódio.
- `mean_V`: média dos valores V(s) conhecidos — dá uma ideia de convergência dos valores estimados.

Essas métricas não substituem uma análise estatística robusta (vários seeds independentes, intervalos de confiança), mas ajudam a monitorar tendências e detectar regressões de desempenho.

## Limitações e suposições

- Abordagem tabular: funciona para espaços de estado relativamente pequenos. Em observações contínuas com alta dimensionalidade, a tabela será extremamente esparsa e o aprendizado lento.
- Discretização simples (arredondamento): fácil de implementar, mas sensível à escala das variáveis. Se diferentes componentes da observação tiverem ordens de grandeza distintas, recomenda-se normalização ou binning por dimensão.
- Reprodutibilidade parcial: o uso de `reset(seed=...)` tende a tornar o episódio inicial determinístico, mas a reprodução perfeita depende de como o ambiente usa semente(s) internas.
- Recompensas e sinal: o design de recompensa do ambiente controla fortemente o aprendizado; aqui usamos as recompensas fornecidas pelo `flappy_bird_gymnasium` sem shaping adicional.

## Extensões e próximas etapas sugeridas

As seguintes melhorias são naturais para transformar essa POC em uma linha de experimentos mais séria:

- Melhor discretização: substituir `np.round` por binning (com limites conhecidos por dimensão) ou por tile coding para melhores generalizações.
- Métodos funcionais: migrar de tabelas para aproximadores (rede neural) e aplicar DQN ou políticas ator-crítico para lidar com estados contínuos complexos.
- Avaliação estatística: rodar múltiplos experimentos independentes (com diferentes seeds) e reportar média ± desvio padrão das métricas.
- Salvamento de checkpoints e exportação de runs (JSON) para análise offline e comparação reproduzível.
- Geração de vídeos dos replays ou integração com ferramentas de visualização para inspeção qualitativa.

## Leitura recomendada (conceitos)

- Sutton & Barto — Reinforcement Learning: An Introduction (capítulos sobre MDPs, Monte Carlo, TD, Q-Learning)
- Tutoriais sobre discretização: binning, tile coding e hashing para RL tabular
- Artigos e tutoriais sobre DQN e aproximação de função para agentes em domínios com observações contínuas

---
