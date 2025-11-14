"""
comparador.py

Lê todos os arquivos .csv encontrados nas pastas chamadas:
  - mcfv-results
  - mcev-results
  - qlrn-results

Gera um gráfico combinado com todos os resultados sobrepostos e um
gráfico por pasta com os CSVs dessa pasta sobrepostos.

Os gráficos são salvos em `images/` no diretório raiz do projeto.

Título do gráfico: "Convergência da Função de Valor dos Estados"
Eixos: x = "Episódio"  |  y = "Média de V(s)"
"""

from __future__ import annotations

import os
import glob
import csv
import argparse
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def find_target_dirs(root: str = ".") -> dict:
    """Procura recursivamente por diretórios com nomes-alvo.

    Retorna um dicionário {folder_name: [path1, path2, ...]}.
    """
    names = ["mcfv-results", "mcev-results"] #"qlrn-results"
    found = {n: [] for n in names}

    # procurar no root e em root/results
    search_bases = [root, os.path.join(root, "results")]
    for base in search_bases:
        for n in names:
            pattern = os.path.join(base, "**", n)
            for p in glob.glob(pattern, recursive=True):
                if os.path.isdir(p) and p not in found[n]:
                    found[n].append(os.path.abspath(p))
    return found


def read_mean_v(csv_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Tenta ler (x, y) de um CSV onde y é a média de V(s).

    Suporta formatos com cabeçalho (ex: episode,mean_V) e sem cabeçalho
    (duas colunas: episodio, mean_V).
    Retorna (x, y) como arrays numpy ou None se falhar.
    """
    # Tentar com numpy.genfromtxt com header
    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        if data.size == 0:
            return None
        if hasattr(data, "dtype") and data.dtype.names:
            names = list(data.dtype.names)
            # preferir coluna chamada mean_V (insensível a caixa)
            lower = [n.lower() for n in names]
            if "mean_v" in lower:
                idx = lower.index("mean_v")
                y = data[names[idx]]
            else:
                # usar segunda coluna (índice 1) quando possível
                idx = 1 if len(names) > 1 else 0
                y = data[names[idx]]
            if "episode" in lower:
                x = data[names[lower.index("episode")]]
            else:
                x = np.arange(len(y))
            return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    except Exception:
        # fallback para leitura manual
        pass

    # Leitura CSV manual (mais tolerante):
    try:
        xs: List[float] = []
        ys: List[float] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows:
                return None
            # detectar se primeira linha é cabeçalho textual
            first = rows[0]
            is_header = any(any(c.isalpha() for c in cell) for cell in first)
            start_idx = 1 if is_header else 0
            # identificar colunas quando houver header
            idx_x = None
            idx_y = None
            if is_header:
                lowered = [c.strip().lower() for c in first]
                for i, h in enumerate(lowered):
                    if "episode" in h or h == "ep":
                        idx_x = i
                    if "mean" in h and "v" in h:
                        idx_y = i
                if idx_y is None:
                    idx_y = 1 if len(first) > 1 else 0
            for r in rows[start_idx:]:
                if not r or len(r) < 1:
                    continue
                try:
                    xval = float(r[idx_x]) if idx_x is not None else len(xs)
                except Exception:
                    xval = len(xs)
                try:
                    yval = float(r[idx_y]) if idx_y is not None else float(r[1] if len(r) > 1 else r[0])
                except Exception:
                    continue
                xs.append(xval)
                ys.append(yval)
        if not ys:
            return None
        return np.array(xs, dtype=float), np.array(ys, dtype=float)
    except Exception:
        return None


def plot_overlaid(series: List[Tuple[str, Tuple[np.ndarray, np.ndarray]]], out_path: str, title: str) -> None:
    plt.figure(figsize=(10, 6))
    for label, (x, y) in series:
        plt.plot(x, y, label=label)
    plt.xlabel("Episódio")
    plt.ylabel("Média de V(s)")
    plt.title(title)
    plt.grid(True)
    if series:
        plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(root: str = ".", out_dir: str = "images") -> int:
    root = os.path.abspath(root)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    found = find_target_dirs(root)
    combined_series: List[Tuple[str, Tuple[np.ndarray, np.ndarray]]] = []

    for folder_name, paths in found.items():
        folder_series: List[Tuple[str, Tuple[np.ndarray, np.ndarray]]] = []
        for p in paths:
            # procurar todos os CSVs recursivamente dentro da pasta
            for csv_file in glob.glob(os.path.join(p, "**", "*.csv"), recursive=True):
                data = read_mean_v(csv_file)
                if data is None:
                    print(f"Aviso: falha ao ler CSV: {csv_file}")
                    continue
                x, y = data
                label = os.path.relpath(csv_file, start=root)
                folder_series.append((label, (x, y)))
                combined_series.append((f"{folder_name}: {label}", (x, y)))

        if folder_series:
            out_path = os.path.join(out_dir, f"{folder_name}_convergence.png")
            plot_overlaid(folder_series, out_path, "Convergência da Função de Valor dos Estados")
            print(f"Salvou gráfico por pasta: {out_path}")
        else:
            print(f"Nenhum CSV encontrado em pasta(s) para: {folder_name}")

    if combined_series:
        out_path = os.path.join(out_dir, "all_convergence.png")
        plot_overlaid(combined_series, out_path, "Convergência da Função de Valor dos Estados")
        print(f"Salvou gráfico combinado: {out_path}")
    else:
        print("Nenhum CSV encontrado nas pastas alvo. Nenhum gráfico combinado gerado.")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerador de gráficos de convergência para CSVs de mean_V per episode")
    parser.add_argument("--root", "-r", default=".", help="Diretório raiz onde procurar pelas pastas (default: .)")
    parser.add_argument("--out", "-o", default="images", help="Diretório de saída para os gráficos (default: images)")
    args = parser.parse_args()
    raise SystemExit(main(root=args.root, out_dir=args.out))
