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
import warnings


def find_target_dirs(root: str = ".") -> dict:
    """Procura recursivamente por diretórios com nomes-alvo.

    Retorna um dicionário {folder_name: [path1, path2, ...]}.
    """
    names = ["mcfv-results", "mcev-results"]
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


def _aggregate_series(series: List[Tuple[str, Tuple[np.ndarray, np.ndarray]]]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Agrega várias séries (label, (x,y)) em uma grade x comum.

    Retorna (x_common, mean, min, max) ou None se não houver dados válidos.
    """
    if not series:
        return None
    # coletar todos os xs válidos
    valid = [(s[1][0], s[1][1]) for s in series if s[1] and s[1][0] is not None and s[1][1] is not None]
    if not valid:
        return None
    xs_list = [x for x, _ in valid]
    ys_list = [y for _, y in valid]
    xs_all = np.unique(np.concatenate(xs_list))
    xs_sorted = np.sort(xs_all)
    mat = np.full((len(valid), len(xs_sorted)), np.nan)
    for i, (x, y) in enumerate(valid):
        try:
            if x.size == 0 or y.size == 0:
                continue
            y_interp = np.interp(xs_sorted, x, y)
            # evitar extrapolação: valores fora do domínio original tornam-se NaN
            mask = (xs_sorted < x.min()) | (xs_sorted > x.max())
            if mask.any():
                y_interp[mask] = np.nan
            mat[i, :] = y_interp
        except Exception:
            continue

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(mat, axis=0)
        mn = np.nanmin(mat, axis=0)
        mx = np.nanmax(mat, axis=0)

    return xs_sorted, mean, mn, mx


def _plot_aggregate(xs: np.ndarray, mean: np.ndarray, mn: np.ndarray, mx: np.ndarray, out_path: str, title: str, label: Optional[str] = None, color: Optional[str] = None, fill_alpha: float = 0.2) -> None:
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    # plot and obtain the color chosen by matplotlib when color not provided
    if color is None:
        line, = ax.plot(xs, mean, label=label or "Média", linewidth=2.6)
        color = line.get_color()
    else:
        line, = ax.plot(xs, mean, label=label or "Média", color=color, linewidth=2.6)
    ax.fill_between(xs, mn, mx, color=color, alpha=fill_alpha)
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Média de V(s)")
    ax.set_title(title)
    ax.grid(True)
    if label:
        ax.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_combined_aggregates(aggregates: List[Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]], out_path: str, title: str) -> None:
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    cmap = plt.get_cmap("tab10")
    for i, (folder_name, (xs, mean, mn, mx)) in enumerate(aggregates):
        color = cmap(i % 10)
        ax.plot(xs, mean, label=folder_name, color=color, linewidth=2.6)
        ax.fill_between(xs, mn, mx, color=color, alpha=0.12)
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Média de V(s)")
    ax.set_title(title)
    ax.grid(True)
    if aggregates:
        ax.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(root: str = ".", out_dir: str = "images") -> int:
    root = os.path.abspath(root)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    found = find_target_dirs(root)
    combined_aggregates: List[Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = []

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

        if folder_series:
            agg = _aggregate_series(folder_series)
            if agg is None:
                print(f"Aviso: não foi possível agregar séries para: {folder_name}")
            else:
                xs, mean, mn, mx = agg
                out_path = os.path.join(out_dir, f"{folder_name}_convergence.png")
                _plot_aggregate(xs, mean, mn, mx, out_path, "Convergência da Função de Valor dos Estados", label=folder_name)
                combined_aggregates.append((folder_name, (xs, mean, mn, mx)))
                print(f"Salvou gráfico por pasta: {out_path}")
        else:
            print(f"Nenhum CSV encontrado em pasta(s) para: {folder_name}")

    if combined_aggregates:
        out_path = os.path.join(out_dir, "all_convergence.png")
        _plot_combined_aggregates(combined_aggregates, out_path, "Convergência da Função de Valor dos Estados")
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
