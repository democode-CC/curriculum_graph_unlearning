"""
Curriculum construction for CUNO.

ComplexityCalculator  -- scores each forget node by unlearning difficulty.
CurriculumDesigner    -- sorts and partitions nodes into ordered stages.

Supported complexity metrics
-----------------------------
Topology-based  : degree, betweenness, pagerank, clustering, eigenvector
Model-response  : prediction_confidence, gradient_norm, retain_coupling
Retain-coupling : multihop_retain_coverage, retain_betweenness, class_boundary
"""

from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ComplexityCalculator
# ---------------------------------------------------------------------------

class ComplexityCalculator:
    """Compute a scalar complexity score for each node in the forget set."""

    def __init__(self, data, model=None, retain_mask=None, device='cpu',
                 num_layers=2, hop_decay=0.5):
        self.data        = data
        self.model       = model
        self.retain_mask = retain_mask
        self.device      = device
        self.num_layers  = num_layers
        self.hop_decay   = hop_decay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, nodes, metric: str) -> dict:
        """
        Returns a dict {node_id: score} for every node in *nodes*.

        Args:
            nodes:  Iterable of node indices (int or 0-d tensor).
            metric: One of the supported metric names.
        """
        dispatch = {
            'degree':                   self._degree,
            'betweenness':              self._betweenness,
            'pagerank':                 self._pagerank,
            'clustering':               self._clustering,
            'eigenvector':              self._eigenvector,
            'prediction_confidence':    self._prediction_confidence,
            'gradient_norm':            self._gradient_norm,
            'retain_coupling':          self._retain_coupling,
            'multihop_retain_coverage': self._multihop_retain_coverage,
            'retain_betweenness':       self._retain_betweenness,
            'class_boundary':           self._class_boundary,
        }
        if metric not in dispatch:
            raise ValueError(f"Unknown metric '{metric}'. "
                             f"Choose from: {list(dispatch)}")
        return dispatch[metric](nodes)

    # ------------------------------------------------------------------
    # Topology-based metrics
    # ------------------------------------------------------------------

    def _ids(self, nodes):
        return [n.item() if torch.is_tensor(n) else int(n) for n in nodes]

    def _to_nx(self):
        ei = self.data.edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_edges_from(zip(ei[0], ei[1]))
        return G

    def _degree(self, nodes):
        ei = self.data.edge_index
        result = {}
        for idx in self._ids(nodes):
            deg = ((ei[0] == idx) | (ei[1] == idx)).sum().item()
            result[idx] = deg
        return result

    def _betweenness(self, nodes):
        bc = nx.betweenness_centrality(self._to_nx())
        return {idx: bc.get(idx, 0.0) for idx in self._ids(nodes)}

    def _pagerank(self, nodes):
        pr = nx.pagerank(self._to_nx())
        return {idx: pr.get(idx, 0.0) for idx in self._ids(nodes)}

    def _clustering(self, nodes):
        cc = nx.clustering(self._to_nx())
        return {idx: cc.get(idx, 0.0) for idx in self._ids(nodes)}

    def _eigenvector(self, nodes):
        G = self._to_nx()
        try:
            ec = nx.eigenvector_centrality(G, max_iter=1000)
        except Exception:
            ec = {n: G.degree(n) for n in G.nodes()}
        return {idx: ec.get(idx, 0.0) for idx in self._ids(nodes)}

    # ------------------------------------------------------------------
    # Model-response metrics
    # ------------------------------------------------------------------

    def _require_model(self, name):
        if self.model is None:
            raise ValueError(f"'{name}' requires a model passed to ComplexityCalculator.")

    def _prediction_confidence(self, nodes):
        self._require_model('prediction_confidence')
        self.model.eval()
        with torch.no_grad():
            data  = self.data.to(self.device)
            out   = self.model(data.x, data.edge_index)
            probs = F.softmax(out, dim=-1)
            neg_H = (probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return {idx: neg_H[idx].item() for idx in self._ids(nodes)}

    def _gradient_norm(self, nodes):
        self._require_model('gradient_norm')
        was_training = self.model.training
        self.model.train()
        saved = {p: p.requires_grad for p in self.model.parameters()}
        for p in self.model.parameters():
            p.requires_grad_(True)

        result = {}
        data = self.data.to(self.device)
        with torch.enable_grad():
            for idx in self._ids(nodes):
                self.model.zero_grad()
                out  = self.model(data.x, data.edge_index)
                loss = F.cross_entropy(out[idx].unsqueeze(0), data.y[idx].unsqueeze(0))
                loss.backward()
                gn = sum(
                    p.grad.detach().norm(2).item() ** 2
                    for p in self.model.parameters() if p.grad is not None
                ) ** 0.5
                result[idx] = gn
                self.model.zero_grad()

        for p, s in saved.items():
            p.requires_grad_(s)
        if not was_training:
            self.model.eval()
        return result

    def _retain_coupling(self, nodes):
        self._require_model('retain_coupling')
        if self.retain_mask is None:
            raise ValueError("'retain_coupling' requires retain_mask.")
        self.model.eval()
        result = {}
        with torch.no_grad():
            data  = self.data.to(self.device)
            emb   = self.model(data.x, data.edge_index)
            emb   = F.normalize(emb, dim=-1)
            ei    = data.edge_index
            r_set = set(self.retain_mask.nonzero(as_tuple=True)[0].cpu().tolist())
            for idx in self._ids(nodes):
                nbr_mask = (ei[0] == idx) | (ei[1] == idx)
                nbrs = (
                    set(ei[0, nbr_mask].cpu().tolist() + ei[1, nbr_mask].cpu().tolist())
                    & r_set
                ) - {idx}
                if nbrs:
                    v_emb = emb[idx].unsqueeze(0)
                    n_emb = emb[list(nbrs)]
                    result[idx] = (v_emb * n_emb).sum(dim=-1).mean().item()
                else:
                    result[idx] = 0.0
        return result

    # ------------------------------------------------------------------
    # Retain-coupling / graph-structure metrics
    # ------------------------------------------------------------------

    def _multihop_retain_coverage(self, nodes):
        if self.retain_mask is None:
            raise ValueError("'multihop_retain_coverage' requires retain_mask.")
        L      = self.num_layers
        alpha  = self.hop_decay   # default 0.5
        ei     = self.data.edge_index.cpu().numpy()
        r_set  = set(self.retain_mask.nonzero(as_tuple=True)[0].cpu().tolist())

        adj = defaultdict(set)
        for s, d in zip(ei[0], ei[1]):
            adj[s].add(d); adj[d].add(s)

        result = {}
        for idx in self._ids(nodes):
            visited  = {idx}
            frontier = {idx}
            score    = 0.0
            for k in range(1, L + 1):
                nxt = set()
                for v in frontier:
                    nxt |= (adj[v] - visited)
                visited |= nxt
                score   += (alpha ** (k - 1)) * len(nxt & r_set)
                frontier = nxt
            result[idx] = score
        return result

    def _retain_betweenness(self, nodes, max_sample=500):
        if self.retain_mask is None:
            raise ValueError("'retain_betweenness' requires retain_mask.")
        G = self._to_nx()
        all_retain = self.retain_mask.nonzero(as_tuple=True)[0].cpu().tolist()
        if len(all_retain) > max_sample:
            rng = np.random.default_rng(seed=42)
            retain = rng.choice(all_retain, size=max_sample, replace=False).tolist()
        else:
            retain = all_retain
        bc = nx.betweenness_centrality_subset(G, sources=retain, targets=retain, normalized=True)
        return {idx: bc.get(idx, 0.0) for idx in self._ids(nodes)}

    def _class_boundary(self, nodes):
        if self.retain_mask is None:
            raise ValueError("'class_boundary' requires retain_mask.")
        ei     = self.data.edge_index
        labels = self.data.y
        r_set  = set(self.retain_mask.nonzero(as_tuple=True)[0].cpu().tolist())
        result = {}
        for idx in self._ids(nodes):
            y_v = labels[idx].item()
            nbr_mask = (ei[0] == idx) | (ei[1] == idx)
            nbrs = (
                set(ei[0, nbr_mask].cpu().tolist() + ei[1, nbr_mask].cpu().tolist())
                & r_set
            ) - {idx}
            if nbrs:
                same = sum(1 for u in nbrs if labels[u].item() == y_v)
                result[idx] = 1.0 - same / len(nbrs)
            else:
                result[idx] = 0.0
        return result


# ---------------------------------------------------------------------------
# CurriculumDesigner
# ---------------------------------------------------------------------------

class CurriculumDesigner:
    """
    Partition the forget set into K ordered curriculum stages.

    Nodes are first ranked by their complexity score (hard-to-easy by default),
    then split into stages with optional overlap between consecutive stages.
    """

    def __init__(self, forget_mask, data, complexity_metric='retain_coupling',
                 num_curricula=8, mode='overlapping', overlap_ratio=0.2,
                 curriculum_order='hard_to_easy',
                 model=None, retain_mask=None, device='cpu',
                 num_layers=2, hop_decay=0.5):

        self.forget_mask    = forget_mask
        self.num_curricula  = num_curricula
        self.mode           = mode
        self.overlap_ratio  = overlap_ratio
        self.order          = curriculum_order

        forget_indices = forget_mask.nonzero(as_tuple=True)[0].cpu().numpy()

        # Clip num_curricula so every stage has >= 2 nodes (n_min = 2)
        n_min = 2
        if len(forget_indices) < self.num_curricula * n_min:
            self.num_curricula = max(1, len(forget_indices) // n_min)
            print(f"[CurriculumDesigner] Clipped stages to {self.num_curricula} "
                  f"(forget set too small).")

        calc = ComplexityCalculator(
            data, model=model, retain_mask=retain_mask,
            device=device, num_layers=num_layers, hop_decay=hop_decay,
        )
        scores = calc.calculate(forget_indices, complexity_metric)

        reverse = (curriculum_order == 'hard_to_easy')
        self._sorted = [node for node, _ in
                        sorted(scores.items(), key=lambda x: x[1], reverse=reverse)]

    def design_curricula(self):
        """Return a list of K boolean masks, one per curriculum stage."""
        if self.mode == 'non_overlapping':
            return self._non_overlapping(self._sorted)
        return self._overlapping(self._sorted)

    def _non_overlapping(self, sorted_idx):
        n = len(sorted_idx)
        chunk = n // self.num_curricula
        curricula = []
        for i in range(self.num_curricula):
            start = i * chunk
            end   = start + chunk if i < self.num_curricula - 1 else n
            mask  = torch.zeros_like(self.forget_mask)
            mask[sorted_idx[start:end]] = True
            curricula.append(mask)
        return curricula

    def _overlapping(self, sorted_idx):
        n    = len(sorted_idx)
        r    = self.overlap_ratio
        step = int(n / (self.num_curricula * (1 - r) + r))
        chunk = int(step / (1 - r))
        curricula = []
        for i in range(self.num_curricula):
            start = i * step
            if start >= n:
                break
            end  = min(start + chunk, n)
            mask = torch.zeros_like(self.forget_mask)
            mask[sorted_idx[start:end]] = True
            curricula.append(mask)
        return curricula
