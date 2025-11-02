"""
Unlearning Methods for Graph Neural Networks
Includes: Retrain, Gradient Ascent, Curriculum Unlearning, NPO, and Full Method
"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import networkx as nx
from copy import deepcopy

from learning import create_model, train_homogeneous, train_knowledge_graph
from data import get_forget_retain_split


class ComplexityCalculator:
    """Calculate graph complexity metrics for curriculum design"""
    
    def __init__(self, data, is_kg=False):
        self.data = data
        self.is_kg = is_kg
        
    def calculate_complexity(self, nodes, metric='degree'):
        """
        Calculate complexity scores for nodes
        
        Args:
            nodes: List of node indices
            metric: Complexity metric (degree, betweenness, pagerank, clustering, eigenvector)
            
        Returns:
            complexity_scores: Dictionary mapping node to complexity score
        """
        if metric == 'degree':
            return self._calculate_degree(nodes)
        elif metric == 'betweenness':
            return self._calculate_betweenness(nodes)
        elif metric == 'pagerank':
            return self._calculate_pagerank(nodes)
        elif metric == 'clustering':
            return self._calculate_clustering(nodes)
        elif metric == 'eigenvector':
            return self._calculate_eigenvector(nodes)
        else:
            raise ValueError(f"Unknown complexity metric: {metric}")
    
    def _calculate_degree(self, nodes):
        """Calculate node degree"""
        if self.is_kg:
            edge_index = self.data['train']['edge_index']
        else:
            edge_index = self.data.edge_index
        
        degrees = {}
        for node in nodes:
            node_item = node.item() if torch.is_tensor(node) else node
            # Count both in and out degrees
            degree = ((edge_index[0] == node_item).sum() + (edge_index[1] == node_item).sum()).item()
            degrees[node_item] = degree
        
        return degrees
    
    def _calculate_betweenness(self, nodes):
        """Calculate betweenness centrality"""
        G = self._to_networkx()
        betweenness = nx.betweenness_centrality(G)
        
        result = {}
        for node in nodes:
            node_item = node.item() if torch.is_tensor(node) else node
            result[node_item] = betweenness.get(node_item, 0.0)
        
        return result
    
    def _calculate_pagerank(self, nodes):
        """Calculate PageRank"""
        G = self._to_networkx()
        pagerank = nx.pagerank(G)
        
        result = {}
        for node in nodes:
            node_item = node.item() if torch.is_tensor(node) else node
            result[node_item] = pagerank.get(node_item, 0.0)
        
        return result
    
    def _calculate_clustering(self, nodes):
        """Calculate clustering coefficient"""
        G = self._to_networkx()
        clustering = nx.clustering(G)
        
        result = {}
        for node in nodes:
            node_item = node.item() if torch.is_tensor(node) else node
            result[node_item] = clustering.get(node_item, 0.0)
        
        return result
    
    def _calculate_eigenvector(self, nodes):
        """Calculate eigenvector centrality"""
        G = self._to_networkx()
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            # If doesn't converge, use degree as fallback
            eigenvector = {node: G.degree(node) for node in G.nodes()}
        
        result = {}
        for node in nodes:
            node_item = node.item() if torch.is_tensor(node) else node
            result[node_item] = eigenvector.get(node_item, 0.0)
        
        return result
    
    def _to_networkx(self):
        """Convert PyG data to NetworkX graph"""
        if self.is_kg:
            edge_index = self.data['train']['edge_index'].cpu().numpy()
        else:
            edge_index = self.data.edge_index.cpu().numpy()
        
        G = nx.Graph()
        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_edges_from(edges)
        
        return G


class CurriculumDesigner:
    """Design curriculum for unlearning"""
    
    def __init__(self, forget_mask, data, is_kg, complexity_metric='degree', 
                 num_curricula=4, mode='non_overlapping', overlap_ratio=0.2):
        self.forget_mask = forget_mask
        self.data = data
        self.is_kg = is_kg
        self.complexity_metric = complexity_metric
        self.num_curricula = num_curricula
        self.mode = mode
        self.overlap_ratio = overlap_ratio
        
        # Get forget nodes/edges
        self.forget_indices = forget_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        
        # Calculate complexity
        self.calculator = ComplexityCalculator(data, is_kg)
        self.complexity_scores = self.calculator.calculate_complexity(
            self.forget_indices, metric=complexity_metric
        )
    
    def design_curricula(self):
        """
        Design curricula from simple to complex
        
        Returns:
            curricula: List of curricula, each containing node/edge indices
        """
        # Sort nodes by complexity (ascending - simple to complex)
        sorted_nodes = sorted(self.complexity_scores.items(), key=lambda x: x[1])
        sorted_indices = [node for node, _ in sorted_nodes]
        
        if self.mode == 'non_overlapping':
            return self._non_overlapping_split(sorted_indices)
        else:
            return self._overlapping_split(sorted_indices)
    
    def _non_overlapping_split(self, sorted_indices):
        """Split into non-overlapping curricula"""
        n = len(sorted_indices)
        chunk_size = n // self.num_curricula
        
        curricula = []
        for i in range(self.num_curricula):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_curricula - 1 else n
            
            curriculum_nodes = sorted_indices[start:end]
            
            # Create mask
            mask = torch.zeros_like(self.forget_mask)
            mask[curriculum_nodes] = True
            
            curricula.append(mask)
        
        return curricula
    
    def _overlapping_split(self, sorted_indices):
        """Split into overlapping curricula"""
        n = len(sorted_indices)
        
        # Calculate step size with overlap
        step_size = int(n / (self.num_curricula * (1 - self.overlap_ratio) + self.overlap_ratio))
        chunk_size = int(step_size / (1 - self.overlap_ratio))
        
        curricula = []
        for i in range(self.num_curricula):
            start = int(i * step_size)
            end = min(start + chunk_size, n)
            
            if start >= n:
                break
            
            curriculum_nodes = sorted_indices[start:end]
            
            # Create mask
            mask = torch.zeros_like(self.forget_mask)
            mask[curriculum_nodes] = True
            
            curricula.append(mask)
        
        return curricula


def retrain_baseline(args, data, is_kg, retain_mask):
    """
    Baseline 1: Retrain from scratch on retain set only
    
    Args:
        args: Arguments
        data: Graph data
        is_kg: Whether it's a knowledge graph
        retain_mask: Mask for retain set
        
    Returns:
        model: Retrained model
    """
    print("\n" + "="*60)
    print("Baseline: Retrain from Scratch")
    print("="*60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create new model
    model = create_model(args, data, is_kg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_lr, weight_decay=args.weight_decay)
    
    # Temporarily modify training mask to only include retain set
    if not is_kg:
        original_train_mask = data.train_mask.clone()
        data.train_mask = retain_mask
        data = data.to(device)
    
    # Train on retain set only
    pbar = tqdm(range(args.learning_epochs), desc="Retraining")
    for epoch in pbar:
        if is_kg:
            # Filter to retain edges only
            retain_data = {
                'num_entities': data['num_entities'],
                'num_relations': data['num_relations'],
                'train': {
                    'edge_index': data['train']['edge_index'][:, retain_mask],
                    'edge_type': data['train']['edge_type'][retain_mask],
                    'num_edges': retain_mask.sum().item()
                },
                'entity_ids': data['entity_ids']
            }
            loss = train_knowledge_graph(model, retain_data, optimizer, device)
        else:
            loss = train_homogeneous(model, data, optimizer, device)
        
        pbar.set_postfix({'Loss': f'{loss:.4f}'})
    
    # Restore original mask
    if not is_kg:
        data.train_mask = original_train_mask
    
    print("Retrain completed!")
    return model


def gradient_ascent_baseline(args, model, data, is_kg, forget_mask, retain_mask):
    """
    Baseline 2: Gradient Ascent
    
    Args:
        args: Arguments
        model: Trained model
        data: Graph data
        is_kg: Whether it's a knowledge graph
        forget_mask: Mask for forget set
        retain_mask: Mask for retain set
        
    Returns:
        model: Unlearned model
    """
    print("\n" + "="*60)
    print("Baseline: Gradient Ascent")
    print("="*60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if not is_kg:
        data = data.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.unlearn_lr)
    
    pbar = tqdm(range(args.unlearn_epochs), desc="Gradient Ascent")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        if is_kg:
            # Unlearn forget edges
            edge_index = data['train']['edge_index'].to(device)
            edge_type = data['train']['edge_type'].to(device)
            entity_ids = data['entity_ids'].to(device)
            
            forget_indices = forget_mask.nonzero(as_tuple=True)[0]
            if len(forget_indices) > 0:
                batch_size = min(len(forget_indices), 512)
                sample_idx = forget_indices[torch.randperm(len(forget_indices))[:batch_size]]
                
                head_idx = edge_index[0, sample_idx]
                tail_idx = edge_index[1, sample_idx]
                rel_idx = edge_type[sample_idx]
                
                # Gradient ascent: maximize loss (minimize negative loss)
                scores = model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
                loss = -scores.mean()  # Negative to maximize
                
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        else:
            # Unlearn forget nodes
            out = model(data.x, data.edge_index)
            
            # Gradient ascent: maximize loss on forget set
            loss = -F.cross_entropy(out[forget_mask], data.y[forget_mask])
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    print("Gradient Ascent completed!")
    return model


def curriculum_gradient_ascent(args, model, data, is_kg, forget_mask, retain_mask):
    """
    Variant 1: Gradient Ascent + Curriculum Unlearning (Step 1)
    
    Args:
        args: Arguments
        model: Trained model
        data: Graph data
        is_kg: Whether it's a knowledge graph
        forget_mask: Mask for forget set
        retain_mask: Mask for retain set
        
    Returns:
        model: Unlearned model
    """
    print("\n" + "="*60)
    print("Variant 1: Curriculum Gradient Ascent")
    print(f"Curricula: {args.num_curricula}, Metric: {args.complexity_metric}, Mode: {args.curriculum_mode}")
    print("="*60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if not is_kg:
        data = data.to(device)
    
    # Design curricula
    designer = CurriculumDesigner(
        forget_mask=forget_mask,
        data=data,
        is_kg=is_kg,
        complexity_metric=args.complexity_metric,
        num_curricula=args.num_curricula,
        mode=args.curriculum_mode,
        overlap_ratio=args.overlap_ratio
    )
    curricula = designer.design_curricula()
    
    print(f"Designed {len(curricula)} curricula")
    for i, curriculum in enumerate(curricula):
        print(f"  Curriculum {i+1}: {curriculum.sum().item()} nodes/edges")
    
    # Unlearn curriculum by curriculum
    epochs_per_curriculum = args.unlearn_epochs // len(curricula)
    
    for curriculum_idx, curriculum_mask in enumerate(curricula):
        print(f"\nUnlearning Curriculum {curriculum_idx + 1}/{len(curricula)}...")
        
        optimizer = optim.Adam(model.parameters(), lr=args.unlearn_lr)
        
        pbar = tqdm(range(epochs_per_curriculum), desc=f"Curriculum {curriculum_idx+1}")
        for epoch in pbar:
            model.train()
            optimizer.zero_grad()
            
            if is_kg:
                edge_index = data['train']['edge_index'].to(device)
                edge_type = data['train']['edge_type'].to(device)
                entity_ids = data['entity_ids'].to(device)
                
                curriculum_indices = curriculum_mask.nonzero(as_tuple=True)[0]
                if len(curriculum_indices) > 0:
                    batch_size = min(len(curriculum_indices), 512)
                    sample_idx = curriculum_indices[torch.randperm(len(curriculum_indices))[:batch_size]]
                    
                    head_idx = edge_index[0, sample_idx]
                    tail_idx = edge_index[1, sample_idx]
                    rel_idx = edge_type[sample_idx]
                    
                    scores = model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
                    loss = -scores.mean()
                    
                    loss.backward()
                    optimizer.step()
                    
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            else:
                out = model(data.x, data.edge_index)
                loss = -F.cross_entropy(out[curriculum_mask], data.y[curriculum_mask])
                
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    print("Curriculum Gradient Ascent completed!")
    return model


def npo_gradient_ascent(args, model, data, is_kg, forget_mask, retain_mask):
    """
    Variant 2: Gradient Ascent + NPO (Step 2)
    NPO (Negative Preference Optimization) modifies the loss function
    
    Args:
        args: Arguments
        model: Trained model
        data: Graph data
        is_kg: Whether it's a knowledge graph
        forget_mask: Mask for forget set
        retain_mask: Mask for retain set
        
    Returns:
        model: Unlearned model
    """
    print("\n" + "="*60)
    print("Variant 2: NPO Gradient Ascent")
    print(f"Beta: {args.npo_beta}, Lambda: {args.npo_lambda}, Temperature: {args.npo_temperature}")
    print("="*60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if not is_kg:
        data = data.to(device)
    
    # Clone model for reference (frozen)
    reference_model = deepcopy(model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(model.parameters(), lr=args.unlearn_lr)
    
    pbar = tqdm(range(args.unlearn_epochs), desc="NPO")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        if is_kg:
            edge_index = data['train']['edge_index'].to(device)
            edge_type = data['train']['edge_type'].to(device)
            entity_ids = data['entity_ids'].to(device)
            
            # Forget loss (NPO-modified)
            forget_indices = forget_mask.nonzero(as_tuple=True)[0]
            if len(forget_indices) > 0:
                batch_size = min(len(forget_indices), 256)
                forget_sample = forget_indices[torch.randperm(len(forget_indices))[:batch_size]]
                
                head_idx = edge_index[0, forget_sample]
                tail_idx = edge_index[1, forget_sample]
                rel_idx = edge_type[forget_sample]
                
                # Current model scores
                scores_current = model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
                
                # Reference model scores
                with torch.no_grad():
                    scores_ref = reference_model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
                
                # NPO loss: encourage current model to deviate from reference on forget set
                forget_loss = -torch.mean(
                    F.logsigmoid(args.npo_beta * (scores_ref - scores_current) / args.npo_temperature)
                )
            else:
                forget_loss = 0.0
            
            # Retain loss (preserve utility)
            retain_indices = retain_mask.nonzero(as_tuple=True)[0]
            if len(retain_indices) > 0:
                batch_size = min(len(retain_indices), 256)
                retain_sample = retain_indices[torch.randperm(len(retain_indices))[:batch_size]]
                
                head_idx = edge_index[0, retain_sample]
                tail_idx = edge_index[1, retain_sample]
                rel_idx = edge_type[retain_sample]
                
                scores_current = model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
                
                # Generate negative samples
                neg_tail = torch.randint(0, data['num_entities'], (batch_size,), device=device)
                neg_scores = model.predict_link(entity_ids, edge_index, edge_type, head_idx, neg_tail, rel_idx)
                
                retain_loss = F.margin_ranking_loss(
                    scores_current, neg_scores,
                    torch.ones(batch_size, device=device),
                    margin=1.0
                )
            else:
                retain_loss = 0.0
            
            # Combined loss
            loss = args.npo_lambda * forget_loss + (1 - args.npo_lambda) * retain_loss
            
        else:
            out = model(data.x, data.edge_index)
            
            # Forget loss (NPO-modified)
            if forget_mask.sum() > 0:
                # Current model predictions
                logits_current = out[forget_mask]
                
                # Reference model predictions
                with torch.no_grad():
                    logits_ref = reference_model(data.x, data.edge_index)[forget_mask]
                
                # NPO loss: maximize difference from reference
                log_probs_current = F.log_softmax(logits_current / args.npo_temperature, dim=-1)
                log_probs_ref = F.log_softmax(logits_ref / args.npo_temperature, dim=-1)
                
                # KL divergence from reference (we want to maximize this)
                forget_loss = -args.npo_beta * F.kl_div(log_probs_current, log_probs_ref.detach(), 
                                                       reduction='batchmean', log_target=True)
            else:
                forget_loss = 0.0
            
            # Retain loss (preserve utility)
            if retain_mask.sum() > 0:
                retain_loss = F.cross_entropy(out[retain_mask], data.y[retain_mask])
            else:
                retain_loss = 0.0
            
            # Combined loss
            loss = args.npo_lambda * forget_loss + (1 - args.npo_lambda) * retain_loss
        
        if torch.is_tensor(loss):
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        else:
            pbar.set_postfix({'Loss': f'{loss:.4f}'})
    
    print("NPO Gradient Ascent completed!")
    return model


def full_method(args, model, data, is_kg, forget_mask, retain_mask):
    """
    Variant 3: Full Method (Gradient Ascent + Curriculum + NPO)
    
    Args:
        args: Arguments
        model: Trained model
        data: Graph data
        is_kg: Whether it's a knowledge graph
        forget_mask: Mask for forget set
        retain_mask: Mask for retain set
        
    Returns:
        model: Unlearned model
    """
    print("\n" + "="*60)
    print("Variant 3: Full Method (Curriculum + NPO)")
    print(f"Curricula: {args.num_curricula}, Metric: {args.complexity_metric}")
    print(f"Beta: {args.npo_beta}, Lambda: {args.npo_lambda}")
    print("="*60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if not is_kg:
        data = data.to(device)
    
    # Clone reference model
    reference_model = deepcopy(model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    # Design curricula
    designer = CurriculumDesigner(
        forget_mask=forget_mask,
        data=data,
        is_kg=is_kg,
        complexity_metric=args.complexity_metric,
        num_curricula=args.num_curricula,
        mode=args.curriculum_mode,
        overlap_ratio=args.overlap_ratio
    )
    curricula = designer.design_curricula()
    
    print(f"Designed {len(curricula)} curricula")
    
    # Unlearn curriculum by curriculum with NPO
    epochs_per_curriculum = args.unlearn_epochs // len(curricula)
    
    for curriculum_idx, curriculum_mask in enumerate(curricula):
        print(f"\nUnlearning Curriculum {curriculum_idx + 1}/{len(curricula)} with NPO...")
        
        optimizer = optim.Adam(model.parameters(), lr=args.unlearn_lr)
        
        pbar = tqdm(range(epochs_per_curriculum), desc=f"Curriculum {curriculum_idx+1}")
        for epoch in pbar:
            model.train()
            optimizer.zero_grad()
            
            if is_kg:
                edge_index = data['train']['edge_index'].to(device)
                edge_type = data['train']['edge_type'].to(device)
                entity_ids = data['entity_ids'].to(device)
                
                # NPO forget loss on current curriculum
                curriculum_indices = curriculum_mask.nonzero(as_tuple=True)[0]
                if len(curriculum_indices) > 0:
                    batch_size = min(len(curriculum_indices), 256)
                    sample_idx = curriculum_indices[torch.randperm(len(curriculum_indices))[:batch_size]]
                    
                    head_idx = edge_index[0, sample_idx]
                    tail_idx = edge_index[1, sample_idx]
                    rel_idx = edge_type[sample_idx]
                    
                    scores_current = model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
                    
                    with torch.no_grad():
                        scores_ref = reference_model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
                    
                    forget_loss = -torch.mean(
                        F.logsigmoid(args.npo_beta * (scores_ref - scores_current) / args.npo_temperature)
                    )
                else:
                    forget_loss = 0.0
                
                # Retain loss
                retain_indices = retain_mask.nonzero(as_tuple=True)[0]
                if len(retain_indices) > 0:
                    batch_size = min(len(retain_indices), 256)
                    retain_sample = retain_indices[torch.randperm(len(retain_indices))[:batch_size]]
                    
                    head_idx = edge_index[0, retain_sample]
                    tail_idx = edge_index[1, retain_sample]
                    rel_idx = edge_type[retain_sample]
                    
                    scores_current = model.predict_link(entity_ids, edge_index, edge_type, head_idx, tail_idx, rel_idx)
                    neg_tail = torch.randint(0, data['num_entities'], (batch_size,), device=device)
                    neg_scores = model.predict_link(entity_ids, edge_index, edge_type, head_idx, neg_tail, rel_idx)
                    
                    retain_loss = F.margin_ranking_loss(scores_current, neg_scores,
                                                       torch.ones(batch_size, device=device), margin=1.0)
                else:
                    retain_loss = 0.0
                
                loss = args.npo_lambda * forget_loss + (1 - args.npo_lambda) * retain_loss
                
            else:
                out = model(data.x, data.edge_index)
                
                # NPO forget loss on current curriculum
                if curriculum_mask.sum() > 0:
                    logits_current = out[curriculum_mask]
                    
                    with torch.no_grad():
                        logits_ref = reference_model(data.x, data.edge_index)[curriculum_mask]
                    
                    log_probs_current = F.log_softmax(logits_current / args.npo_temperature, dim=-1)
                    log_probs_ref = F.log_softmax(logits_ref / args.npo_temperature, dim=-1)
                    
                    forget_loss = -args.npo_beta * F.kl_div(log_probs_current, log_probs_ref.detach(),
                                                           reduction='batchmean', log_target=True)
                else:
                    forget_loss = 0.0
                
                # Retain loss
                if retain_mask.sum() > 0:
                    retain_loss = F.cross_entropy(out[retain_mask], data.y[retain_mask])
                else:
                    retain_loss = 0.0
                
                loss = args.npo_lambda * forget_loss + (1 - args.npo_lambda) * retain_loss
            
            if torch.is_tensor(loss):
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            else:
                pbar.set_postfix({'Loss': f'{loss:.4f}'})
    
    print("Full Method completed!")
    return model


def unlearn(args, model, data, is_kg, forget_mask, retain_mask):
    """
    Main unlearning function that dispatches to appropriate method
    
    Args:
        args: Arguments
        model: Trained model
        data: Graph data
        is_kg: Whether it's a knowledge graph
        forget_mask: Mask for forget set
        retain_mask: Mask for retain set
        
    Returns:
        model: Unlearned model
    """
    if args.unlearn_method == 'retrain':
        return retrain_baseline(args, data, is_kg, retain_mask)
    elif args.unlearn_method == 'gradient_ascent':
        return gradient_ascent_baseline(args, model, data, is_kg, forget_mask, retain_mask)
    elif args.unlearn_method == 'curriculum_ga':
        return curriculum_gradient_ascent(args, model, data, is_kg, forget_mask, retain_mask)
    elif args.unlearn_method == 'npo_ga':
        return npo_gradient_ascent(args, model, data, is_kg, forget_mask, retain_mask)
    elif args.unlearn_method == 'full_method':
        return full_method(args, model, data, is_kg, forget_mask, retain_mask)
    else:
        raise ValueError(f"Unknown unlearning method: {args.unlearn_method}")



