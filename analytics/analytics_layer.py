"""
analytics/analytics_layer.py
-----------------------------
Analytics for Semantic Knowledge Graph:
- Vote dispersion
- Expertise correlation
- Question-level statistics
- Response times
- Graph metrics
"""
import networkx as nx
from datetime import datetime
from collections import defaultdict, Counter
from statistics import pstdev, mean
from typing import Dict, Any


def compute_response_times(annotations: dict) -> dict:
    """
    Compute response-time analytics for questions and answers.
    Returns per-theme and per-expert averages, and lists unanswered questions.
    """
    theme_times = {}
    expert_times = {}
    unanswered = []
    
    questions = annotations.get('questions', {})
    answers = annotations.get('answers', {})
    
    # Index answers by question_id
    answers_by_question = defaultdict(list)
    for aid, ans in answers.items():
        ans_data = ans if isinstance(ans, dict) else ans.to_dict()
        qid = ans_data.get('metadata', {}).get('question_id')
        if qid:
            answers_by_question[qid].append(ans_data)
    
    for qid, qdata in questions.items():
        qdata = qdata if isinstance(qdata, dict) else qdata.to_dict()
        q_meta = qdata.get('metadata', {})
        q_time_str = q_meta.get('created_at')
        theme_id = q_meta.get('theme_id')
        q_time = None
        
        # Parse question timestamp
        if q_time_str:
            try:
                q_time = datetime.fromisoformat(q_time_str)
            except Exception:
                pass
        
        q_answers = answers_by_question.get(qid, [])
        if not q_answers:
            unanswered.append(qid)
            continue
        
        # Calculate response times
        response_times = []
        for ans in q_answers:
            a_meta = ans.get('metadata', {})
            a_time_str = a_meta.get('created_at')
            expert_id = a_meta.get('expert_id')
            if not a_time_str or not q_time:
                continue
            try:
                a_time = datetime.fromisoformat(a_time_str)
                diff_hours = (a_time - q_time).total_seconds() / 3600
                response_times.append(diff_hours)
                if expert_id:
                    expert_times.setdefault(expert_id, []).append(diff_hours)
            except Exception:
                continue
        
        if response_times:
            avg_time = mean(response_times)
            if theme_id:
                theme_times.setdefault(theme_id, []).append(avg_time)
    
    # Aggregate theme and expert averages
    theme_avg = {tid: round(mean(times), 2) for tid, times in theme_times.items()}
    expert_avg = {eid: round(mean(times), 2) for eid, times in expert_times.items()}
    
    return {
        'theme_avg_response_hours': theme_avg,
        'expert_avg_response_hours': expert_avg,
        'unanswered_questions': unanswered,
        'total_unanswered': len(unanswered)
    }


def compute_question_trends(annotations: dict) -> dict:
    """
    Compute monthly question creation trends (overall + per theme).
    """
    questions = annotations.get('questions', {})
    overall = Counter()
    per_theme = defaultdict(Counter)
    
    for qid, qdata in questions.items():
        qdata = qdata if isinstance(qdata, dict) else qdata.to_dict()
        meta = qdata.get('metadata', {})
        created = meta.get('created_at')
        theme_id = meta.get('theme_id')
        if not created:
            continue
        try:
            dt = datetime.fromisoformat(created)
            month_key = dt.strftime("%Y-%m")
            overall[month_key] += 1
            if theme_id:
                per_theme[theme_id][month_key] += 1
        except Exception:
            continue
    
    # Compute total questions per theme
    per_theme_total = {tid: sum(c.values()) for tid, c in per_theme.items()}
    
    return {
        'overall_trend': dict(overall),
        'per_theme_trend': {tid: dict(c) for tid, c in per_theme.items()},
        'per_theme_total': per_theme_total
    }


def compute_vote_dispersion(annotations: dict) -> dict:
    """
    For each question, compute vote dispersion and expertise correlation.
    """
    try:
        from scipy.stats import entropy
    except ImportError:
        print("⚠️  scipy not available - skipping entropy calculation")
        entropy = None
    
    question_vote_stats = {}
    
    answers = annotations.get('answers', {})
    votes = annotations.get('votes', {})
    experts = annotations.get('experts', {})
    
    # Convert to dicts if needed
    answers = {k: (v if isinstance(v, dict) else v.to_dict()) for k, v in answers.items()}
    votes = {k: (v if isinstance(v, dict) else v.to_dict()) for k, v in votes.items()}
    experts = {k: (v if isinstance(v, dict) else v.to_dict()) for k, v in experts.items()}
    
    # Group votes by answer_id
    votes_by_answer = defaultdict(list)
    for v_id, v in votes.items():
        a_id = v.get('answer_id')
        if a_id:
            votes_by_answer[a_id].append(v)
    
    for q_id, q_data in annotations.get('questions', {}).items():
        q_data = q_data if isinstance(q_data, dict) else q_data.to_dict()
        
        # Find answers for this question
        related_answers = [
            (a_id, a) for a_id, a in answers.items()
            if a.get('metadata', {}).get('question_id') == q_id
        ]
        
        all_vote_values = []
        for a_id, a in related_answers:
            vote_values = [
                1 if v.get('metadata', {}).get('vote_value', '').lower() in 
                    ['upvote', 'like', 'endorse', 'approve']
                else -1 
                for v in votes_by_answer.get(a_id, [])
            ]
            all_vote_values.extend(vote_values)
        
        # Dispersion metrics
        if all_vote_values:
            std_dev = pstdev(all_vote_values) if len(all_vote_values) > 1 else 0
            if entropy:
                value_counts = Counter(all_vote_values)
                probs = [c / len(all_vote_values) for c in value_counts.values()]
                vote_entropy = entropy(probs)
            else:
                vote_entropy = 0
        else:
            std_dev, vote_entropy = 0, 0
        
        # Expertise correlation
        expert_ids = [
            v.get('metadata', {}).get('expert_id')
            for a_id, a in related_answers
            for v in votes_by_answer.get(a_id, [])
            if v.get('metadata', {}).get('expert_id')
        ]
        expertise_areas = [
            experts.get(e_id, {}).get('metadata', {}).get('area_of_expertise')
            for e_id in expert_ids if e_id
        ]
        expertise_areas = [e for e in expertise_areas if e]
        
        expertise_corr = 0
        if len(expertise_areas) > 1:
            common_area = Counter(expertise_areas).most_common(1)[0][1]
            expertise_corr = common_area / len(expertise_areas)
        
        question_vote_stats[q_id] = {
            'vote_std_dev': round(std_dev, 3),
            'vote_entropy': round(vote_entropy, 3),
            'expertise_agreement': round(expertise_corr, 3),
            'num_votes': len(all_vote_values),
            'num_experts': len(expertise_areas)
        }
    
    return question_vote_stats


def compute_graph_metrics(annotations: dict) -> dict:
    """
    Build knowledge graph and compute centrality metrics.
    """
    G = nx.DiGraph()
    
    # Build graph from all relations
    for doc_type in ['themes', 'questions', 'answers', 'votes', 'experts', 'documents']:
        for doc_id, doc_data in annotations.get(doc_type, {}).items():
            doc_data = doc_data if isinstance(doc_data, dict) else doc_data.to_dict()
            relations = doc_data.get('relations', [])
            
            for rel in relations:
                if isinstance(rel, dict):
                    src = rel.get('source')
                    tgt = rel.get('target')
                    rel_type = rel.get('relation')
                else:
                    src = rel.source
                    tgt = rel.target
                    rel_type = rel.relation
                
                if src and tgt:
                    G.add_edge(src, tgt, relation=rel_type)
    
    # Compute metrics
    try:
        degree_centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        pagerank = nx.pagerank(G)
        
        # Community detection
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = [list(c) for c in greedy_modularity_communities(G.to_undirected())]
        except Exception:
            communities = []
        
        # Graph density and clustering
        density = nx.density(G)
        try:
            clustering = nx.average_clustering(G.to_undirected()) if G.number_of_nodes() > 0 else 0
        except Exception:
            clustering = 0
        
        metrics = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "degree_centrality": {k: round(v, 4) for k, v in list(degree_centrality.items())[:20]},
            "betweenness_centrality": {k: round(v, 4) for k, v in list(betweenness.items())[:20]},
            "pagerank": {k: round(v, 4) for k, v in list(pagerank.items())[:20]},
            "communities": [c[:10] for c in communities[:10]],  # Limit size
            "graph_density": round(density, 4),
            "average_clustering": round(clustering, 4),
        }
    
    except Exception as e:
        print(f"Warning: Graph metrics computation failed: {e}")
        metrics = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "error": str(e)
        }
    
    return {"KGQualityMetrics": metrics}


def compute_all_analytics(annotations: dict) -> dict:
    """Compute all analytics metrics."""
    print("\n📊 Computing analytics...")
    
    analytics = {}
    
    try:
        print("  - Response times...")
        analytics['response_times'] = compute_response_times(annotations)
    except Exception as e:
        print(f"  ⚠️  Response times failed: {e}")
        analytics['response_times'] = {}
    
    try:
        print("  - Question trends...")
        analytics['question_trends'] = compute_question_trends(annotations)
    except Exception as e:
        print(f"  ⚠️  Question trends failed: {e}")
        analytics['question_trends'] = {}
    
    try:
        print("  - Vote dispersion...")
        analytics['vote_dispersion'] = compute_vote_dispersion(annotations)
    except Exception as e:
        print(f"  ⚠️  Vote dispersion failed: {e}")
        analytics['vote_dispersion'] = {}
    
    try:
        print("  - Graph metrics...")
        analytics['graph_metrics'] = compute_graph_metrics(annotations)
    except Exception as e:
        print(f"  ⚠️  Graph metrics failed: {e}")
        analytics['graph_metrics'] = {}
    
    print("✓ Analytics complete")
    return analytics