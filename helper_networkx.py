import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

def print_graph_output(G, sep=' + '):
    print(f"\nGraph info: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Nodes and their properties:")
    for node, data in G.nodes(data=True):
        print(f"  {node}: properties={data}")
    print("\nEdges and their properties:")
    for edge1, edge2, data in G.edges(data=True):
        print(f"  {edge1}{sep}{edge2}: properties={data}")





def create_network_from_lists(sentences, weights, sep=' ', remove_list=None, filter_singletons=True):
    """
    Create a NetworkX graph from lists of sentences and weights.
    
    Parameters:
    -----------
    sentences : list
        List of sentences (strings)
    weights : list
        List of weights corresponding to each sentence
    sep : str, default=' '
        Separator string to split words in sentences
    remove_list : list, default=None
        List of words to filter out from sentences
    
    Returns:
    --------
    networkx.Graph
        Graph where nodes are words with weight and occurrence properties,
        and edges connect word pairs with weight and occurrence properties
    """
    
    if remove_list is None:
        remove_list = []
    
    # Check if sentences and weights have same length
    if len(sentences) != len(weights):
        raise ValueError("Sentences and weights lists must have the same length")
    
    # Convert remove_list to set for faster lookup
    remove_set = set(remove_list)
    
    # Initialize graph
    G = nx.Graph()
    
    # Process each sentence and weight pair
    for sentence, weight in zip(sentences, weights):
        
        # Split sentence and filter words
        words = [word.strip() for word in sentence.split(sep) if word.strip()]
        filtered_words = [word for word in words if word not in remove_set]
        
        # Skip if less than 2 words after filtering
        if filter_singletons and len(filtered_words) < 2:
            continue
        
        # Process nodes
        for word in filtered_words:
            if G.has_node(word):
                # Update existing node
                G.nodes[word]['weight'] += weight
                G.nodes[word]['occurrence'] += 1
            else:
                # Create new node
                G.add_node(word, weight=weight, occurrence=1)
        
        # Process edges (all pairs of words in the sentence)
        for word1, word2 in combinations(filtered_words, 2):
            if G.has_edge(word1, word2):
                # Update existing edge
                G.edges[word1, word2]['weight'] += weight
                G.edges[word1, word2]['occurrence'] += 1
            else:
                # Create new edge
                G.add_edge(word1, word2, weight=weight, occurrence=1)
    
    return G



def plot_weighted_graph(G, figsize=(12, 8), node_size_multiplier=300, edge_width_multiplier=2, 
                       layout='spring', node_color='lightblue', edge_color='gray', 
                       font_size=10, with_labels=True,
                       node_size_attribute='weight', edge_width_attribute='weight',
                       plot_title=None, node_label_attribute=None, edge_label_attribute=None,
                       **kwargs):
    """
    Plot a NetworkX graph with node sizes based on node attributes and edge thickness based on edge attributes.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to plot
    figsize : tuple, default=(12, 8)
        Figure size (width, height)
    node_size_multiplier : float, default=300
        Multiplier for node sizes (node_attribute_value * multiplier = size)
    edge_width_multiplier : float, default=2
        Multiplier for edge widths (edge_attribute_value * multiplier = width)
    layout : str, default='spring'
        Layout algorithm ('spring', 'circular', 'random', 'shell', 'kamada_kawai')
    node_color : str, default='lightblue'
        Color of the nodes
    edge_color : str, default='gray'
        Color of the edges
    font_size : int, default=10
        Font size for node labels
    with_labels : bool, default=True
        Whether to show node labels
    node_size_attribute : str, default='weight'
        Node attribute to use for determining node size
    edge_width_attribute : str, default='weight'
        Edge attribute to use for determining edge width
    plot_title : str, default=None
        Custom title for the plot. If None, auto-generates based on attributes
    node_label_attribute : str, default=None
        Node attribute to display in labels (format: "node_name\\n(attribute_value)")
    edge_label_attribute : str, default=None
        Edge attribute to display as edge labels
    """
    
    plt.figure(figsize=figsize)
    
    # Choose layout
    layout_functions = {
        'spring': nx.spring_layout,
        'circular': nx.circular_layout,
        'random': nx.random_layout,
        'shell': nx.shell_layout,
        'kamada_kawai': nx.kamada_kawai_layout
    }
    
    if layout in layout_functions:
        pos = layout_functions[layout](G)
    else:
        pos = nx.spring_layout(G,**kwargs)
    
    # Get node attribute values and sizes
    node_values = [G.nodes[node].get(node_size_attribute, 1) for node in G.nodes()]
    node_sizes = [value * node_size_multiplier for value in node_values]
    
    # Get edge attribute values and widths
    edge_values = [G.edges[edge].get(edge_width_attribute, 1) for edge in G.edges()]
    edge_widths = [value * edge_width_multiplier for value in edge_values]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_color, alpha=0.6)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color, 
                          alpha=0.8, edgecolors='black', linewidths=1)
    
    # Draw labels
    if with_labels:
        if node_label_attribute:
            # Create custom labels with node name + attribute value
            labels = {}
            for node in G.nodes():
                attr_value = G.nodes[node].get(node_label_attribute, 'N/A')
                labels[node] = f"{node}\n({attr_value})"
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size)
        else:
            # Use default node names
            nx.draw_networkx_labels(G, pos, font_size=font_size)
    
    # Draw edge labels if specified
    if edge_label_attribute:
        edge_labels = {}
        for edge in G.edges():
            attr_value = G.edges[edge].get(edge_label_attribute, 'N/A')
            edge_labels[edge] = f"{attr_value}"
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=font_size-2)
    
    # Set title
    if plot_title:
        title = plot_title
    else:
        title = f"Network Graph\n(Node size = {node_size_attribute}, Edge thickness = {edge_width_attribute})"
    
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()




def export_graph_to_graphml(graph: nx.Graph, filepath: str):
    """
    Exports a NetworkX graph to a GraphML file, preserving all node and edge attributes.

    If a node does not have a 'label' attribute, it assigns the node's identifier as the label.

    Parameters:
        graph (nx.Graph): The NetworkX graph to export.
        filepath (str): The output file path (should end in .graphml).
    """
    # Add 'label' attribute only if not already present
    for node in graph.nodes():
        if 'label' not in graph.nodes[node]:
            graph.nodes[node]['label'] = str(node)
    
    # Ensure all attribute values are GraphML-compatible (no lists, dicts, etc.)
    for n, attrs in graph.nodes(data=True):
        for k, v in attrs.items():
            if isinstance(v, (list, dict, set, tuple)):
                graph.nodes[n][k] = str(v)

    for u, v, attrs in graph.edges(data=True):
        for k, val in attrs.items():
            if isinstance(val, (list, dict, set, tuple)):
                graph.edges[u, v][k] = str(val)

    # Write to GraphML
    nx.write_graphml(graph, filepath)