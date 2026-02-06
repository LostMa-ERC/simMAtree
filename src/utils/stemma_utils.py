import networkx as nx


def leaves(graph: nx.DiGraph):
    """
    Return the terminal leaves of a tree

    Parameters
    ----------
    graph : nx.DiGraph
        Tree (directed acyclic graph)

    Returns
    -------
    List
        List of leaf nodes
    """
    return [node for node in graph.nodes() if graph.out_degree(node) == 0]


def root(graph: nx.DiGraph):
    """
    Return the root of a tree

    Parameters
    ----------
    graph : nx.DiGraph
        Tree (directed acyclic graph)

    Returns
    -------
    node or None
        Root node, or None if no root found
    """
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            return node
    return None


def generate_stemma(complete_tree: nx.DiGraph) -> nx.DiGraph:
    """
    Generate clean stemma from complete tree

    Removes:
    - Dead terminal leaves
    - Non-branching dead nodes

    Parameters
    ----------
    complete_tree : nx.DiGraph
        Complete tree with 'state' node attributes

    Returns
    -------
    nx.DiGraph
        Cleaned stemma
    """
    stemma = nx.DiGraph(complete_tree)
    living = {n: stemma.nodes[n]["state"] for n in stemma.nodes()}

    # Recursively remove dead terminal leaves
    terminal_dead = [n for n in leaves(stemma) if not living[n]]
    while terminal_dead:
        for node in terminal_dead:
            stemma.remove_node(node)
        living = {n: living[n] for n in stemma.nodes() if n in living}
        terminal_dead = [n for n in leaves(stemma) if not living[n]]

    # Remove non-branching dead nodes
    unwanted = [
        n
        for n in list(stemma.nodes())
        if n in living
        and not living[n]
        and stemma.out_degree(n) == 1
        and stemma.in_degree(n) == 1
    ]

    while unwanted:
        for node in unwanted:
            preds = list(stemma.predecessors(node))
            succs = list(stemma.successors(node))
            if preds and succs:
                stemma.add_edge(preds[0], succs[0])
            stemma.remove_node(node)

        living = {n: living[n] for n in stemma.nodes() if n in living}
        unwanted = [
            n
            for n in list(stemma.nodes())
            if n in living
            and not living[n]
            and stemma.out_degree(n) == 1
            and stemma.in_degree(n) == 1
        ]

    # Remove root if it's dead and has only one child
    root_node = root(stemma)
    if (
        root_node
        and root_node in living
        and not living[root_node]
        and stemma.out_degree(root_node) == 1
    ):
        stemma.remove_node(root_node)

    return stemma
