import networkx as nx
import re
import plotly.graph_objects as go
import streamlit as st


def rdf_to_nx_graph(rdf_graph):
    nx_graph = nx.Graph()

    for subj, pred, obj in rdf_graph:
        nx_graph.add_node(subj)
        nx_graph.add_node(obj)
        nx_graph.add_edge(subj, obj, type=pred)

    return nx_graph


def search_in_rdf_graph(graph, keywords, client, rdf_text, ontologie=None):
    result = []
    fin = False
    i = 0
    syn = False
    if len(keywords) > 0:
        keyword = keywords[0]
        while not fin:
            selected_elements = {'keyword': keyword,
                                 'nodes_kw': set(),
                                 'predicates_kw': []}
            for subject, predicate, obj in graph:
                # Rechercher le mot-clé dans le sujet (nœud ressource)
                if keyword.lower() in str(subject).lower():
                    selected_elements['nodes_kw'].add(subject)

                # Rechercher le mot-clé dans la propriété (prédicat)
                if keyword.lower() in str(predicate).lower():
                    selected_elements['predicates_kw'].append(
                        {"predicate": predicate, "node_src": subject,
                         "node_dest": obj})
                # Rechercher le mot-clé dans la valeur
                if keyword.lower() in str(obj).lower():
                    selected_elements['nodes_kw'].add(obj)
            if len(selected_elements['nodes_kw']) > 0 or len(selected_elements['predicates_kw']) > 0 or syn:
                result.append(selected_elements)
                i += 1
                if i >= len(keywords):
                    fin = True
                else:
                    keyword = keywords[i]
                    syn = False
            else:
                keyword = synonyme(keyword, client, text=rdf_text, ontology=ontologie)
                syn = True
                if keyword is None:
                    result.append(selected_elements)
                    i += 1
                    if i >= len(keywords):
                        fin = True
                    else:
                        keyword = keywords[i]
                        syn = False
    return result


def synonyme(keyword, client, text=None, ontology=None):
    '''
    assistant = "you use only the following context to answer.\nHere's an rdf graph in xml format: "+ str(text)

    prompt_template = """
        give me a keyword synonyme of {} so that when i search for node, relation, literal coresponding to that keyword in the rdf graph given it return me some results.
        return me only the keyword synonyme not rdf elements.
        the keyword must correspond to an element in rdf graph.
        """.format(keyword)


    messages = [{'role': 'system', 'content': 'print only the synonyme word.'},
                {"role": "user", "content": prompt_template}, 
                {"role": "assistant", "content": assistant},
                ]'''

    if ontology is not None:
        context = ontology
    else:
        context = text
    prompt_template = """
            use only the following context to answer.
            Here's an rdf graph in xml format: {}
            give me a keyword synonyme of {} so that when i search for node, relation, literal coresponding to that keyword in the rdf graph given it return me some results.
            the synonym must be a node name or literal or relation type.
            return me only the keyword synonyme not rdf elements.
            the keyword must correspond to an element in rdf graph.
            return #synonym
            """.format(str(context), keyword)
    messages = [{"role": "user", "content": prompt_template}]
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=messages,
        temperature=0
    )
    # matches = re.findall(r'"([^"]*)"', response.choices[0].message.content)
    matches = re.findall(r'#([^"]*)', response.choices[0].message.content)

    result = None
    if len(matches) >= 1:
        result = matches[0]
    return result


def graph_mots_cle(predicates, nodes):
    kw_nx = nx.Graph()
    for node in nodes:
        kw_nx.add_node(node)
    for pred in predicates:
        kw_nx.add_edge(pred['node_src'], pred['node_dest'],
                       type=pred['predicate'])
    return kw_nx


def is_node_pred(node, predicates):
    result = None
    for pred in predicates['predicates_kw']:
        if str(pred['node_dest']) == str(node) or str(pred['node_src']) == str(node):
            result = pred
            break
    return result


def graph_construct(graph_kw, nx_graph):
    nodes_done = set()
    predicates_res = []
    nodes_to_do = set(graph_kw['nodes_kw'])

    for pred in graph_kw['predicates_kw']:
        nodes_to_do.add(pred['node_src'])
        nodes_to_do.add(pred['node_dest'])

    node = list(nodes_to_do)[0]
    stop = False
    last_node = False
    while not stop:
        cas = 0
        pred = is_node_pred(node, graph_kw)
        if pred is not None:
            node = pred['node_src']
            node2 = pred['node_dest']
            cas = 1
            if node in nodes_done:
                node = node2
                cas = 0
                predicates_res.append(pred)
                # nodes_done.add(node)
            elif node2 in nodes_done:
                cas = 0
                predicates_res.append(pred)
                # nodes_done.add(node)
        if cas == 0:
            shortest_path = []
            for nd in nodes_to_do:
                if str(nd) != str(node) and (nd not in nodes_done or last_node):
                    sp = nx.shortest_path(nx_graph, source=node, target=nd)
                    if len(shortest_path) == 0:
                        shortest_path = sp
                    elif len(sp) < len(shortest_path):
                        shortest_path = sp
            nodes_done.add(node)
            if len(shortest_path) > 0:
                for i in range(len(shortest_path)-1):
                    predicates_res.append({
                            'predicate': nx_graph[shortest_path[i]][shortest_path[i+1]]['type'],
                            'node_src': shortest_path[i],
                            'node_dest': shortest_path[i+1],
                        })
                node = shortest_path[-1]
        elif cas == 1:
            shortest_path = []
            for nd in nodes_to_do:
                if str(nd) != str(node) and str(nd) != str(node2) and (nd not in nodes_done or last_node):
                    sp = nx.shortest_path(nx_graph, source=node, target=nd)
                    sp2 = nx.shortest_path(nx_graph, source=node2, target=nd)
                    if len(shortest_path) == 0:
                        if len(sp) < len(sp2):
                            shortest_path = sp
                        else:
                            shortest_path = sp2
                    elif len(sp) < len(shortest_path):
                        shortest_path = sp
                    elif len(sp2) < len(shortest_path):
                        shortest_path = sp2
            nodes_done.add(node)
            nodes_done.add(node2)
            predicates_res.append({
                        'predicate': nx_graph[node][node2]['type'],
                        'node_src': node,
                        'node_dest': node2,
                    })
            if len(shortest_path) > 0:
                for i in range(len(shortest_path)-1):
                    predicates_res.append({
                            'predicate': nx_graph[shortest_path[i]][shortest_path[i+1]]['type'],
                            'node_src': shortest_path[i],
                            'node_dest': shortest_path[i+1],
                        })
                node = shortest_path[-1]
        if node in nodes_done:
            stop = True
            for nd in nodes_to_do:
                if nd not in nodes_done:
                    node = nd
                    stop = False
                    break
        if len(nodes_done) == len(nodes_to_do)-1:
            last_node = True

    node_res = set(nodes_to_do)
    for predicate in predicates_res:
        node_res.add(predicate['node_src'])
        node_res.add(predicate['node_dest'])
    for node in node_res:
        for node2 in node_res:
            if nx_graph.has_edge(node, node2):
                predicates_res.append({
                            'predicate': nx_graph[node][node2]['type'],
                            'node_src': node,
                            'node_dest': node2,
                        })
    return predicates_res, list(node_res)


def produit_cartesien(graphs_kw):
    list_dicts = []
    for keyword in graphs_kw:
        if len(keyword['nodes_kw']) > 0:
            list_dicts.append(list(keyword['nodes_kw']))
        elif len(keyword['predicates_kw']) > 0:
            list_dicts.append(keyword['predicates_kw'])

    combinations = []
    if len(list_dicts)==0: return []

    list_ind = [0 for i in range(len(list_dicts))]
    stop = False
    while not stop:
        selected = {'nodes_kw': [],
                    'predicates_kw': []}
        for j in range(len(list_dicts)):
            if isinstance(list_dicts[j][list_ind[j]], dict):
                selected['predicates_kw'].append(list_dicts[j][list_ind[j]])
            else:
                selected['nodes_kw'].append(list_dicts[j][list_ind[j]])

        list_ind[0] = list_ind[0] + 1
        combinations.append(selected)
        for i in range(len(list_dicts)):
            if list_ind[i] >= len(list_dicts[i]):
                list_ind[i] = 0
                if i < len(list_dicts)-1:
                    list_ind[i+1] += 1
                else:
                    stop = True
                    break
    return combinations


def ranking(nx_kw, nx_result):
    N = len(nx_result.nodes()) + len(nx_result.edges())
    A = len(nx_kw.nodes()) + len(nx_kw.edges())
    return A/N

def draw_graph_plotly(G):

    pos = nx.spring_layout(G)

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text_brieve = []
    node_text = []
    for i in G.nodes():
        node_text_brieve.append(str(i).split(('/'))[-1])
        node_text.append(str(i))
    edge_text_brieve = [str(edge[2]['type']).split('/')[-1][:10]
                        for edge in G.edges(data=True)]
    edge_text = [str(edge[2]['type']) for edge in G.edges(data=True)]

    # Create edges
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Add edge weights
    edge_trace.text = edge_text_brieve

    # Create middle point coordinates for edge text
    edge_text_trace = go.Scatter(
        x=[(pos[edge[0]][0] + pos[edge[1]][0]) / 2 for edge in G.edges()],
        y=[(pos[edge[0]][1] + pos[edge[1]][1]) / 2 for edge in G.edges()],
        text=edge_trace.text,
        mode='text',
        hoverinfo='text',
        hovertext=edge_text,
        textposition='top center'
    )

    # Create nodes
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text_brieve,
        hovertext=node_text,
        textposition='bottom center',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)
        )
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, edge_text_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False,
                                   showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False,
                                   showticklabels=False)
                    ))

    # Show the plot
    st.plotly_chart(fig)


def draw_hist(data):

    sorted_data = sorted(data, key=lambda x: x['rank'], reverse=True)

    top_10_combinations = sorted_data[:10]

    combinations = [item['combinaison'] for item in top_10_combinations]
    ranks = [item['rank'] for item in top_10_combinations]

    fig = go.Figure([go.Bar(x=combinations, y=ranks)])

    fig.update_layout(title='Top 10 Combinations by Rank',
                      xaxis_title='Combination',
                      yaxis_title='Rank')

    st.plotly_chart(fig)