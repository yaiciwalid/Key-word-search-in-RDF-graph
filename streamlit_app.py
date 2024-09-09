import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from rdflib import Graph
from xml.dom import minidom
import plotly.graph_objects as go
from utils import rdf_to_nx_graph, search_in_rdf_graph, graph_mots_cle, \
    graph_construct, produit_cartesien, ranking
import os
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = ""

client = OpenAI()


if 'graph1' not in st.session_state:
    st.session_state.graph1 = None

if 'graph2' not in st.session_state:
    st.session_state.graph2 = None

if 'graph3' not in st.session_state:
    st.session_state.graph3 = None
if 'produit_cartesien' not in st.session_state:
    st.session_state.produit_cartesien = []

if 'kw_sub_graphs' not in st.session_state:
    st.session_state.kw_sub_graphs = None

if 'selected_elements' not in st.session_state:
    st.session_state.selected_elements = None

if 'rank_pc' not in st.session_state:
    st.session_state.rank_pc = []

if 'rank_base' not in st.session_state:
    st.session_state.rank_base = None


def extract_kw_nodes_predicates(selected_elements):
    kw_sub_graphs = {'nodes_kw': [],
                     'predicates_kw': []}
    for element in selected_elements:
        if len(list(element['nodes_kw'])) > 0:
            kw_sub_graphs['nodes_kw'].extend(list(element['nodes_kw']))
        if len(element['predicates_kw']) > 0:
            kw_sub_graphs['predicates_kw'].extend(element['predicates_kw'])

    return kw_sub_graphs


def draw_graph(graph, node_font, predicate_font, st_graph):
    pos = nx.spring_layout(graph)
    # Set the figure size
    plt.figure(figsize=(12, 10))
    nx.draw(graph, pos, with_labels=True, font_weight='bold',
            node_color='skyblue', node_size=400, font_size=node_font,
            edge_color='gray', width=1.5)
    # Get edge labels from the 'type' attribute
    edge_labels = {(u, v): d['type'] for u, v, d in graph.edges(data=True)}

    # Draw edge labels
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels,
                                 font_color='red', font_size=predicate_font)

    plt.title("NetworkX Graph")
    st_graph.pyplot(plt.gcf())


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


if __name__ == "__main__":

    st.title("Key words search in RDF graph")

    uploaded_file_rdt_dataset = st.file_uploader("Choisir un Dataset rdf", type=['xml'])
    uploaded_file_rdt_ontologie = st.file_uploader("Choisir une ontologie rdf", type=['xml'])

    if uploaded_file_rdt_dataset is not None:
        rdf_file = uploaded_file_rdt_dataset.getvalue()
        rdf_graph = Graph()

        rdf_data = minidom.parseString(rdf_file)

        file_path_ontologie = "Dataset/livres_ontologie.owl.xml"
        rdf_graph = Graph()

        if uploaded_file_rdt_ontologie is not None:
            rdf_file_ontologie = uploaded_file_rdt_ontologie.getvalue()
            rdf_ontologie = minidom.parseString(rdf_file_ontologie)
            ontologie = rdf_ontologie.toxml()
        else:
            ontologie = None


        rdf_graph.parse(data=rdf_data.toxml(), format="xml")
        nx_graph = rdf_to_nx_graph(rdf_graph)

        st.session_state.graph1 = nx_graph
        draw_graph_plotly(st.session_state.graph1)
        keywords_input = st.text_input("Entrez des mots-clés séparés par des points-virgules:")

        keywords = [keyword.strip() for keyword in keywords_input.split(';')]        

        if st.button("Generate produit cartésien"):
            st.session_state.selected_elements = search_in_rdf_graph(rdf_graph, keywords, client, rdf_data.toxml(), ontologie= ontologie)
            kw_cartesien_sub_graphs = produit_cartesien(st.session_state.selected_elements)
            st.session_state.produit_cartesien = []
            st.session_state.rank_pc = []
            st.session_state.rank_base = None
            for i in range(len(kw_cartesien_sub_graphs)):
                st.session_state.produit_cartesien.append('Option ' + str(i))
                l, m = graph_construct(kw_cartesien_sub_graphs[i], nx_graph)
                n = graph_mots_cle(l, m)
                rank = ranking(graph_mots_cle(kw_cartesien_sub_graphs[i]['predicates_kw'],
                                            kw_cartesien_sub_graphs[i]['nodes_kw']), n)
                st.session_state.rank_pc.append({'combinaison': i,
                                                'rank': rank})

        keywords_list = st.selectbox("Choisissez quel combinaison afficher",
                                    st.session_state.produit_cartesien)

        if keywords_list:
            selected_option_index = int(keywords_list.split()[-1])

            kw_cartesien_sub_graphs = produit_cartesien(st.session_state.selected_elements)
            kw_graph_predicates, kw_graph_nodes = graph_construct(
                kw_cartesien_sub_graphs[selected_option_index], nx_graph)
            nx_kw_graph = graph_mots_cle(kw_graph_predicates, kw_graph_nodes)
            st.session_state.graph3 = nx_kw_graph
        if len(st.session_state.rank_pc) > 0:
            draw_hist(st.session_state.rank_pc)

        if st.session_state.rank_base is not None:
            st.write('Score de ranking: '+str(st.session_state.rank_base))
        if st.session_state.graph3 is not None:
            draw_graph_plotly(st.session_state.graph3)
