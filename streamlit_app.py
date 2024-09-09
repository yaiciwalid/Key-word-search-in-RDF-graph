import streamlit as st
from rdflib import Graph
from xml.dom import minidom
from utils import rdf_to_nx_graph, search_in_rdf_graph, graph_mots_cle, \
    graph_construct, produit_cartesien, ranking, draw_graph_plotly, draw_hist
import os
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = ""

client = OpenAI()

# Predefined datasets and ontologies (for dropdown options)
default_datasets = {
    "Books Dataset": "./Datasets/livres.rdf.xml",
    "Movies Dataset": "./Datasets/film.rdf.xml"
}

default_ontologies = {
    "Books Ontology": "Datasets/livres_ontologie.owl.xml",
}

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


if __name__ == "__main__":

    st.title("Key words search in RDF graph")

    # Dropdown for predefined datasets and ontologies
    selected_dataset = st.selectbox("Select a default RDF Dataset", ["None"] + list(default_datasets.keys()))
    selected_ontology = st.selectbox("Select a default RDF Ontology", ["None"] + list(default_ontologies.keys()))

    # File uploaders to allow custom datasets/ontologies
    uploaded_file_rdt_dataset = st.file_uploader("Or upload your own Dataset rdf", type=['xml'])
    uploaded_file_rdt_ontologie = st.file_uploader("Or upload your own Ontology rdf", type=['xml'])


    # Load RDF dataset from either dropdown or uploaded file
    if selected_dataset != "None":
        with open(default_datasets[selected_dataset], "r") as f:
            rdf_file = f.read()
            rdf_data = minidom.parseString(rdf_file)
    elif uploaded_file_rdt_dataset is not None:
        rdf_file = uploaded_file_rdt_dataset.getvalue()
        rdf_data = minidom.parseString(rdf_file)
    else:
        rdf_data = None
        st.session_state.rank_base = None
        st.session_state.rank_pc = None
        st.session_state.graph3 = None
        st.session_state.produit_cartesien = None

    # Load RDF ontology from either dropdown or uploaded file
    if selected_ontology != "None":
        with open(default_ontologies[selected_ontology], "r") as f:
            rdf_file_ontologie = f.read()
            rdf_ontologie = minidom.parseString(rdf_file_ontologie)
            ontologie = rdf_ontologie.toxml()
    elif uploaded_file_rdt_ontologie is not None:
        rdf_file_ontologie = uploaded_file_rdt_ontologie.getvalue()
        rdf_ontologie = minidom.parseString(rdf_file_ontologie)
        ontologie = rdf_ontologie.toxml()
    else:
        ontologie = None

    # If an RDF dataset has been loaded (either from file or dropdown)
    if rdf_data is not None:
        rdf_graph = Graph()
        rdf_graph.parse(data=rdf_data.toxml(), format="xml")
        nx_graph = rdf_to_nx_graph(rdf_graph)

        st.session_state.graph1 = nx_graph
        draw_graph_plotly(st.session_state.graph1)

        # User enters keywords
        keywords_input = st.text_input("Enter keywords separated by semicolons:")
        if keywords_input!='':
            keywords = [keyword.strip() for keyword in keywords_input.split(';')]
        else:
            keywords = []

        if st.button("Generate produit cartésien") and len(keywords)>0:
            st.session_state.selected_elements = search_in_rdf_graph(rdf_graph, keywords, client, rdf_data.toxml(), ontologie=ontologie)
            kw_cartesien_sub_graphs = produit_cartesien(st.session_state.selected_elements)
            if kw_cartesien_sub_graphs == []:
                st.warning("Aucune correspondance trouvée")
            st.session_state.produit_cartesien = []
            st.session_state.rank_pc = []
            st.session_state.rank_base = None
            for i in range(len(kw_cartesien_sub_graphs)):
                st.session_state.produit_cartesien.append('Option ' + str(i))
                l, m = graph_construct(kw_cartesien_sub_graphs[i], nx_graph)
                n = graph_mots_cle(l, m)
                rank = ranking(graph_mots_cle(kw_cartesien_sub_graphs[i]['predicates_kw'], kw_cartesien_sub_graphs[i]['nodes_kw']), n)
                st.session_state.rank_pc.append({'combinaison': i, 'rank': rank})

        # Choose a combination of graphs to display
        keywords_list = st.selectbox("Choose a combination to display", st.session_state.produit_cartesien)

        if keywords_list:
            selected_option_index = int(keywords_list.split()[-1])
            kw_cartesien_sub_graphs = produit_cartesien(st.session_state.selected_elements)
            kw_graph_predicates, kw_graph_nodes = graph_construct(kw_cartesien_sub_graphs[selected_option_index], nx_graph)
            nx_kw_graph = graph_mots_cle(kw_graph_predicates, kw_graph_nodes)
            st.session_state.graph3 = nx_kw_graph

        if st.session_state.rank_pc is not None and len(st.session_state.rank_pc) > 0:
            st.session_state.rank_base = next((item['rank'] for item in st.session_state.rank_pc if item['combinaison'] == selected_option_index), None)
            draw_hist(st.session_state.rank_pc)


        if st.session_state.rank_base is not None:
            st.write('Ranking score: ' + str(st.session_state.rank_base))

        if st.session_state.graph3 is not None:
            draw_graph_plotly(st.session_state.graph3)