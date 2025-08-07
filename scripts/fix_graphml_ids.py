import xml.etree.ElementTree as ET
import argparse

def sanitize_graphml_node_ids(input_path, output_path):
    print(f"📥 Lecture brute du fichier GraphML : {input_path}")
    tree = ET.parse(input_path)
    root = tree.getroot()

    ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}

    node_id_map = {}
    new_id_counter = 0

    # Renommage des nœuds
    for node in root.findall(".//graphml:node", ns):
        old_id = node.attrib['id']
        new_id = f"n{new_id_counter}"
        node_id_map[old_id] = new_id
        node.attrib['id'] = new_id
        new_id_counter += 1

    # Mise à jour des arêtes
    for edge in root.findall(".//graphml:edge", ns):
        if 'source' in edge.attrib and edge.attrib['source'] in node_id_map:
            edge.attrib['source'] = node_id_map[edge.attrib['source']]
        if 'target' in edge.attrib and edge.attrib['target'] in node_id_map:
            edge.attrib['target'] = node_id_map[edge.attrib['target']]

    print(f"💾 Export vers : {output_path}")
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"✅ Fichier GraphML corrigé généré avec {new_id_counter} nœuds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix GraphML node IDs from tuple to string")
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    args = parser.parse_args()

    sanitize_graphml_node_ids(args.input_path, args.output_path)