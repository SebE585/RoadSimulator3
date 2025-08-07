import os
from core.utils import get_simulation_output_dir
from simulator.vizualisation.map_renderer import generate_html_map, export_png_map
from simulator.vizualisation.graphs import generate_speed_graph, generate_stop_graph
from simulator.vizualisation.summary import generate_summary_json, generate_summary_log

def generate_all_outputs_from_csv(df, output_dir, timestamp):
    assert os.path.isdir(output_dir), f"Invalid output_dir: {output_dir}"

    html_path = os.path.join(output_dir, "map.html")
    png_path = os.path.join(output_dir, "map.png")
    generate_html_map(df, html_path)
    export_png_map(html_path, png_path)

    speed_graph_path = os.path.join(output_dir, "speed_graph.png")
    stop_graph_path = os.path.join(output_dir, "stop_graph.png")
    generate_speed_graph(df, speed_graph_path)
    generate_stop_graph(df, stop_graph_path)

    summary_json_path = os.path.join(output_dir, "summary.json")
    summary_log_path = os.path.join(output_dir, "summary.log")
    generate_summary_json(df, summary_json_path)
    generate_summary_log(df, summary_log_path)
