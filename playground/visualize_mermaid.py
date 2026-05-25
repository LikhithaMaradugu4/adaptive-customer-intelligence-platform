from app.graph import graph


mermaid_graph = (
    graph.get_graph()
    .draw_mermaid()
)

with open(
    "graph.mmd",
    "w"
) as f:

    f.write(mermaid_graph)

print(
    "\nMermaid graph saved "
    "as graph.mmd"
)