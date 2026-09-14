from dotenv import load_dotenv
from neuprint import Client, NeuronCriteria, fetch_neurons, fetch_adjacencies
from pathlib import Path

    
load_dotenv()
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

def load_neuprint_data():
    client = Client(
        "https://neuprint.janelia.org",
        dataset="optic-lobe:v1.1",
    )

    regions = ["LA(R)", "ME(R)", "LO(R)", "LOP(R)", "AME(R)"]
    criteria = NeuronCriteria(
        rois=regions,
        roi_req="any",
        client=client
    )

    neurons, roi_counts = fetch_neurons(criteria, client=client)
    neuron_info, roi_connections = fetch_adjacencies(
        sources=neurons.bodyId.tolist(),
        targets=neurons.bodyId.tolist(),
        client=client,
    )

    edges = roi_connections.groupby(
        ["bodyId_pre", "bodyId_post"], as_index=False
    )["weight"].sum()

    neurons.to_parquet(output_dir / "neurons.parquet", index=False)
    edges.to_parquet(output_dir / "edges.parquet", index=False)
    roi_counts.to_parquet(output_dir / "roi_counts.parquet", index=False)
