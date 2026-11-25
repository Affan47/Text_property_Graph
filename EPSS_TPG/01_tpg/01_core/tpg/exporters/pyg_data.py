"""Single tensor-layout contract used by training and inference."""
import json


def to_pyg_data(payload, label=0, soft_label=False):
    import torch
    from torch_geometric.data import Data
    n, t, e, r = (payload[k] for k in ("num_nodes", "num_node_types", "num_edges", "num_edge_types"))
    x = torch.tensor(payload["x"], dtype=torch.float).reshape(n, payload["node_feature_dim"])
    data = Data(x=x, edge_index=torch.tensor(payload["edge_index"], dtype=torch.long).reshape(2, e),
        edge_type=torch.tensor(payload["edge_type"], dtype=torch.long),
        edge_attr=torch.tensor(payload["edge_attr"], dtype=torch.float).reshape(e, r),
        edge_features=torch.tensor(payload["edge_features"], dtype=torch.float).reshape(e, payload["edge_feature_dim"]),
        node_type=x[:, :t].argmax(dim=1),
        node_pos=torch.tensor(payload["node_pos"], dtype=torch.long).reshape(n, 3),
        y=torch.tensor([label], dtype=torch.float if soft_label else torch.long), num_nodes=n)
    data.type_feature_dim = t
    data.num_edge_types = r
    data.graph_version = payload["graph_version"]
    data.type_vocabulary_json = json.dumps(payload["type_vocabulary"], sort_keys=True)
    data.edge_properties_json = json.dumps(payload["edge_properties"], sort_keys=True)
    return data
