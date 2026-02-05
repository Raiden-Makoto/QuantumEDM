import numpy as np
from data.data_loader import get_data

def pack_for_metal(dataset_percent=0.05, output_path="data/packed/"):
    # 1. Get the filtered data using your existing loader logic
    # We use validation_split=0 because we want to pack the full set for now
    train_loader = get_data(dataset_percent=dataset_percent, validation_split=0)
    dataset = train_loader.dataset
    
    node_list = []
    edge_list = []
    metadata = []
    
    node_offset = 0
    edge_offset = 0
    
    print(f"Packing {len(dataset)} molecules...")
    
    for i, data in enumerate(dataset):
        num_nodes = data.x.size(0)
        num_edges = data.edge_index.size(1)
        
        # --- Node Buffer: [x, y, z, atom_type] ---
        # We pack as 4x float32 (16 bytes per node)
        pos = data.pos.numpy().astype(np.float32) # [N, 3]
        z = data.z.numpy().astype(np.float32).reshape(-1, 1) # [N, 1]
        nodes = np.hstack([pos, z]) 
        node_list.append(nodes)
        
        # --- Edge Buffer: [row, col] ---
        # We pack as 2x int32 (8 bytes per edge)
        edges = data.edge_index.numpy().T.astype(np.int32) # [E, 2]
        edge_list.append(edges)
        
        # --- Metadata: [node_start, n_nodes, edge_start, n_edges] ---
        metadata.append([node_offset, num_nodes, edge_offset, num_edges])
        
        node_offset += num_nodes
        edge_offset += num_edges

    # Save to binary
    import os
    os.makedirs(output_path, exist_ok=True)
    
    np.concatenate(node_list).tofile(f"{output_path}qm9_nodes.bin")
    np.concatenate(edge_list).tofile(f"{output_path}qm9_edges.bin")
    np.array(metadata, dtype=np.int32).tofile(f"{output_path}qm9_metadata.bin")
    
    print(f"Success! Files saved to {output_path}")
    print(f"Total Nodes: {node_offset} | Total Edges: {edge_offset}")

if __name__ == "__main__":
    pack_for_metal()