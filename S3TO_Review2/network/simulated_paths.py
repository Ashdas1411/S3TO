CLIENT_TO_EDGE_PATHS = {
    0: {"edge1": ["s1"], "edge2": ["s1", "s2", "s3"]},
    1: {"edge1": ["s1"], "edge2": ["s1", "s2", "s3"]},
    2: {"edge1": ["s2", "s1"], "edge2": ["s2", "s3"]},
    3: {"edge1": ["s2", "s1"], "edge2": ["s2", "s3"]},
    4: {"edge1": ["s3", "s2", "s1"], "edge2": ["s3"]},
    5: {"edge1": ["s3", "s2", "s1"], "edge2": ["s3"]},
}


def path_for(client_id, edge):
    return CLIENT_TO_EDGE_PATHS[int(client_id)][edge]
