"""Explicit OpenFlow port paths for the fixed six-client S3TO Mininet topology."""

# DPIDs are represented as integers because OS-Ken's datapath IDs are integers.
CLIENT_IPS = {
    0: "10.0.0.11",
    1: "10.0.0.12",
    2: "10.0.0.13",
    3: "10.0.0.14",
    4: "10.0.0.15",
    5: "10.0.0.16",
}
GLOBAL_IP = "10.0.0.103"
EDGE_IPS = {"edge1": "10.0.0.101", "edge2": "10.0.0.102"}
# All experiment endpoints whose old IPv4 flow rules must be removed between modes.
ALL_EXPERIMENT_HOST_IPS = tuple(CLIENT_IPS.values()) + tuple(EDGE_IPS.values()) + (GLOBAL_IP,)

# Zero-based client IDs. Paths list (DPID, output port) per traversed switch.
# These match the fixed topology in run_s3to_mininet_fl.py exactly.
S3TO_UPLINK_PATHS = {
    0: {"edge1": ((1, 4),), "edge2": ((1, 3), (2, 4), (3, 3))},
    1: {"edge1": ((1, 4),), "edge2": ((1, 3), (2, 4), (3, 3))},
    2: {"edge1": ((2, 3), (1, 4)), "edge2": ((2, 4), (3, 3))},
    3: {"edge1": ((2, 3), (1, 4)), "edge2": ((2, 4), (3, 3))},
    4: {"edge1": ((3, 4), (2, 3), (1, 4)), "edge2": ((3, 3),)},
    5: {"edge1": ((3, 4), (2, 3), (1, 4)), "edge2": ((3, 3),)},
}

# Explicit reverse paths keep acknowledgements/responses on deterministic routes
# and make flow-counter coverage symmetric between hierarchical and direct modes.
S3TO_DOWNLINK_PATHS = {
    0: {"edge1": ((1, 1),), "edge2": ((3, 4), (2, 3), (1, 1))},
    1: {"edge1": ((1, 2),), "edge2": ((3, 4), (2, 3), (1, 2))},
    2: {"edge1": ((1, 3), (2, 1)), "edge2": ((3, 4), (2, 1))},
    3: {"edge1": ((1, 3), (2, 2)), "edge2": ((3, 4), (2, 2))},
    4: {"edge1": ((1, 3), (2, 4), (3, 1)), "edge2": ((3, 1),)},
    5: {"edge1": ((1, 3), (2, 4), (3, 2)), "edge2": ((3, 2),)},
}


def validate_s3to_paths():
    """Check that every client has a forward and reverse route to both edges."""
    clients = set(CLIENT_IPS)
    expected_edges = set(EDGE_IPS)
    for routes in (S3TO_UPLINK_PATHS, S3TO_DOWNLINK_PATHS):
        if set(routes) != clients:
            raise ValueError("every client must have routes for both S3TO edges")
        for cid, by_edge in routes.items():
            if set(by_edge) != expected_edges:
                raise ValueError(f"client {cid} routes must cover both edges")
            if any(not path for path in by_edge.values()):
                raise ValueError(f"client {cid} has an empty S3TO path")
    return True


validate_s3to_paths()

# One flow entry is installed at each switch on the client->global path.
DIRECT_UPLINK_PATHS = {
    0: ((1, 3), (2, 5)),
    1: ((1, 3), (2, 5)),
    2: ((2, 5),),
    3: ((2, 5),),
    4: ((3, 4), (2, 5)),
    5: ((3, 4), (2, 5)),
}

# The reverse paths use global->client IP matches and corresponding egress ports.
DIRECT_DOWNLINK_PATHS = {
    0: ((2, 3), (1, 1)),
    1: ((2, 3), (1, 2)),
    2: ((2, 1),),
    3: ((2, 2),),
    4: ((2, 4), (3, 1)),
    5: ((2, 4), (3, 2)),
}


def validate_direct_paths():
    """Raise on common port-map mistakes before controller tests or setup."""
    clients = set(CLIENT_IPS)
    if clients != set(range(6)):
        raise ValueError("direct topology requires client IDs 0 through 5")
    if set(DIRECT_UPLINK_PATHS) != clients or set(DIRECT_DOWNLINK_PATHS) != clients:
        raise ValueError("every client must have both direct paths")
    for cid in clients:
        uplink = DIRECT_UPLINK_PATHS[cid]
        downlink = DIRECT_DOWNLINK_PATHS[cid]
        if not uplink or uplink[-1] != (2, 5):
            raise ValueError(f"client {cid} uplink must terminate at global link s2:5")
        if not downlink:
            raise ValueError(f"client {cid} downlink cannot be empty")
        expected_edge_port = {0: (1, 1), 1: (1, 2), 2: (2, 1), 3: (2, 2), 4: (3, 1), 5: (3, 2)}[cid]
        if downlink[-1] != expected_edge_port:
            raise ValueError(f"client {cid} downlink terminates at the wrong client port")
    return True


validate_direct_paths()
