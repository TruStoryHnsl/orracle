"""Pipeline: DAG validation, topological sort, serialization."""

from __future__ import annotations

import copy
import uuid
from collections import defaultdict
from pathlib import Path

import yaml

from nodes.base import NodeRegistry, PortType, PORT_COMPATIBILITY


class Pipeline:
    """A directed acyclic graph of processing nodes."""

    def __init__(self, name: str = '', description: str = ''):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.description = description
        self.nodes: dict[str, dict] = {}       # node_id -> {type, params, position, enabled}
        self.connections: list[list[str]] = []  # [[src_node, src_port, dst_node, dst_port], ...]

    def add_node(self, node_type: str, node_id: str | None = None,
                 params: dict | None = None, position: dict | None = None) -> str:
        """Add a node to the pipeline. Returns node_id."""
        if node_id is None:
            base = node_type.replace('_', '')
            existing = [k for k in self.nodes if k.startswith(base)]
            node_id = f'{base}_{len(existing) + 1}'

        if NodeRegistry.get(node_type) is None:
            raise ValueError(f'Unknown node type: {node_type}')

        self.nodes[node_id] = {
            'type': node_type,
            'params': params or {},
            'position': position or {'x': 100, 'y': 100},
            'enabled': True,
        }
        return node_id

    def remove_node(self, node_id: str):
        """Remove a node and all its connections."""
        self.nodes.pop(node_id, None)
        self.connections = [
            c for c in self.connections
            if c[0] != node_id and c[2] != node_id
        ]

    def add_connection(self, src_node: str, src_port: str,
                       dst_node: str, dst_port: str):
        """Add a connection between two node ports."""
        conn = [src_node, src_port, dst_node, dst_port]
        if conn not in self.connections:
            self.connections.append(conn)

    def remove_connection(self, src_node: str, src_port: str,
                          dst_node: str, dst_port: str):
        """Remove a specific connection."""
        conn = [src_node, src_port, dst_node, dst_port]
        if conn in self.connections:
            self.connections.remove(conn)

    def validate(self) -> list[str]:
        """Validate the pipeline. Returns list of error strings."""
        errors = []

        # Check all nodes have valid types
        for node_id, node_data in self.nodes.items():
            node_type = NodeRegistry.get(node_data['type'])
            if node_type is None:
                errors.append(f'Node {node_id}: unknown type "{node_data["type"]}"')
                continue
            # Validate config
            config_errors = node_type.validate_config(node_data.get('params', {}))
            for err in config_errors:
                errors.append(f'Node {node_id}: {err}')

        # Check connections reference valid nodes and ports
        for conn in self.connections:
            src_node, src_port, dst_node, dst_port = conn
            if src_node not in self.nodes:
                errors.append(f'Connection references missing source node: {src_node}')
                continue
            if dst_node not in self.nodes:
                errors.append(f'Connection references missing destination node: {dst_node}')
                continue

            src_type = NodeRegistry.get(self.nodes[src_node]['type'])
            dst_type = NodeRegistry.get(self.nodes[dst_node]['type'])
            if src_type is None or dst_type is None:
                continue

            # Check port existence
            src_ports = {p.name: p for p in src_type.outputs}
            dst_ports = {p.name: p for p in dst_type.inputs}

            if src_port not in src_ports:
                errors.append(f'Node {src_node} has no output port "{src_port}"')
            if dst_port not in dst_ports:
                errors.append(f'Node {dst_node} has no input port "{dst_port}"')

            # Check type compatibility
            if src_port in src_ports and dst_port in dst_ports:
                src_pt = src_ports[src_port].port_type
                dst_pt = dst_ports[dst_port].port_type
                if dst_pt not in PORT_COMPATIBILITY.get(src_pt, set()):
                    errors.append(
                        f'Type mismatch: {src_node}.{src_port} ({src_pt.value}) '
                        f'-> {dst_node}.{dst_port} ({dst_pt.value})'
                    )

        # Check for required inputs without connections
        for node_id, node_data in self.nodes.items():
            if not node_data.get('enabled', True):
                continue
            node_type = NodeRegistry.get(node_data['type'])
            if node_type is None:
                continue
            for port in node_type.inputs:
                if port.required:
                    has_conn = any(
                        c[2] == node_id and c[3] == port.name
                        for c in self.connections
                    )
                    if not has_conn:
                        errors.append(
                            f'Node {node_id}: required input "{port.name}" is not connected'
                        )

        # Check for cycles
        cycle = self._detect_cycle()
        if cycle:
            errors.append(f'Pipeline contains a cycle: {" -> ".join(cycle)}')

        return errors

    def _detect_cycle(self) -> list[str] | None:
        """Detect cycles using DFS. Returns cycle path or None."""
        adj = defaultdict(set)
        for conn in self.connections:
            adj[conn[0]].add(conn[2])

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self.nodes}
        parent = {}

        def dfs(node):
            color[node] = GRAY
            for neighbor in adj[node]:
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    # Found cycle — reconstruct path
                    cycle = [neighbor, node]
                    current = node
                    while current in parent and parent[current] != neighbor:
                        current = parent[current]
                        cycle.append(current)
                    cycle.append(neighbor)
                    return list(reversed(cycle))
                if color[neighbor] == WHITE:
                    parent[neighbor] = node
                    result = dfs(neighbor)
                    if result:
                        return result
            color[node] = BLACK
            return None

        for node in self.nodes:
            if color[node] == WHITE:
                result = dfs(node)
                if result:
                    return result
        return None

    def topological_sort(self) -> list[str]:
        """Return nodes in execution order (Kahn's algorithm).
        Only includes enabled nodes.
        """
        # Build adjacency from connections
        in_degree = defaultdict(int)
        adj = defaultdict(set)
        enabled = {nid for nid, nd in self.nodes.items() if nd.get('enabled', True)}

        for nid in enabled:
            in_degree.setdefault(nid, 0)

        for conn in self.connections:
            src, _, dst, _ = conn
            if src in enabled and dst in enabled:
                adj[src].add(dst)
                in_degree[dst] += 1

        # Start with nodes having no incoming edges
        queue = [n for n in enabled if in_degree[n] == 0]
        queue.sort()  # Deterministic order
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in sorted(adj[node]):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def get_node_inputs(self, node_id: str) -> dict[str, tuple[str, str]]:
        """Get input connections for a node: {input_port: (src_node, src_port)}."""
        inputs = {}
        for conn in self.connections:
            if conn[2] == node_id:
                inputs[conn[3]] = (conn[0], conn[1])
        return inputs

    def to_dict(self) -> dict:
        """Serialize pipeline to dict."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'nodes': copy.deepcopy(self.nodes),
            'connections': [list(c) for c in self.connections],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Pipeline:
        """Deserialize pipeline from dict."""
        p = cls(name=data.get('name', ''), description=data.get('description', ''))
        p.id = data.get('id', p.id)
        p.nodes = data.get('nodes', {})
        p.connections = data.get('connections', [])
        return p

    def save(self, directory: str) -> str:
        """Save pipeline to YAML file. Returns file path."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = Path(directory) / f'{self.id}.yaml'
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)
        return str(path)

    @classmethod
    def load(cls, path: str) -> Pipeline:
        """Load pipeline from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
