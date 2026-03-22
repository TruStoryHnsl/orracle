"""Base node system: PortType, Port, DataChunk, BaseNode, NodeRegistry."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PortType(Enum):
    """Data types that flow between node ports."""
    TEXT = 'text'
    TEXT_BATCH = 'text_batch'
    FILES = 'files'
    METRICS = 'metrics'
    TOKENS = 'tokens'
    VIDEO = 'video'
    FRAMES = 'frames'
    AUDIO = 'audio'


# Which port types can connect to which (source -> allowed destinations)
PORT_COMPATIBILITY = {
    PortType.TEXT: {PortType.TEXT},
    PortType.TEXT_BATCH: {PortType.TEXT_BATCH},
    PortType.FILES: {PortType.FILES},
    PortType.METRICS: {PortType.METRICS},
    PortType.TOKENS: {PortType.TOKENS},
    PortType.VIDEO: {PortType.VIDEO},
    PortType.FRAMES: {PortType.FRAMES},
    PortType.AUDIO: {PortType.AUDIO},
}


@dataclass
class Port:
    """A typed input or output port on a node."""
    name: str
    port_type: PortType
    description: str = ''
    required: bool = True

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'type': self.port_type.value,
            'description': self.description,
            'required': self.required,
        }


@dataclass
class DataChunk:
    """Unit of data flowing between nodes."""
    text: str
    metadata: dict = field(default_factory=dict)
    history: list[str] = field(default_factory=list)

    def with_text(self, new_text: str, node_id: str) -> DataChunk:
        """Return a new DataChunk with updated text and history."""
        return DataChunk(
            text=new_text,
            metadata={**self.metadata},
            history=[*self.history, node_id],
        )

    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'metadata': self.metadata,
            'history': self.history,
        }


class BaseNode:
    """Base class for all pipeline nodes."""
    node_type: str = ''
    label: str = ''
    category: str = ''
    description: str = ''
    inputs: list[Port] = []
    outputs: list[Port] = []
    params_schema: dict = {}

    def process(self, inputs: dict[str, Any], config: dict) -> dict[str, Any]:
        """Execute the node on full data. Returns {port_name: data}."""
        raise NotImplementedError

    def preview(self, inputs: dict[str, Any], config: dict, n: int = 10) -> dict[str, Any]:
        """Run on a sample for live editor preview. Default: slice inputs then process."""
        sampled = {}
        for key, val in inputs.items():
            if isinstance(val, list):
                sampled[key] = val[:n]
            else:
                sampled[key] = val
        return self.process(sampled, config)

    def validate_config(self, config: dict) -> list[str]:
        """Return list of config validation errors (empty = valid)."""
        errors = []
        schema = self.params_schema
        props = schema.get('properties', {})
        required = schema.get('required', [])
        for req in required:
            if req not in config:
                errors.append(f'Missing required parameter: {req}')
        for key, val in config.items():
            if key in props:
                prop = props[key]
                if prop.get('type') == 'number' and not isinstance(val, (int, float)):
                    errors.append(f'{key}: expected number')
                if prop.get('type') == 'integer' and not isinstance(val, int):
                    errors.append(f'{key}: expected integer')
                if prop.get('type') == 'boolean' and not isinstance(val, bool):
                    errors.append(f'{key}: expected boolean')
                if 'enum' in prop and val not in prop['enum']:
                    errors.append(f'{key}: must be one of {prop["enum"]}')
                if 'minimum' in prop and isinstance(val, (int, float)) and val < prop['minimum']:
                    errors.append(f'{key}: minimum is {prop["minimum"]}')
                if 'maximum' in prop and isinstance(val, (int, float)) and val > prop['maximum']:
                    errors.append(f'{key}: maximum is {prop["maximum"]}')
        return errors

    def type_info(self) -> dict:
        """Serialize node type info for the frontend palette."""
        return {
            'type': self.node_type,
            'label': self.label,
            'category': self.category,
            'description': self.description,
            'inputs': [p.to_dict() for p in self.inputs],
            'outputs': [p.to_dict() for p in self.outputs],
            'params_schema': copy.deepcopy(self.params_schema),
        }


class NodeRegistry:
    """Registry of all available node types."""
    _nodes: dict[str, BaseNode] = {}

    @classmethod
    def register(cls, node_cls: type[BaseNode]) -> type[BaseNode]:
        """Decorator to register a node class."""
        instance = node_cls()
        if not instance.node_type:
            raise ValueError(f'{node_cls.__name__} has no node_type')
        cls._nodes[instance.node_type] = instance
        return node_cls

    @classmethod
    def get(cls, node_type: str) -> BaseNode | None:
        return cls._nodes.get(node_type)

    @classmethod
    def all_types(cls) -> dict[str, BaseNode]:
        return dict(cls._nodes)

    @classmethod
    def type_list(cls) -> list[dict]:
        """All registered types as dicts, grouped by category."""
        return [node.type_info() for node in cls._nodes.values()]
