"""PipelineRunner — execute pipelines in full, preview, or validate mode."""

from __future__ import annotations

import difflib
import time
import threading
import uuid
from typing import Any, Callable

from executor.dag import Pipeline
from nodes.base import NodeRegistry, DataChunk


class PipelineRunner:
    """Execute a Pipeline's DAG."""

    def __init__(self):
        self.tasks: dict[str, dict] = {}
        self._lock = threading.Lock()

    def validate(self, pipeline: Pipeline) -> dict:
        """Validate pipeline without executing."""
        errors = pipeline.validate()
        order = pipeline.topological_sort() if not errors else []
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'execution_order': order,
            'node_count': len(pipeline.nodes),
            'connection_count': len(pipeline.connections),
        }

    def preview(self, pipeline: Pipeline, n: int = 10) -> dict:
        """Run pipeline in preview mode — process only n samples per node."""
        errors = pipeline.validate()
        if errors:
            return {'error': 'Pipeline has validation errors', 'errors': errors}

        order = pipeline.topological_sort()
        node_outputs: dict[str, dict[str, Any]] = {}
        node_previews: dict[str, dict] = {}

        for node_id in order:
            node_data = pipeline.nodes[node_id]
            node_type = NodeRegistry.get(node_data['type'])
            if node_type is None:
                continue

            # Gather inputs from upstream connections
            inputs = {}
            input_map = pipeline.get_node_inputs(node_id)
            for input_port, (src_node, src_port) in input_map.items():
                if src_node in node_outputs and src_port in node_outputs[src_node]:
                    data = node_outputs[src_node][src_port]
                    # Sample for preview
                    if isinstance(data, list):
                        inputs[input_port] = data[:n]
                    else:
                        inputs[input_port] = data

            try:
                result = node_type.preview(inputs, node_data.get('params', {}), n)
                node_outputs[node_id] = result

                # Build preview summary
                preview = {'node_id': node_id, 'type': node_data['type'], 'status': 'ok'}
                for key, val in result.items():
                    if isinstance(val, list):
                        preview[f'{key}_count'] = len(val)
                        # Include text samples for TEXT_BATCH outputs
                        if val and hasattr(val[0], 'text'):
                            preview[f'{key}_samples'] = [
                                {'text': c.text[:200], 'metadata': c.metadata}
                                for c in val[:3]
                            ]
                    elif isinstance(val, dict):
                        preview[key] = val
                node_previews[node_id] = preview

            except Exception as e:
                node_previews[node_id] = {
                    'node_id': node_id,
                    'type': node_data['type'],
                    'status': 'error',
                    'error': str(e),
                }

        return {'previews': node_previews, 'execution_order': order}

    def preview_diff(self, pipeline: Pipeline, n: int = 20) -> dict:
        """Run pipeline and track per-chunk text changes at every node.

        Returns the chunk that was modified by the most nodes, with a
        full node-by-node diff timeline.
        """
        errors = pipeline.validate()
        if errors:
            return {'error': 'Pipeline has validation errors', 'errors': errors}

        order = pipeline.topological_sort()
        node_outputs: dict[str, dict[str, Any]] = {}

        # Track text snapshots per chunk across the pipeline.
        # Key: chunk index (position in the first TEXT_BATCH output).
        # Value: list of {node_id, label, before, after, changed}
        chunk_timelines: dict[int, list[dict]] = {}
        chunk_originals: dict[int, str] = {}   # original text at entry
        chunk_metadata: dict[int, dict] = {}
        first_batch_node: str | None = None

        # Identify TEXT_BATCH processing nodes (nodes whose type has
        # a text_batch input and a text_batch output).
        text_nodes = set()
        for node_id in order:
            node_data = pipeline.nodes[node_id]
            nt = NodeRegistry.get(node_data['type'])
            if nt is None:
                continue
            has_batch_in = any(p.port_type.value == 'text_batch' for p in nt.inputs)
            has_batch_out = any(p.port_type.value == 'text_batch' for p in nt.outputs)
            if has_batch_in and has_batch_out:
                text_nodes.add(node_id)

        for node_id in order:
            node_data = pipeline.nodes[node_id]
            node_type = NodeRegistry.get(node_data['type'])
            if node_type is None:
                continue

            # Gather inputs
            inputs = {}
            input_map = pipeline.get_node_inputs(node_id)
            for input_port, (src_node, src_port) in input_map.items():
                if src_node in node_outputs and src_port in node_outputs[src_node]:
                    data = node_outputs[src_node][src_port]
                    if isinstance(data, list):
                        inputs[input_port] = data[:n]
                    else:
                        inputs[input_port] = data

            # Snapshot text BEFORE this node processes it
            before_texts: dict[int, str] = {}
            if node_id in text_nodes:
                batch_in = None
                for port_name, data in inputs.items():
                    if isinstance(data, list) and data and isinstance(data[0], DataChunk):
                        batch_in = data
                        break
                if batch_in:
                    if first_batch_node is None:
                        first_batch_node = node_id
                        for i, chunk in enumerate(batch_in):
                            chunk_originals[i] = chunk.text
                            chunk_metadata[i] = dict(chunk.metadata)
                            chunk_timelines[i] = []
                    for i, chunk in enumerate(batch_in):
                        before_texts[i] = chunk.text

            try:
                result = node_type.preview(inputs, node_data.get('params', {}), n)
                node_outputs[node_id] = result
            except Exception as e:
                return {
                    'error': f'Node {node_id} failed: {e}',
                    'node_id': node_id,
                }

            # Snapshot text AFTER and record diffs
            if node_id in text_nodes and before_texts:
                # Find the output batch
                batch_out = None
                for port_name in ('out', 'passed', 'clean'):
                    if port_name in result and isinstance(result[port_name], list):
                        if result[port_name] and isinstance(result[port_name][0], DataChunk):
                            batch_out = result[port_name]
                            break
                if batch_out is None:
                    # Try any list of DataChunks in result
                    for val in result.values():
                        if isinstance(val, list) and val and isinstance(val[0], DataChunk):
                            batch_out = val
                            break

                if batch_out:
                    type_info = node_type.type_info()
                    label = type_info.get('label', node_id)
                    # Match by index (preview preserves order and count
                    # unless the node is a filter that removes items).
                    # Build a quick lookup from before-text to index.
                    for i, chunk in enumerate(batch_out):
                        if i in before_texts:
                            before = before_texts[i]
                            after = chunk.text
                            changed = before != after
                            if i not in chunk_timelines:
                                chunk_timelines[i] = []
                            entry = {
                                'node_id': node_id,
                                'label': label,
                                'changed': changed,
                                'chars_before': len(before),
                                'chars_after': len(after),
                            }
                            if changed:
                                entry['chars_removed'] = max(0, len(before) - len(after))
                                entry['chars_added'] = max(0, len(after) - len(before))
                            chunk_timelines[i].append(entry)

        if not chunk_timelines:
            return {
                'error': 'No text processing nodes found in pipeline',
                'execution_order': order,
            }

        # Find the chunk modified by the most nodes
        def edit_count(idx):
            return sum(1 for e in chunk_timelines[idx] if e['changed'])

        best_idx = max(chunk_timelines.keys(), key=edit_count)
        best_edits = edit_count(best_idx)

        # Build the detailed diff timeline for the winner
        timeline = chunk_timelines[best_idx]
        # We need the actual text at each stage to produce diffs.
        # Re-run just for this chunk to capture stage texts.
        # (We already have the data in node_outputs, just need to trace it.)
        # Simpler: re-walk the pipeline outputs and grab this chunk's text.
        stage_texts = [chunk_originals.get(best_idx, '')]
        for entry in timeline:
            node_id = entry['node_id']
            out = node_outputs.get(node_id, {})
            batch_out = None
            for port_name in ('out', 'passed', 'clean'):
                if port_name in out and isinstance(out[port_name], list):
                    if out[port_name] and isinstance(out[port_name][0], DataChunk):
                        batch_out = out[port_name]
                        break
            if batch_out is None:
                for val in out.values():
                    if isinstance(val, list) and val and isinstance(val[0], DataChunk):
                        batch_out = val
                        break
            if batch_out and best_idx < len(batch_out):
                stage_texts.append(batch_out[best_idx].text)
            else:
                stage_texts.append(stage_texts[-1])

        # Generate diffs for nodes that actually changed the text
        detailed_timeline = []
        text_cursor = 0  # index into stage_texts
        for entry in timeline:
            text_cursor += 1
            before = stage_texts[text_cursor - 1]
            after = stage_texts[text_cursor]
            detail = dict(entry)
            if entry['changed']:
                # Produce a compact unified diff
                diff_lines = list(difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    n=2,
                ))
                segments = []
                for line in diff_lines[2:]:  # skip --- / +++
                    if line.startswith('@@'):
                        segments.append({'type': 'header', 'text': line.strip()})
                    elif line.startswith('+'):
                        segments.append({'type': 'insert', 'text': line[1:]})
                    elif line.startswith('-'):
                        segments.append({'type': 'delete', 'text': line[1:]})
                    elif line.startswith(' '):
                        segments.append({'type': 'context', 'text': line[1:]})
                # Cap diff size for the response
                if len(segments) > 120:
                    segments = segments[:100]
                    segments.append({'type': 'header',
                                     'text': f'... ({len(diff_lines) - 102} more lines)'})
                detail['diff'] = segments
            detailed_timeline.append(detail)

        # Summary stats across all chunks
        all_edit_counts = [edit_count(i) for i in chunk_timelines]
        original_text = chunk_originals.get(best_idx, '')
        final_text = stage_texts[-1] if stage_texts else ''

        return {
            'chunk_index': best_idx,
            'metadata': chunk_metadata.get(best_idx, {}),
            'edit_count': best_edits,
            'total_nodes': len(timeline),
            'original_chars': len(original_text),
            'final_chars': len(final_text),
            'original_text': original_text[:3000],
            'final_text': final_text[:3000],
            'original_truncated': len(original_text) > 3000,
            'final_truncated': len(final_text) > 3000,
            'timeline': detailed_timeline,
            'execution_order': order,
            'stats': {
                'total_chunks': len(chunk_timelines),
                'avg_edits': round(sum(all_edit_counts) / max(1, len(all_edit_counts)), 1),
                'max_edits': max(all_edit_counts) if all_edit_counts else 0,
                'chunks_with_zero_edits': sum(1 for c in all_edit_counts if c == 0),
                'edit_distribution': {str(i): all_edit_counts.count(i)
                                      for i in sorted(set(all_edit_counts))},
            },
        }

    def execute(self, pipeline: Pipeline,
                progress_callback: Callable | None = None) -> str:
        """Launch full pipeline execution in background. Returns task_id."""
        task_id = f'run_{uuid.uuid4().hex[:8]}'
        task = {
            'id': task_id,
            'pipeline_id': pipeline.id,
            'status': 'running',
            'started': time.strftime('%Y-%m-%d %H:%M:%S'),
            'finished': None,
            'progress': {},
            'node_metrics': {},
            'error': None,
            'log': [],
        }
        with self._lock:
            self.tasks[task_id] = task

        thread = threading.Thread(
            target=self._run,
            args=(task_id, pipeline, progress_callback),
            daemon=True,
        )
        thread.start()
        return task_id

    def _run(self, task_id: str, pipeline: Pipeline,
             progress_callback: Callable | None):
        task = self.tasks[task_id]

        errors = pipeline.validate()
        if errors:
            task['status'] = 'failed'
            task['error'] = f'Validation failed: {"; ".join(errors)}'
            task['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
            return

        order = pipeline.topological_sort()
        total_nodes = len(order)
        node_outputs: dict[str, dict[str, Any]] = {}

        for i, node_id in enumerate(order):
            node_data = pipeline.nodes[node_id]
            node_type = NodeRegistry.get(node_data['type'])
            if node_type is None:
                continue

            task['progress'] = {
                'current_node': node_id,
                'node_index': i,
                'total_nodes': total_nodes,
                'percent': round(i / max(1, total_nodes) * 100),
            }
            task['log'].append(f'[{i+1}/{total_nodes}] Processing {node_id} ({node_data["type"]})')

            if progress_callback:
                progress_callback(task['progress'])

            # Gather inputs
            inputs = {}
            input_map = pipeline.get_node_inputs(node_id)
            for input_port, (src_node, src_port) in input_map.items():
                if src_node in node_outputs and src_port in node_outputs[src_node]:
                    inputs[input_port] = node_outputs[src_node][src_port]

            try:
                start = time.time()
                result = node_type.process(inputs, node_data.get('params', {}))
                elapsed = time.time() - start

                node_outputs[node_id] = result

                # Extract metrics
                metrics = result.get('metrics', {})
                metrics['elapsed_sec'] = round(elapsed, 2)
                task['node_metrics'][node_id] = metrics
                task['log'].append(f'  -> completed in {elapsed:.2f}s')

            except Exception as e:
                task['status'] = 'failed'
                task['error'] = f'Node {node_id} failed: {str(e)}'
                task['log'].append(f'  -> ERROR: {str(e)}')
                task['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
                return

        task['status'] = 'completed'
        task['progress']['percent'] = 100
        task['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
        task['log'].append('Pipeline completed successfully')

        # Auto-evict after 30 minutes
        def _evict():
            time.sleep(1800)
            with self._lock:
                self.tasks.pop(task_id, None)
        threading.Thread(target=_evict, daemon=True).start()

    def stop(self, task_id: str) -> bool:
        """Stop a running task."""
        with self._lock:
            task = self.tasks.get(task_id)
            if task and task['status'] == 'running':
                task['status'] = 'cancelled'
                task['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
                return True
        return False

    def get_task(self, task_id: str) -> dict | None:
        with self._lock:
            t = self.tasks.get(task_id)
            return dict(t) if t else None

    def list_tasks(self) -> list[dict]:
        with self._lock:
            return sorted(
                [{'id': t['id'], 'pipeline_id': t['pipeline_id'],
                  'status': t['status'], 'started': t['started'],
                  'finished': t['finished'],
                  'progress': t.get('progress', {})}
                 for t in self.tasks.values()],
                key=lambda t: t['started'], reverse=True,
            )
