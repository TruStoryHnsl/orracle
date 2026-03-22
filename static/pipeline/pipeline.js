/**
 * pipeline.js — Editor controller: canvas pan/zoom, node management,
 * undo/redo, save/load, drag-from-palette, box selection
 */

const editor = {
    // State
    pipeline: null,
    selectedNodes: new Set(),
    canvas: null,
    viewport: null,

    // Transform
    panX: 0,
    panY: 0,
    zoom: 1,
    minZoom: 0.1,
    maxZoom: 3,
    gridSize: 20,

    // Interaction state
    isDragging: false,
    isPanning: false,
    isBoxSelecting: false,
    dragStart: null,
    dragNodeOffsets: null,
    dragMoved: false,          // Track if nodes actually moved
    boxStart: null,            // {x, y} in screen coords

    // Undo/redo
    undoStack: [],
    redoStack: [],
    maxUndo: 50,

    _spaceHeld: false,

    // -----------------------------------------------------------------------
    // Initialization
    // -----------------------------------------------------------------------

    async init() {
        this.canvas = document.getElementById('canvas');
        this.viewport = document.getElementById('canvas-viewport');
        const svg = document.getElementById('connections-svg');

        ConnectionManager.init(svg);
        await NodeRenderer.loadTypes();
        NodeRenderer.buildPalette(document.getElementById('palette-list'));

        this._bindEvents();
        this._loadPipelineList();
        this._loadPresetList();

        this.pipeline = { id: '', name: 'Untitled', description: '', nodes: {}, connections: [] };
        this.updateTransform();
    },

    // -----------------------------------------------------------------------
    // Event binding
    // -----------------------------------------------------------------------

    _bindEvents() {
        const vp = this.viewport;

        // --- mousedown on viewport ---
        vp.addEventListener('mousedown', (e) => {
            // Right-click or middle-click → pan
            if (e.button === 2 || e.button === 1) {
                e.preventDefault();
                this._startPan(e);
                return;
            }

            // Space + left-click → pan
            if (e.button === 0 && this._spaceHeld) {
                e.preventDefault();
                this._startPan(e);
                return;
            }

            // Left-click on empty canvas
            if (e.button === 0 && !e.target.closest('.pipeline-node')) {
                // Deselect everything
                this.deselectAll();
                ConnectionManager.deselectAll();

                // Start box selection
                this.isBoxSelecting = true;
                const rect = vp.getBoundingClientRect();
                this.boxStart = { x: e.clientX - rect.left, y: e.clientY - rect.top };
                const box = document.getElementById('selection-box');
                box.style.display = 'none'; // Show on first move
            }
        });

        // --- mousemove on window (so we never lose it) ---
        window.addEventListener('mousemove', (e) => {
            if (this.isPanning) {
                this.panX = e.clientX - this.dragStart.x;
                this.panY = e.clientY - this.dragStart.y;
                this.updateTransform();
                ConnectionManager.redraw();
                return;
            }

            // Node dragging
            if (this.isDragging && this.dragNodeOffsets) {
                this.dragMoved = true;
                for (const [nodeId, offset] of Object.entries(this.dragNodeOffsets)) {
                    const x = this._snapToGrid((e.clientX - offset.dx) / this.zoom - this.panX / this.zoom);
                    const y = this._snapToGrid((e.clientY - offset.dy) / this.zoom - this.panY / this.zoom);
                    const el = this.canvas.querySelector(`.pipeline-node[data-id="${nodeId}"]`);
                    if (el) {
                        el.style.left = x + 'px';
                        el.style.top = y + 'px';
                    }
                    if (this.pipeline.nodes[nodeId]) {
                        this.pipeline.nodes[nodeId].position = { x, y };
                    }
                }
                ConnectionManager.redraw();
                return;
            }

            // Box selection
            if (this.isBoxSelecting && this.boxStart) {
                const rect = vp.getBoundingClientRect();
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;
                const box = document.getElementById('selection-box');
                const x = Math.min(this.boxStart.x, mx);
                const y = Math.min(this.boxStart.y, my);
                const w = Math.abs(mx - this.boxStart.x);
                const h = Math.abs(my - this.boxStart.y);

                // Only show once moved a few pixels (avoid flash on click)
                if (w > 4 || h > 4) {
                    box.style.display = 'block';
                    box.style.left = x + 'px';
                    box.style.top = y + 'px';
                    box.style.width = w + 'px';
                    box.style.height = h + 'px';
                }
                return;
            }

            // Connection dragging
            if (ConnectionManager.dragging) {
                const rect = this.canvas.getBoundingClientRect();
                const style = window.getComputedStyle(this.canvas);
                const matrix = new DOMMatrix(style.transform);
                const cx = (e.clientX - rect.left) / matrix.a;
                const cy = (e.clientY - rect.top) / matrix.d;
                ConnectionManager.updateDrag(cx, cy);
            }
        });

        // --- mouseup on window ---
        window.addEventListener('mouseup', (e) => {
            if (this.isPanning) {
                this.isPanning = false;
                vp.classList.remove('panning');
                return;
            }

            if (this.isDragging) {
                this.isDragging = false;
                // Remove drag visual
                this.canvas.querySelectorAll('.pipeline-node.dragging').forEach(
                    el => el.classList.remove('dragging')
                );
                // Only push undo if something actually moved
                if (this.dragMoved) {
                    this._pushUndo();
                }
                this.dragNodeOffsets = null;
                this.dragMoved = false;
                return;
            }

            // Box selection end
            if (this.isBoxSelecting) {
                this.isBoxSelecting = false;
                const box = document.getElementById('selection-box');
                if (box.style.display === 'block') {
                    this._finishBoxSelect(box);
                }
                box.style.display = 'none';
                return;
            }

            if (ConnectionManager.dragging) {
                const portDot = e.target.closest('.port-dot');
                if (portDot) {
                    const nodeEl = portDot.closest('.pipeline-node');
                    const conn = ConnectionManager.endDrag(
                        nodeEl.dataset.id,
                        portDot.dataset.port,
                        portDot.dataset.type,
                        portDot.dataset.dir === 'out'
                    );
                    if (conn) {
                        this.pipeline.connections = ConnectionManager.toArray();
                        this._pushUndo();
                    }
                } else {
                    ConnectionManager.cancelDrag();
                }
            }
        });

        // --- scroll wheel zoom ---
        vp.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            const newZoom = Math.max(this.minZoom, Math.min(this.maxZoom, this.zoom * delta));

            const rect = vp.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;

            this.panX = mx - (mx - this.panX) * (newZoom / this.zoom);
            this.panY = my - (my - this.panY) * (newZoom / this.zoom);
            this.zoom = newZoom;
            this.updateTransform();
            ConnectionManager.redraw();
        }, { passive: false });

        // --- drag-and-drop from palette ---
        vp.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
        });

        vp.addEventListener('drop', (e) => {
            e.preventDefault();
            const nodeType = e.dataTransfer.getData('node-type');
            if (!nodeType) return;

            const rect = vp.getBoundingClientRect();
            const x = this._snapToGrid((e.clientX - rect.left - this.panX) / this.zoom);
            const y = this._snapToGrid((e.clientY - rect.top - this.panY) / this.zoom);

            this.addNode(nodeType, { x, y });
        });

        // --- node drag start (delegated from canvas) ---
        this.canvas.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return;
            const nodeEl = e.target.closest('.pipeline-node');
            if (!nodeEl || e.target.classList.contains('port-dot')) return;

            const nodeId = nodeEl.dataset.id;

            if (!this.selectedNodes.has(nodeId)) {
                if (!e.shiftKey) this.deselectAll();
                this.selectNode(nodeId, e.shiftKey);
            }

            // Start drag
            this.isDragging = true;
            this.dragMoved = false;
            this.dragNodeOffsets = {};
            for (const selId of this.selectedNodes) {
                const el = this.canvas.querySelector(`.pipeline-node[data-id="${selId}"]`);
                if (el) {
                    el.classList.add('dragging');
                    this.dragNodeOffsets[selId] = {
                        dx: e.clientX - parseFloat(el.style.left) * this.zoom - this.panX,
                        dy: e.clientY - parseFloat(el.style.top) * this.zoom - this.panY,
                    };
                }
            }
        });

        // --- keyboard ---
        document.addEventListener('keydown', (e) => {
            // Don't intercept when typing in form fields
            const tag = e.target.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
                // But still allow Escape
                if (e.key === 'Escape') {
                    e.target.blur();
                }
                return;
            }

            if (e.code === 'Space') {
                e.preventDefault();
                this._spaceHeld = true;
                vp.classList.add('space-held');
            }

            if (e.key === 'Escape') {
                e.preventDefault();
                if (ConnectionManager.dragging) {
                    ConnectionManager.cancelDrag();
                } else {
                    this.deselectAll();
                    ConnectionManager.deselectAll();
                }
            }

            if (e.key === 'Delete' || e.key === 'Backspace') {
                e.preventDefault();
                this.deleteSelected();
            }

            if (e.key === 'z' && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
                e.preventDefault();
                this.undo();
            }
            if ((e.key === 'z' && (e.ctrlKey || e.metaKey) && e.shiftKey) ||
                (e.key === 'y' && (e.ctrlKey || e.metaKey))) {
                e.preventDefault();
                this.redo();
            }
            if (e.key === 's' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.savePipeline();
            }
            if (e.key === 'a' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.selectAll();
            }
            if (e.key === 'd' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.duplicateSelected();
            }
            if (e.key === 'f' && !e.ctrlKey && !e.metaKey) {
                this.fitView();
            }
            if (e.key === 'p' && !e.ctrlKey && !e.metaKey) {
                this.runPreview();
            }
        });

        document.addEventListener('keyup', (e) => {
            if (e.code === 'Space') {
                this._spaceHeld = false;
                vp.classList.remove('space-held');
            }
        });

        // Prevent context menu on canvas (we use right-click for pan)
        vp.addEventListener('contextmenu', (e) => e.preventDefault());
    },

    // -----------------------------------------------------------------------
    // Pan helpers
    // -----------------------------------------------------------------------

    _startPan(e) {
        this.isPanning = true;
        this.dragStart = { x: e.clientX - this.panX, y: e.clientY - this.panY };
        this.viewport.classList.add('panning');
    },

    // -----------------------------------------------------------------------
    // Box selection
    // -----------------------------------------------------------------------

    _finishBoxSelect(boxEl) {
        const boxRect = boxEl.getBoundingClientRect();

        // Find nodes whose screen position overlaps the box
        const nodes = this.canvas.querySelectorAll('.pipeline-node');
        let found = false;
        nodes.forEach(el => {
            const nodeRect = el.getBoundingClientRect();
            if (nodeRect.right >= boxRect.left && nodeRect.left <= boxRect.right &&
                nodeRect.bottom >= boxRect.top && nodeRect.top <= boxRect.bottom) {
                this.selectedNodes.add(el.dataset.id);
                el.classList.add('selected');
                found = true;
            }
        });

        if (found && this.selectedNodes.size === 1) {
            this.openConfig([...this.selectedNodes][0]);
        }
    },

    // -----------------------------------------------------------------------
    // Transform
    // -----------------------------------------------------------------------

    updateTransform() {
        this.canvas.style.transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`;
        document.getElementById('zoom-indicator').textContent = Math.round(this.zoom * 100) + '%';

        // Move the grid dots with the canvas
        const vp = this.viewport;
        const gx = this.panX % (this.gridSize * this.zoom);
        const gy = this.panY % (this.gridSize * this.zoom);
        const size = this.gridSize * this.zoom;
        vp.style.backgroundPosition = `${gx}px ${gy}px`;
        vp.style.backgroundSize = `${size}px ${size}px`;
    },

    _snapToGrid(v) {
        return Math.round(v / this.gridSize) * this.gridSize;
    },

    zoomTo(level) {
        const vpRect = this.viewport.getBoundingClientRect();
        const cx = vpRect.width / 2;
        const cy = vpRect.height / 2;
        this.panX = cx - (cx - this.panX) * (level / this.zoom);
        this.panY = cy - (cy - this.panY) * (level / this.zoom);
        this.zoom = level;
        this.updateTransform();
        ConnectionManager.redraw();
    },

    fitView() {
        const nodes = Object.values(this.pipeline.nodes);
        if (nodes.length === 0) {
            this.panX = 0;
            this.panY = 0;
            this.zoom = 1;
            this.updateTransform();
            return;
        }

        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const n of nodes) {
            const p = n.position || { x: 0, y: 0 };
            minX = Math.min(minX, p.x);
            minY = Math.min(minY, p.y);
            maxX = Math.max(maxX, p.x + 180);
            maxY = Math.max(maxY, p.y + 80);
        }

        const vpRect = this.viewport.getBoundingClientRect();
        const pad = 60;
        const contentW = maxX - minX + pad * 2;
        const contentH = maxY - minY + pad * 2;
        this.zoom = Math.min(1.5, Math.min(
            vpRect.width / contentW,
            vpRect.height / contentH
        ));
        this.panX = (vpRect.width - contentW * this.zoom) / 2 - (minX - pad) * this.zoom;
        this.panY = (vpRect.height - contentH * this.zoom) / 2 - (minY - pad) * this.zoom;
        this.updateTransform();
        ConnectionManager.redraw();
    },

    // -----------------------------------------------------------------------
    // Node operations
    // -----------------------------------------------------------------------

    addNode(nodeType, position) {
        const typeInfo = NodeRenderer.nodeTypes[nodeType];
        if (!typeInfo) return null;

        const base = nodeType.replace(/_/g, '');
        // Find a unique ID
        let idx = 1;
        while (this.pipeline.nodes[`${base}_${idx}`]) idx++;
        const nodeId = `${base}_${idx}`;

        const nodeConfig = {
            type: nodeType,
            params: this._defaultParams(typeInfo),
            position: position || { x: 200, y: 200 },
            enabled: true,
        };

        this.pipeline.nodes[nodeId] = nodeConfig;

        const el = NodeRenderer.createNodeElement(nodeId, nodeConfig, typeInfo);
        this.canvas.appendChild(el);
        this._pushUndo();

        this.deselectAll();
        this.selectNode(nodeId);
        return nodeId;
    },

    _defaultParams(typeInfo) {
        const params = {};
        if (typeInfo.params_schema && typeInfo.params_schema.properties) {
            for (const [key, prop] of Object.entries(typeInfo.params_schema.properties)) {
                if (prop.default !== undefined) {
                    params[key] = prop.default;
                }
            }
        }
        return params;
    },

    removeNode(nodeId) {
        delete this.pipeline.nodes[nodeId];
        const el = this.canvas.querySelector(`.pipeline-node[data-id="${nodeId}"]`);
        if (el) el.remove();
        ConnectionManager.removeNodeConnections(nodeId);
        this.pipeline.connections = ConnectionManager.toArray();
        this.selectedNodes.delete(nodeId);
    },

    duplicateSelected() {
        if (this.selectedNodes.size === 0) return;
        const newIds = [];
        const idMap = {};

        for (const nodeId of this.selectedNodes) {
            const orig = this.pipeline.nodes[nodeId];
            if (!orig) continue;
            const pos = orig.position || { x: 100, y: 100 };
            const newId = this.addNode(orig.type, {
                x: pos.x + 40,
                y: pos.y + 40,
            });
            if (newId) {
                // Copy params
                this.pipeline.nodes[newId].params = JSON.parse(JSON.stringify(orig.params || {}));
                idMap[nodeId] = newId;
                newIds.push(newId);
            }
        }

        // Copy connections between duplicated nodes
        for (const conn of ConnectionManager.connections) {
            if (idMap[conn.srcNode] && idMap[conn.dstNode]) {
                ConnectionManager.connections.push({
                    id: `conn_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
                    srcNode: idMap[conn.srcNode],
                    srcPort: conn.srcPort,
                    dstNode: idMap[conn.dstNode],
                    dstPort: conn.dstPort,
                    portType: conn.portType,
                });
            }
        }
        ConnectionManager.redraw();
        this.pipeline.connections = ConnectionManager.toArray();

        // Select the new nodes
        this.deselectAll();
        for (const id of newIds) this.selectNode(id, true);
        this._pushUndo();
    },

    selectNode(nodeId, addToSelection) {
        if (!addToSelection) {
            this.deselectAll();
        }
        this.selectedNodes.add(nodeId);
        const el = this.canvas.querySelector(`.pipeline-node[data-id="${nodeId}"]`);
        if (el) el.classList.add('selected');

        if (this.selectedNodes.size === 1) {
            this.openConfig(nodeId);
        }
    },

    deselectAll() {
        for (const id of this.selectedNodes) {
            const el = this.canvas.querySelector(`.pipeline-node[data-id="${id}"]`);
            if (el) el.classList.remove('selected');
        }
        this.selectedNodes.clear();

        document.getElementById('config-title').textContent = 'No Selection';
        const body = document.getElementById('config-body');
        body.textContent = '';
        const msg = document.createElement('div');
        msg.className = 'empty-state';
        const p = document.createElement('p');
        p.className = 'text-muted';
        p.textContent = 'Select a node to configure it';
        msg.appendChild(p);
        body.appendChild(msg);
        document.getElementById('preview-section').style.display = 'none';
    },

    selectAll() {
        for (const nodeId of Object.keys(this.pipeline.nodes)) {
            this.selectedNodes.add(nodeId);
            const el = this.canvas.querySelector(`.pipeline-node[data-id="${nodeId}"]`);
            if (el) el.classList.add('selected');
        }
    },

    deleteSelected() {
        if (ConnectionManager.selectedConnection) {
            const connId = ConnectionManager.selectedConnection;
            ConnectionManager.removeConnection(connId);
            this.pipeline.connections = ConnectionManager.toArray();
            this._pushUndo();
            return;
        }

        if (this.selectedNodes.size > 0) {
            for (const nodeId of [...this.selectedNodes]) {
                this.removeNode(nodeId);
            }
            this._pushUndo();
        }
    },

    openConfig(nodeId) {
        const nodeConfig = this.pipeline.nodes[nodeId];
        if (!nodeConfig) return;
        const typeInfo = NodeRenderer.nodeTypes[nodeConfig.type];
        NodeRenderer.openConfigPanel(nodeId, nodeConfig, typeInfo);
    },

    updateNodeConfig(nodeId, updates) {
        const node = this.pipeline.nodes[nodeId];
        if (!node) return;
        Object.assign(node, updates);

        const el = this.canvas.querySelector(`.pipeline-node[data-id="${nodeId}"]`);
        if (el) {
            if (updates.enabled !== undefined) {
                el.classList.toggle('disabled', !updates.enabled);
            }
        }
        this._pushUndo();
    },

    updateNodeParam(nodeId, key, value) {
        const node = this.pipeline.nodes[nodeId];
        if (!node) return;
        if (!node.params) node.params = {};
        node.params[key] = value;
    },

    onConnectionSelected(connId) {
        const conn = ConnectionManager.connections.find(c => c.id === connId);
        if (conn) {
            // Also deselect any selected nodes
            this.deselectAll();

            document.getElementById('config-title').textContent = 'Connection';
            const body = document.getElementById('config-body');
            body.textContent = '';
            const info = document.createElement('div');
            info.className = 'config-section';

            const items = [
                ['From', `${conn.srcNode}.${conn.srcPort}`],
                ['To', `${conn.dstNode}.${conn.dstPort}`],
                ['Type', conn.portType],
            ];
            for (const [label, val] of items) {
                const row = document.createElement('div');
                row.className = 'metric-row';
                const labelSpan = document.createElement('span');
                labelSpan.className = 'metric-label';
                labelSpan.textContent = label;
                const valSpan = document.createElement('span');
                valSpan.className = 'metric-value';
                valSpan.textContent = val;
                row.appendChild(labelSpan);
                row.appendChild(valSpan);
                info.appendChild(row);
            }

            const delBtn = document.createElement('button');
            delBtn.className = 'btn btn-danger btn-sm mt-2';
            delBtn.textContent = 'Delete Connection';
            delBtn.addEventListener('click', () => {
                ConnectionManager.removeConnection(connId);
                this.pipeline.connections = ConnectionManager.toArray();
                this.deselectAll();
                this._pushUndo();
            });
            info.appendChild(delBtn);
            body.appendChild(info);
        }
    },

    // -----------------------------------------------------------------------
    // Undo / Redo
    // -----------------------------------------------------------------------

    _pushUndo() {
        const state = JSON.stringify({
            nodes: this.pipeline.nodes,
            connections: this.pipeline.connections,
        });
        // Don't push duplicate states
        if (this.undoStack.length > 0 && this.undoStack[this.undoStack.length - 1] === state) return;
        this.undoStack.push(state);
        if (this.undoStack.length > this.maxUndo) this.undoStack.shift();
        this.redoStack = [];
    },

    undo() {
        if (this.undoStack.length < 2) return;
        const current = this.undoStack.pop();
        this.redoStack.push(current);
        const prev = this.undoStack[this.undoStack.length - 1];
        this._restoreState(prev);
        showToast('Undo', 'info', 800);
    },

    redo() {
        if (this.redoStack.length === 0) return;
        const state = this.redoStack.pop();
        this.undoStack.push(state);
        this._restoreState(state);
        showToast('Redo', 'info', 800);
    },

    _restoreState(stateJson) {
        const state = JSON.parse(stateJson);
        this.pipeline.nodes = state.nodes;
        this.pipeline.connections = state.connections;
        this._rebuildCanvas();
    },

    // -----------------------------------------------------------------------
    // Canvas rebuild
    // -----------------------------------------------------------------------

    _rebuildCanvas() {
        this.canvas.querySelectorAll('.pipeline-node').forEach(el => el.remove());
        this.selectedNodes.clear();

        for (const [nodeId, nodeConfig] of Object.entries(this.pipeline.nodes)) {
            const typeInfo = NodeRenderer.nodeTypes[nodeConfig.type];
            const el = NodeRenderer.createNodeElement(nodeId, nodeConfig, typeInfo);
            this.canvas.appendChild(el);
        }

        ConnectionManager.fromArray(this.pipeline.connections, NodeRenderer.nodeTypes);
    },

    // -----------------------------------------------------------------------
    // Save / Load
    // -----------------------------------------------------------------------

    async savePipeline() {
        this.pipeline.connections = ConnectionManager.toArray();

        try {
            let resp;
            if (this.pipeline.id) {
                resp = await fetch(`/api/pipelines/${this.pipeline.id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.pipeline),
                });
            } else {
                resp = await fetch('/api/pipelines', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.pipeline),
                });
            }
            const data = await resp.json();
            this.pipeline.id = data.id;
            showToast('Pipeline saved', 'success');
            this._loadPipelineList();
        } catch (e) {
            showToast('Save failed: ' + e.message, 'error');
        }
    },

    async loadPipeline(pipelineId) {
        if (!pipelineId) return;

        try {
            const resp = await fetch(`/api/pipelines/${pipelineId}`);
            const data = await resp.json();
            this.pipeline = data;
            this._rebuildCanvas();
            this.undoStack = [JSON.stringify({ nodes: data.nodes, connections: data.connections })];
            this.redoStack = [];
            this.fitView();
            showToast(`Loaded: ${data.name}`, 'success');
        } catch (e) {
            showToast('Load failed: ' + e.message, 'error');
        }
    },

    newPipeline() {
        this.pipeline = { id: '', name: 'Untitled', description: '', nodes: {}, connections: [] };
        this._rebuildCanvas();
        this.panX = 0;
        this.panY = 0;
        this.zoom = 1;
        this.updateTransform();
        this.undoStack = [];
        this.redoStack = [];
        document.getElementById('pipeline-select').value = '';
    },

    async _loadPipelineList() {
        try {
            const resp = await fetch('/api/pipelines');
            const pipelines = await resp.json();
            const select = document.getElementById('pipeline-select');
            while (select.children.length > 1) select.removeChild(select.lastChild);
            for (const p of pipelines) {
                const opt = document.createElement('option');
                opt.value = p.id;
                opt.textContent = `${p.name} (${p.node_count} nodes)`;
                if (p.id === this.pipeline?.id) opt.selected = true;
                select.appendChild(opt);
            }
        } catch (e) { /* ignore */ }
    },

    async _loadPresetList() {
        try {
            const resp = await fetch('/api/presets');
            const presets = await resp.json();
            const select = document.getElementById('preset-select');
            while (select.children.length > 1) select.removeChild(select.lastChild);
            for (const p of presets) {
                const opt = document.createElement('option');
                opt.value = p.filename;
                opt.textContent = `${p.name} (${p.node_count} nodes)`;
                select.appendChild(opt);
            }
        } catch (e) { /* ignore */ }
    },

    async loadPreset(name) {
        if (!name) return;
        try {
            const resp = await fetch(`/api/presets/${name}/load`, { method: 'POST' });
            const data = await resp.json();
            if (data.error) {
                showToast(data.error, 'error');
                return;
            }
            this.pipeline = data;
            this._rebuildCanvas();
            this.undoStack = [JSON.stringify({ nodes: data.nodes, connections: data.connections })];
            this.redoStack = [];
            this._loadPipelineList();

            document.getElementById('preset-select').value = '';
            setTimeout(() => this.fitView(), 50);
            showToast(`Loaded preset: ${data.name}`, 'success');
        } catch (e) {
            showToast('Failed to load preset: ' + e.message, 'error');
        }
    },

    // -----------------------------------------------------------------------
    // Preview / Execute
    // -----------------------------------------------------------------------

    async runPreview() {
        if (!this.pipeline.id) {
            await this.savePipeline();
        }
        if (!this.pipeline.id) return;

        showToast('Running preview...', 'info');
        try {
            const resp = await fetch(`/api/pipelines/${this.pipeline.id}/preview-diff`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ n: 20 }),
            });
            const data = await resp.json();

            if (data.error) {
                showToast(data.error, 'error');
                return;
            }

            // Show the diff view in the config panel
            this.deselectAll();
            PreviewDiffView.render(data);

            showToast(`Preview: best sample edited by ${data.edit_count}/${data.total_nodes} nodes`, 'success');
        } catch (e) {
            showToast('Preview failed: ' + e.message, 'error');
        }
    },

    async refreshNodePreview(nodeId) {
        if (!this.pipeline.id) return;
        try {
            const resp = await fetch(`/api/nodes/${this.pipeline.id}/${nodeId}/preview`, {
                method: 'POST',
            });
            const data = await resp.json();
            NodeRenderer.updateNodePreview(nodeId, data);
            if (this.selectedNodes.has(nodeId)) {
                NodeRenderer.updatePreviewPanel(data);
            }
        } catch (e) { /* silent */ }
    },

    async runExecute() {
        if (!this.pipeline.id) {
            await this.savePipeline();
        }
        if (!this.pipeline.id) return;

        try {
            const resp = await fetch(`/api/pipelines/${this.pipeline.id}/execute`, {
                method: 'POST',
            });
            const data = await resp.json();
            showToast(`Pipeline running: ${data.task_id}`, 'info');
            window.location.href = `/batch?task=${data.task_id}&pipeline=${this.pipeline.id}`;
        } catch (e) {
            showToast('Execute failed: ' + e.message, 'error');
        }
    },
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => editor.init());
