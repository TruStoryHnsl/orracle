/**
 * nodes.js — Node DOM rendering, config panel auto-generation from params_schema
 */

const NodeRenderer = {
    nodeTypes: {},  // Loaded from API: {type_name: type_info}
    previewDebounce: null,

    async loadTypes() {
        const resp = await fetch('/api/nodes/types');
        const data = await resp.json();
        this.nodeTypes = {};
        for (const t of data.types) {
            this.nodeTypes[t.type] = t;
        }
        return data;
    },

    // Build the palette sidebar from loaded types
    buildPalette(container) {
        container.textContent = '';
        const categories = {};
        for (const [type, info] of Object.entries(this.nodeTypes)) {
            const cat = info.category || 'other';
            if (!categories[cat]) categories[cat] = [];
            categories[cat].push(info);
        }

        const catOrder = ['source', 'text', 'encoding', 'video'];
        const sorted = Object.keys(categories).sort((a, b) => {
            const ai = catOrder.indexOf(a);
            const bi = catOrder.indexOf(b);
            return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
        });

        for (const cat of sorted) {
            const catEl = document.createElement('div');
            catEl.className = 'palette-category';
            catEl.textContent = cat;
            container.appendChild(catEl);

            for (const info of categories[cat]) {
                const node = document.createElement('div');
                node.className = 'palette-node';
                node.dataset.type = info.type;
                node.draggable = true;

                const nameDiv = document.createElement('div');
                nameDiv.textContent = info.label;
                node.appendChild(nameDiv);

                const descDiv = document.createElement('div');
                descDiv.className = 'node-desc';
                descDiv.textContent = info.description;
                node.appendChild(descDiv);

                node.addEventListener('dragstart', (e) => {
                    e.dataTransfer.setData('node-type', info.type);
                    e.dataTransfer.effectAllowed = 'copy';
                });
                // Double-click palette to add at canvas center
                node.addEventListener('dblclick', () => {
                    if (!window.editor) return;
                    const vp = document.getElementById('canvas-viewport');
                    const rect = vp.getBoundingClientRect();
                    const x = editor._snapToGrid((rect.width / 2 - editor.panX) / editor.zoom);
                    const y = editor._snapToGrid((rect.height / 2 - editor.panY) / editor.zoom);
                    editor.addNode(info.type, { x, y });
                });
                container.appendChild(node);
            }
        }

        document.getElementById('node-count').textContent = Object.keys(this.nodeTypes).length;
    },

    // Create a node DOM element on the canvas
    createNodeElement(nodeId, nodeConfig, typeInfo) {
        const el = document.createElement('div');
        el.className = 'pipeline-node';
        el.dataset.id = nodeId;
        el.dataset.type = nodeConfig.type;
        if (!nodeConfig.enabled) el.classList.add('disabled');

        const pos = nodeConfig.position || { x: 100, y: 100 };
        el.style.left = pos.x + 'px';
        el.style.top = pos.y + 'px';

        // Header
        const header = document.createElement('div');
        header.className = 'node-header';

        const headerLabel = document.createElement('span');
        headerLabel.textContent = typeInfo ? typeInfo.label : nodeConfig.type;
        header.appendChild(headerLabel);

        const headerCat = document.createElement('span');
        headerCat.className = 'node-category';
        headerCat.textContent = typeInfo ? typeInfo.category : '';
        header.appendChild(headerCat);

        el.appendChild(header);

        // Ports
        const portsContainer = document.createElement('div');
        portsContainer.className = 'node-ports';

        const inputsDiv = document.createElement('div');
        inputsDiv.className = 'node-inputs';
        const outputsDiv = document.createElement('div');
        outputsDiv.className = 'node-outputs';

        if (typeInfo) {
            for (const port of (typeInfo.inputs || [])) {
                inputsDiv.appendChild(this._createPort(nodeId, port, 'in'));
            }
            for (const port of (typeInfo.outputs || [])) {
                outputsDiv.appendChild(this._createPort(nodeId, port, 'out'));
            }
        }

        portsContainer.appendChild(inputsDiv);
        portsContainer.appendChild(outputsDiv);
        el.appendChild(portsContainer);

        // Preview badge
        const badge = document.createElement('div');
        badge.className = 'node-preview-badge';
        badge.id = `preview-${nodeId}`;
        badge.textContent = '';
        el.appendChild(badge);

        // Selection — left-click only, don't stopPropagation so canvas drag handler fires
        el.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return;  // Only left-click selects/drags
            if (e.target.classList.contains('port-dot')) return;
            if (window.editor) editor.selectNode(nodeId, e.shiftKey);
        });

        // Double-click to open config
        el.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            if (window.editor) editor.openConfig(nodeId);
        });

        return el;
    },

    _createPort(nodeId, portInfo, dir) {
        const port = document.createElement('div');
        port.className = 'port';

        const dot = document.createElement('div');
        dot.className = 'port-dot';
        dot.dataset.port = portInfo.name;
        dot.dataset.type = portInfo.type;
        dot.dataset.dir = dir;
        dot.title = `${portInfo.name} (${portInfo.type})${portInfo.description ? ': ' + portInfo.description : ''}`;

        const label = document.createElement('span');
        label.textContent = portInfo.name;

        // Port drag events
        dot.addEventListener('mousedown', (e) => {
            e.stopPropagation();
            e.preventDefault();

            const pos = ConnectionManager.getPortPosition(nodeId, portInfo.name, dir === 'out');
            if (pos) {
                ConnectionManager.startDrag(nodeId, portInfo.name, portInfo.type, dir === 'out', pos.x, pos.y);
            }
        });

        if (dir === 'in') {
            port.appendChild(dot);
            port.appendChild(label);
        } else {
            port.appendChild(label);
            port.appendChild(dot);
        }
        return port;
    },

    // Open the config panel for a node
    openConfigPanel(nodeId, nodeConfig, typeInfo) {
        const titleEl = document.getElementById('config-title');
        const bodyEl = document.getElementById('config-body');
        const previewSection = document.getElementById('preview-section');

        titleEl.textContent = typeInfo ? typeInfo.label : nodeConfig.type;
        bodyEl.textContent = '';

        if (!typeInfo || !typeInfo.params_schema || !typeInfo.params_schema.properties) {
            const msg = document.createElement('p');
            msg.className = 'text-muted';
            msg.textContent = 'No configurable parameters';
            bodyEl.appendChild(msg);
            previewSection.style.display = 'none';
            return;
        }

        const schema = typeInfo.params_schema;
        const params = nodeConfig.params || {};

        // Enabled toggle
        const enabledGroup = document.createElement('div');
        enabledGroup.className = 'form-group';
        const enabledLabel = document.createElement('label');
        const enabledInput = document.createElement('input');
        enabledInput.type = 'checkbox';
        enabledInput.checked = nodeConfig.enabled !== false;
        enabledInput.style.cssText = 'width:auto;margin-right:0.5rem';
        enabledInput.addEventListener('change', (e) => {
            if (window.editor) editor.updateNodeConfig(nodeId, { enabled: e.target.checked });
        });
        enabledLabel.appendChild(enabledInput);
        enabledLabel.appendChild(document.createTextNode('Enabled'));
        enabledGroup.appendChild(enabledLabel);
        bodyEl.appendChild(enabledGroup);

        const section = document.createElement('div');
        section.className = 'config-section';
        const sTitle = document.createElement('div');
        sTitle.className = 'config-section-title';
        sTitle.textContent = 'Parameters';
        section.appendChild(sTitle);

        for (const [key, prop] of Object.entries(schema.properties)) {
            const group = document.createElement('div');
            group.className = 'form-group';

            const label = document.createElement('label');
            label.textContent = prop.title || key;
            if (prop.description) {
                label.title = prop.description;
            }
            group.appendChild(label);

            const value = params[key] !== undefined ? params[key] : prop.default;
            const input = this._createControl(key, prop, value, nodeId);
            group.appendChild(input);

            section.appendChild(group);
        }

        bodyEl.appendChild(section);
        previewSection.style.display = 'block';
    },

    _createControl(key, prop, value, nodeId) {
        let input;

        if (prop.type === 'boolean') {
            const wrap = document.createElement('div');
            wrap.style.cssText = 'display:flex;align-items:center;gap:0.5rem';
            input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = !!value;
            input.style.width = 'auto';
            const label = document.createElement('span');
            label.textContent = value ? 'On' : 'Off';
            label.style.cssText = 'font-size:0.8rem;color:var(--text-muted)';
            input.addEventListener('change', () => {
                label.textContent = input.checked ? 'On' : 'Off';
                this._onParamChange(nodeId, key, input.checked);
            });
            wrap.appendChild(input);
            wrap.appendChild(label);
            return wrap;
        }

        if (prop.enum) {
            input = document.createElement('select');
            for (const opt of prop.enum) {
                const option = document.createElement('option');
                option.value = opt;
                option.textContent = opt;
                if (opt === value) option.selected = true;
                input.appendChild(option);
            }
            input.addEventListener('change', () => {
                this._onParamChange(nodeId, key, input.value);
            });
            return input;
        }

        if (prop.type === 'number' && prop.minimum !== undefined && prop.maximum !== undefined) {
            const wrap = document.createElement('div');
            wrap.style.cssText = 'display:flex;align-items:center;gap:0.5rem';
            input = document.createElement('input');
            input.type = 'range';
            input.min = prop.minimum;
            input.max = prop.maximum;
            input.step = prop.maximum <= 1 ? '0.01' : '1';
            input.value = value !== undefined ? value : prop.default || prop.minimum;
            const display = document.createElement('span');
            display.style.cssText = 'font-size:0.8rem;color:var(--text-secondary);min-width:3em';
            display.textContent = input.value;
            input.addEventListener('input', () => {
                display.textContent = input.value;
                this._onParamChange(nodeId, key, parseFloat(input.value));
            });
            wrap.appendChild(input);
            wrap.appendChild(display);
            return wrap;
        }

        if (prop.type === 'integer') {
            input = document.createElement('input');
            input.type = 'number';
            input.value = value !== undefined ? value : (prop.default || 0);
            if (prop.minimum !== undefined) input.min = prop.minimum;
            if (prop.maximum !== undefined) input.max = prop.maximum;
            input.addEventListener('change', () => {
                this._onParamChange(nodeId, key, parseInt(input.value) || 0);
            });
            return input;
        }

        if (prop.type === 'number') {
            input = document.createElement('input');
            input.type = 'number';
            input.step = 'any';
            input.value = value !== undefined ? value : (prop.default || 0);
            input.addEventListener('change', () => {
                this._onParamChange(nodeId, key, parseFloat(input.value) || 0);
            });
            return input;
        }

        if (prop.format === 'path') {
            input = document.createElement('input');
            input.type = 'text';
            input.value = value || '';
            input.placeholder = '/path/to/directory';
            input.addEventListener('change', () => {
                this._onParamChange(nodeId, key, input.value);
            });
            return input;
        }

        // Default: text input
        if (value && typeof value === 'string' && value.includes('\n')) {
            input = document.createElement('textarea');
            input.rows = 4;
        } else {
            input = document.createElement('input');
            input.type = 'text';
        }
        input.value = value || prop.default || '';
        input.addEventListener('change', () => {
            this._onParamChange(nodeId, key, input.value);
        });
        return input;
    },

    _onParamChange(nodeId, key, value) {
        if (window.editor) {
            editor.updateNodeParam(nodeId, key, value);
        }
        clearTimeout(this.previewDebounce);
        this.previewDebounce = setTimeout(() => {
            if (window.editor) editor.refreshNodePreview(nodeId);
        }, 300);
    },

    // Update the mini preview badge on a node
    updateNodePreview(nodeId, previewData) {
        const badge = document.getElementById(`preview-${nodeId}`);
        if (!badge) return;

        if (!previewData || previewData.status === 'error') {
            badge.textContent = previewData ? 'Error' : '';
            badge.style.color = 'var(--error)';
            return;
        }

        const parts = [];
        for (const [key, val] of Object.entries(previewData)) {
            if (key.endsWith('_count') && typeof val === 'number') {
                const label = key.replace('_count', '');
                parts.push(`${label}: ${val}`);
            }
        }
        if (previewData.metrics) {
            const m = previewData.metrics;
            if (m.total !== undefined) parts.push(`n=${m.total}`);
            if (m.passed !== undefined) parts.push(`pass=${m.passed}`);
            if (m.rejected !== undefined) parts.push(`rej=${m.rejected}`);
        }

        badge.textContent = parts.slice(0, 3).join(' | ');
        badge.style.color = 'var(--text-muted)';
    },

    // Update preview section in config panel
    updatePreviewPanel(previewData) {
        const content = document.getElementById('preview-content');
        if (!content) return;
        content.textContent = '';

        if (!previewData) {
            const msg = document.createElement('p');
            msg.className = 'text-muted';
            msg.textContent = 'No preview data';
            content.appendChild(msg);
            return;
        }

        if (previewData.status === 'error') {
            const errMsg = document.createElement('p');
            errMsg.className = 'text-error';
            errMsg.textContent = previewData.error || 'Error';
            content.appendChild(errMsg);
            return;
        }

        // Show metrics
        if (previewData.metrics) {
            const metricsDiv = document.createElement('div');
            metricsDiv.className = 'preview-sample';
            const lines = [];
            for (const [k, v] of Object.entries(previewData.metrics)) {
                if (typeof v === 'object') {
                    lines.push(`${k}:`);
                    for (const [sk, sv] of Object.entries(v)) {
                        lines.push(`  ${sk}: ${sv}`);
                    }
                } else {
                    lines.push(`${k}: ${v}`);
                }
            }
            metricsDiv.textContent = lines.join('\n');
            content.appendChild(metricsDiv);
        }

        // Show text samples
        for (const key of Object.keys(previewData)) {
            if (key.endsWith('_samples') && Array.isArray(previewData[key])) {
                const label = document.createElement('div');
                label.className = 'config-section-title';
                label.textContent = key.replace('_samples', '') + ' samples';
                content.appendChild(label);

                for (const sample of previewData[key]) {
                    const div = document.createElement('div');
                    div.className = 'preview-sample';
                    div.textContent = sample.text || JSON.stringify(sample);
                    content.appendChild(div);
                }
            }
        }
    },
};
