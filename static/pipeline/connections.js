/**
 * connections.js — SVG bezier connections between node ports
 * Handles: drawing, dragging new connections, type validation, selection
 */

const ConnectionManager = {
    svg: null,
    connections: [],
    tempPath: null,
    dragging: null,
    selectedConnection: null,

    init(svgElement) {
        this.svg = svgElement;
        this.connections = [];
    },

    getPortPosition(nodeId, portName, isOutput) {
        const selector = `.pipeline-node[data-id="${nodeId}"] .port-dot[data-port="${portName}"][data-dir="${isOutput ? 'out' : 'in'}"]`;
        const dot = document.querySelector(selector);
        if (!dot) return null;

        const dotRect = dot.getBoundingClientRect();
        const canvas = document.getElementById('canvas');
        const canvasRect = canvas.getBoundingClientRect();
        const style = window.getComputedStyle(canvas);
        const matrix = new DOMMatrix(style.transform);

        const x = (dotRect.left + dotRect.width / 2 - canvasRect.left) / matrix.a;
        const y = (dotRect.top + dotRect.height / 2 - canvasRect.top) / matrix.d;

        return { x, y };
    },

    bezierPath(x1, y1, x2, y2) {
        const dx = Math.abs(x2 - x1);
        const offset = Math.max(50, dx * 0.4);
        return `M ${x1} ${y1} C ${x1 + offset} ${y1}, ${x2 - offset} ${y2}, ${x2} ${y2}`;
    },

    portColor(portType) {
        const colors = {
            text: '#8bb4e7',
            text_batch: '#7fc97f',
            files: '#e6ab6a',
            metrics: '#d4a0ff',
            tokens: '#ff9999',
            video: '#ff6b6b',
            frames: '#ffd93d',
            audio: '#6bcfff',
        };
        return colors[portType] || '#666';
    },

    redraw() {
        // Clear all paths except temp
        this.svg.querySelectorAll('path:not(.temp)').forEach(p => p.remove());

        for (const conn of this.connections) {
            const srcPos = this.getPortPosition(conn.srcNode, conn.srcPort, true);
            const dstPos = this.getPortPosition(conn.dstNode, conn.dstPort, false);
            if (!srcPos || !dstPos) continue;

            const d = this.bezierPath(srcPos.x, srcPos.y, dstPos.x, dstPos.y);
            const color = this.portColor(conn.portType || 'text_batch');

            // Invisible wide hit-target path (12px)
            const hitPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            hitPath.setAttribute('d', d);
            hitPath.setAttribute('stroke', 'transparent');
            hitPath.setAttribute('stroke-width', '12');
            hitPath.setAttribute('fill', 'none');
            hitPath.style.pointerEvents = 'stroke';
            hitPath.style.cursor = 'pointer';
            hitPath.dataset.connectionId = conn.id;

            // Visible path
            const visPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            visPath.setAttribute('d', d);
            visPath.setAttribute('stroke', color);
            visPath.setAttribute('stroke-width', this.selectedConnection === conn.id ? '3' : '2');
            visPath.setAttribute('fill', 'none');
            visPath.style.pointerEvents = 'none'; // Hit target handles clicks
            visPath.dataset.connectionId = conn.id;
            visPath.classList.add('conn-visible');

            if (this.selectedConnection === conn.id) {
                visPath.classList.add('selected');
            }

            // Hover + click on hit target
            hitPath.addEventListener('mouseenter', () => {
                if (this.selectedConnection !== conn.id) {
                    visPath.setAttribute('stroke-width', '3');
                    visPath.style.filter = 'brightness(1.4)';
                }
            });
            hitPath.addEventListener('mouseleave', () => {
                if (this.selectedConnection !== conn.id) {
                    visPath.setAttribute('stroke-width', '2');
                    visPath.style.filter = '';
                }
            });
            hitPath.addEventListener('click', (e) => {
                e.stopPropagation();
                this.selectConnection(conn.id);
            });

            this.svg.appendChild(visPath);
            this.svg.appendChild(hitPath);
            conn.path = visPath;
        }

        // Mark connected ports
        document.querySelectorAll('.port-dot.connected').forEach(d => d.classList.remove('connected'));
        for (const conn of this.connections) {
            const srcDot = document.querySelector(`.pipeline-node[data-id="${conn.srcNode}"] .port-dot[data-port="${conn.srcPort}"][data-dir="out"]`);
            const dstDot = document.querySelector(`.pipeline-node[data-id="${conn.dstNode}"] .port-dot[data-port="${conn.dstPort}"][data-dir="in"]`);
            if (srcDot) srcDot.classList.add('connected');
            if (dstDot) dstDot.classList.add('connected');
        }
    },

    startDrag(nodeId, portName, portType, isOutput, startX, startY) {
        this.dragging = {
            srcNode: nodeId,
            srcPort: portName,
            srcType: portType,
            isOutput: isOutput,
            startX, startY,
        };

        this.tempPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        this.tempPath.classList.add('temp');
        this.tempPath.setAttribute('stroke', this.portColor(portType));
        this.tempPath.setAttribute('stroke-width', '2');
        this.tempPath.setAttribute('fill', 'none');
        this.svg.appendChild(this.tempPath);

        this._showCompatible(portType, isOutput);
    },

    updateDrag(canvasX, canvasY) {
        if (!this.dragging || !this.tempPath) return;

        const { startX, startY, isOutput } = this.dragging;
        if (isOutput) {
            this.tempPath.setAttribute('d', this.bezierPath(startX, startY, canvasX, canvasY));
        } else {
            this.tempPath.setAttribute('d', this.bezierPath(canvasX, canvasY, startX, startY));
        }
    },

    endDrag(targetNodeId, targetPortName, targetPortType, targetIsOutput) {
        if (!this.dragging) return null;

        if (this.tempPath) {
            this.tempPath.remove();
            this.tempPath = null;
        }
        this._clearCompatible();

        const drag = this.dragging;
        this.dragging = null;

        if (drag.isOutput === targetIsOutput) return null;
        if (drag.srcNode === targetNodeId) return null;
        if (drag.srcType !== targetPortType) return null;

        let srcNode, srcPort, dstNode, dstPort;
        if (drag.isOutput) {
            srcNode = drag.srcNode;
            srcPort = drag.srcPort;
            dstNode = targetNodeId;
            dstPort = targetPortName;
        } else {
            srcNode = targetNodeId;
            srcPort = targetPortName;
            dstNode = drag.srcNode;
            dstPort = drag.srcPort;
        }

        const exists = this.connections.some(c =>
            c.srcNode === srcNode && c.srcPort === srcPort &&
            c.dstNode === dstNode && c.dstPort === dstPort
        );
        if (exists) return null;

        // Inputs accept one connection — remove existing
        this.connections = this.connections.filter(c =>
            !(c.dstNode === dstNode && c.dstPort === dstPort)
        );

        const conn = {
            id: `conn_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
            srcNode, srcPort, dstNode, dstPort,
            portType: drag.srcType,
        };
        this.connections.push(conn);
        this.redraw();
        return conn;
    },

    cancelDrag() {
        if (this.tempPath) {
            this.tempPath.remove();
            this.tempPath = null;
        }
        this._clearCompatible();
        this.dragging = null;
    },

    selectConnection(connId) {
        this.selectedConnection = connId;
        this.redraw();
        if (window.editor) editor.onConnectionSelected(connId);
    },

    deselectAll() {
        this.selectedConnection = null;
        this.redraw();
    },

    removeConnection(connId) {
        this.connections = this.connections.filter(c => c.id !== connId);
        if (this.selectedConnection === connId) this.selectedConnection = null;
        this.redraw();
    },

    removeNodeConnections(nodeId) {
        this.connections = this.connections.filter(c =>
            c.srcNode !== nodeId && c.dstNode !== nodeId
        );
        this.redraw();
    },

    toArray() {
        return this.connections.map(c => [c.srcNode, c.srcPort, c.dstNode, c.dstPort]);
    },

    fromArray(arr, nodeTypes) {
        this.connections = arr.map((c, i) => ({
            id: `conn_${i}`,
            srcNode: c[0],
            srcPort: c[1],
            dstNode: c[2],
            dstPort: c[3],
            portType: this._inferPortType(c[0], c[1], nodeTypes),
        }));
        this.redraw();
    },

    _inferPortType(nodeId, portName, nodeTypes) {
        if (!nodeTypes) return 'text_batch';
        const nodeEl = document.querySelector(`.pipeline-node[data-id="${nodeId}"]`);
        if (!nodeEl) return 'text_batch';
        const dot = nodeEl.querySelector(`.port-dot[data-port="${portName}"]`);
        return dot ? (dot.dataset.type || 'text_batch') : 'text_batch';
    },

    _showCompatible(portType, isOutput) {
        const dir = isOutput ? 'in' : 'out';
        document.querySelectorAll(`.port-dot[data-dir="${dir}"]`).forEach(dot => {
            if (dot.dataset.type === portType) {
                dot.classList.add('compatible');
            } else {
                dot.classList.add('incompatible');
            }
        });
    },

    _clearCompatible() {
        document.querySelectorAll('.port-dot.compatible, .port-dot.incompatible').forEach(d => {
            d.classList.remove('compatible', 'incompatible');
        });
    },
};
