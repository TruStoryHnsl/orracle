/**
 * preview.js — Pipeline preview diff view
 * Shows the most-edited chunk with node-by-node transformation diffs
 */

const PreviewDiffView = {

    render(data) {
        const title = document.getElementById('config-title');
        const body = document.getElementById('config-body');
        const previewSection = document.getElementById('preview-section');
        previewSection.style.display = 'none';

        title.textContent = 'Pipeline Preview';
        body.textContent = '';

        // --- Header stats ---
        const header = this._el('div', 'preview-diff-header');

        const source = data.metadata?.relative || data.metadata?.name || `Chunk #${data.chunk_index}`;
        const sourceEl = this._el('div', 'preview-source-name');
        sourceEl.textContent = source;
        header.appendChild(sourceEl);

        const statsRow = this._el('div', 'preview-stats-row');
        this._addStat(statsRow, 'Edits', `${data.edit_count}/${data.total_nodes} nodes`);
        this._addStat(statsRow, 'Chars', `${data.original_chars.toLocaleString()} \u2192 ${data.final_chars.toLocaleString()}`);
        const delta = data.final_chars - data.original_chars;
        const deltaStr = delta >= 0 ? `+${delta.toLocaleString()}` : delta.toLocaleString();
        this._addStat(statsRow, 'Delta', deltaStr, delta > 0 ? 'var(--success)' : delta < 0 ? 'var(--error)' : 'var(--text-muted)');
        header.appendChild(statsRow);

        // Distribution line
        if (data.stats) {
            const distEl = this._el('div', 'preview-dist');
            distEl.textContent = `${data.stats.total_chunks} chunks sampled \u00b7 avg ${data.stats.avg_edits} edits \u00b7 ${data.stats.chunks_with_zero_edits} untouched`;
            header.appendChild(distEl);
        }

        body.appendChild(header);

        // --- Tab bar: Timeline | Before | After ---
        const tabs = this._el('div', 'preview-tabs');
        const tabTimeline = this._tabBtn('Timeline', true, () => this._showTab('timeline'));
        const tabBefore = this._tabBtn('Before', false, () => this._showTab('before'));
        const tabAfter = this._tabBtn('After', false, () => this._showTab('after'));
        tabs.appendChild(tabTimeline);
        tabs.appendChild(tabBefore);
        tabs.appendChild(tabAfter);
        body.appendChild(tabs);

        // --- Tab content ---
        const content = this._el('div', 'preview-tab-content');
        content.id = 'preview-tab-content';

        // Timeline panel
        const timelinePanel = this._el('div', 'preview-tab-panel');
        timelinePanel.dataset.tab = 'timeline';
        this._buildTimeline(timelinePanel, data.timeline);
        content.appendChild(timelinePanel);

        // Before panel
        const beforePanel = this._el('div', 'preview-tab-panel');
        beforePanel.dataset.tab = 'before';
        beforePanel.style.display = 'none';
        const beforeText = this._el('div', 'preview-fulltext');
        beforeText.textContent = data.original_text;
        if (data.original_truncated) {
            const trunc = this._el('div', 'preview-truncated');
            trunc.textContent = `\u2026 truncated (${data.original_chars.toLocaleString()} chars total)`;
            beforePanel.appendChild(trunc);
        }
        beforePanel.appendChild(beforeText);
        content.appendChild(beforePanel);

        // After panel
        const afterPanel = this._el('div', 'preview-tab-panel');
        afterPanel.dataset.tab = 'after';
        afterPanel.style.display = 'none';
        const afterText = this._el('div', 'preview-fulltext');
        afterText.textContent = data.final_text;
        if (data.final_truncated) {
            const trunc = this._el('div', 'preview-truncated');
            trunc.textContent = `\u2026 truncated (${data.final_chars.toLocaleString()} chars total)`;
            afterPanel.appendChild(trunc);
        }
        afterPanel.appendChild(afterText);
        content.appendChild(afterPanel);

        body.appendChild(content);
    },

    _showTab(tabName) {
        // Toggle tab buttons
        document.querySelectorAll('.preview-tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });
        // Toggle panels
        document.querySelectorAll('.preview-tab-panel').forEach(panel => {
            panel.style.display = panel.dataset.tab === tabName ? 'block' : 'none';
        });
    },

    _buildTimeline(container, timeline) {
        if (!timeline || timeline.length === 0) {
            const empty = this._el('div', 'empty-state');
            const msg = this._el('p', 'text-muted');
            msg.textContent = 'No text processing nodes in pipeline';
            empty.appendChild(msg);
            container.appendChild(empty);
            return;
        }

        for (const entry of timeline) {
            const node = this._el('div', 'tl-node');
            if (entry.changed) {
                node.classList.add('tl-changed');
            }

            // Node header row
            const headerRow = this._el('div', 'tl-header');

            const indicator = this._el('span', 'tl-indicator');
            indicator.textContent = entry.changed ? '\u25cf' : '\u25cb';
            indicator.title = entry.changed ? 'Modified text' : 'No change';
            headerRow.appendChild(indicator);

            const label = this._el('span', 'tl-label');
            label.textContent = entry.label;
            headerRow.appendChild(label);

            if (entry.changed) {
                const delta = this._el('span', 'tl-delta');
                const removed = entry.chars_removed || 0;
                const added = entry.chars_added || 0;
                const parts = [];
                if (removed > 0) parts.push(`-${removed}`);
                if (added > 0) parts.push(`+${added}`);
                delta.textContent = parts.join(' ');
                delta.style.color = removed > added ? 'var(--error)' : 'var(--success)';
                headerRow.appendChild(delta);
            }

            node.appendChild(headerRow);

            // Diff content (collapsed by default, click to toggle)
            if (entry.changed && entry.diff && entry.diff.length > 0) {
                const diffContainer = this._el('div', 'tl-diff');
                diffContainer.style.display = 'none';

                for (const seg of entry.diff) {
                    const line = this._el('div', 'tl-diff-line');
                    if (seg.type === 'insert') {
                        line.classList.add('tl-diff-insert');
                        line.textContent = '+ ' + (seg.text || '');
                    } else if (seg.type === 'delete') {
                        line.classList.add('tl-diff-delete');
                        line.textContent = '- ' + (seg.text || '');
                    } else if (seg.type === 'header') {
                        line.classList.add('tl-diff-header');
                        line.textContent = seg.text;
                    } else {
                        line.classList.add('tl-diff-context');
                        line.textContent = '  ' + (seg.text || '');
                    }
                    diffContainer.appendChild(line);
                }

                node.appendChild(diffContainer);

                // Click header to expand/collapse
                headerRow.style.cursor = 'pointer';
                headerRow.addEventListener('click', () => {
                    const visible = diffContainer.style.display !== 'none';
                    diffContainer.style.display = visible ? 'none' : 'block';
                    node.classList.toggle('tl-expanded', !visible);
                });
            }

            container.appendChild(node);
        }
    },

    // Helpers
    _el(tag, className) {
        const el = document.createElement(tag);
        if (className) el.className = className;
        return el;
    },

    _tabBtn(text, active, onClick) {
        const btn = this._el('button', 'preview-tab-btn');
        btn.textContent = text;
        btn.dataset.tab = text.toLowerCase();
        if (active) btn.classList.add('active');
        btn.addEventListener('click', onClick);
        return btn;
    },

    _addStat(container, label, value, color) {
        const stat = this._el('div', 'preview-stat');
        const lbl = this._el('span', 'preview-stat-label');
        lbl.textContent = label;
        const val = this._el('span', 'preview-stat-value');
        val.textContent = value;
        if (color) val.style.color = color;
        stat.appendChild(lbl);
        stat.appendChild(val);
        container.appendChild(stat);
    },
};
