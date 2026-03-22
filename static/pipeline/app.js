/* Orracle — Main app controller */

const app = {
    sourceEntries: [],
    rules: [],
    selectedText: '',
    filteredRules: [],
    currentLibrary: 'nifty_archive',
    previewData: null,
    previewIndex: 0,
    eventSource: null,

    // --- Tabs ---
    switchTab(tab) {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        document.querySelector('[data-tab="' + tab + '"]').classList.add('active');
        document.getElementById('tab-' + tab).classList.add('active');
    },

    // --- Source ---
    async scanSource() {
        const dir = document.getElementById('source-dir').value.trim();
        if (!dir) return;
        const statsEl = document.getElementById('source-stats');
        statsEl.textContent = 'Scanning...';
        try {
            const resp = await fetch('/api/source/scan', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({directory: dir}),
            });
            const data = await resp.json();
            if (data.error) {
                statsEl.textContent = data.error;
                statsEl.style.color = 'var(--error)';
                return;
            }
            this.sourceEntries = data.entries || [];
            statsEl.textContent = data.total_entries.toLocaleString() + ' stories, ' +
                data.categories.length + ' categories';
            statsEl.style.color = 'var(--accent)';
            document.getElementById('btn-run').disabled = false;
            showToast('Scanned ' + data.total_entries.toLocaleString() + ' stories', 'success');
        } catch (e) {
            statsEl.textContent = 'Scan failed';
            statsEl.style.color = 'var(--error)';
        }
    },

    // --- Preview ---
    async previewRandom() {
        const dir = document.getElementById('source-dir').value.trim();
        if (!dir) { showToast('Set a source directory first', 'error'); return; }
        document.getElementById('preview-story-name').textContent = 'Loading...';

        const resp = await fetch('/api/preview/sample', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({directory: dir, random: true}),
        });
        const data = await resp.json();
        if (data.error) { showToast(data.error, 'error'); return; }
        this.previewData = data;
        this.renderPreview();
    },

    async previewNext() {
        const dir = document.getElementById('source-dir').value.trim();
        if (!dir) return;
        this.previewIndex++;
        document.getElementById('preview-story-name').textContent = 'Loading...';

        const resp = await fetch('/api/preview/sample', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({directory: dir, index: this.previewIndex}),
        });
        const data = await resp.json();
        if (data.error) { showToast(data.error, 'error'); return; }
        this.previewData = data;
        this.renderPreview();
    },

    renderPreview() {
        const d = this.previewData;
        if (!d) return;

        document.getElementById('preview-story-name').textContent =
            d.name + ' (' + (d.category || 'unknown') + ')';

        const stages = d.stages || [];
        const reduction = d.original_chars > 0
            ? Math.round((1 - d.final_chars / d.original_chars) * 100) : 0;
        const statsEl = document.getElementById('preview-stats');
        statsEl.textContent = d.original_chars.toLocaleString() + ' \u2192 ' +
            d.final_chars.toLocaleString() + ' chars (' + reduction + '% removed)' +
            (d.quality_passed ? '' : ' \u2014 REJECTED');
        statsEl.style.color = d.quality_passed ? '' : 'var(--error)';

        // Stage bar
        const bar = document.getElementById('stage-bar');
        bar.textContent = '';
        for (let i = 0; i < stages.length; i++) {
            const s = stages[i];
            const btn = document.createElement('button');
            btn.className = 'stage-btn' + (s.changed ? ' changed' : '');
            btn.textContent = s.label;
            if (s.changed) {
                const tag = document.createElement('span');
                tag.className = 'stage-delta';
                const delta = s.chars_after - s.chars_before;
                tag.textContent = delta > 0 ? ('+' + delta) : String(delta);
                btn.appendChild(tag);
            }
            const idx = i;
            btn.addEventListener('click', function() { app.showStage(idx); });
            bar.appendChild(btn);
        }

        // Show final by default
        this.showStage(stages.length - 1);
    },

    showStage(idx) {
        const d = this.previewData;
        if (!d) return;
        const stages = d.stages || [];
        const stage = stages[idx];
        if (!stage) return;

        document.querySelectorAll('.stage-btn').forEach(function(b, i) {
            b.classList.toggle('active', i === idx);
        });

        document.getElementById('before-label').textContent = stage.before_label || 'Before';
        document.getElementById('after-label').textContent = stage.label || 'After';
        document.getElementById('before-chars').textContent = stage.chars_before.toLocaleString() + ' chars';
        document.getElementById('after-chars').textContent = stage.chars_after.toLocaleString() + ' chars';

        var MAX = 50000;
        document.getElementById('preview-before').textContent =
            stage.text_before.length > MAX ? stage.text_before.substring(0, MAX) + '\n\n[... truncated ...]' : stage.text_before;
        document.getElementById('preview-after').textContent =
            stage.text_after.length > MAX ? stage.text_after.substring(0, MAX) + '\n\n[... truncated ...]' : stage.text_after;
    },

    // --- Rules ---
    async initRules() {
        const resp = await fetch('/api/rules/libraries');
        const libs = await resp.json();
        const select = document.getElementById('library-select');
        for (const lib of libs) {
            const opt = document.createElement('option');
            opt.value = lib.name;
            opt.textContent = lib.name + ' (' + lib.rule_count + ')';
            select.appendChild(opt);
        }
        if (libs.length > 0) {
            select.value = libs[0].name;
            this.loadRuleLibrary(libs[0].name);
        }
    },

    async loadRuleLibrary(name) {
        if (!name) return;
        this.currentLibrary = name;
        const resp = await fetch('/api/rules/library/' + name);
        const data = await resp.json();
        this.rules = data.rules || [];
        this.filteredRules = this.rules.slice();
        document.getElementById('rule-count-badge').textContent = '(' + this.rules.length + ')';

        const cats = [];
        const seen = {};
        for (const r of this.rules) {
            const c = r.category || '';
            if (!seen[c]) { seen[c] = true; cats.push(c); }
        }
        cats.sort();
        const catSelect = document.getElementById('category-filter');
        while (catSelect.children.length > 1) catSelect.removeChild(catSelect.lastChild);
        for (const cat of cats) {
            const opt = document.createElement('option');
            opt.value = cat;
            opt.textContent = cat;
            catSelect.appendChild(opt);
        }
        this.renderRules();
    },

    filterRules(q) {
        q = q.toLowerCase();
        this.filteredRules = this.rules.filter(function(r) {
            return (r.name||'').toLowerCase().indexOf(q) >= 0 ||
                (r.pattern||'').toLowerCase().indexOf(q) >= 0 ||
                (r.description||'').toLowerCase().indexOf(q) >= 0;
        });
        this.renderRules();
    },

    filterByCategory(cat) {
        this.filteredRules = cat ? this.rules.filter(function(r) { return r.category === cat; }) : this.rules.slice();
        this.renderRules();
    },

    renderRules() {
        const container = document.getElementById('rule-list');
        const enabled = this.rules.filter(function(r) { return r.enabled; }).length;
        document.getElementById('rule-stats').textContent = enabled + '/' + this.rules.length + ' enabled';
        container.textContent = '';

        if (!this.filteredRules.length) {
            var empty = document.createElement('div');
            empty.className = 'empty-state';
            empty.textContent = 'No rules';
            container.appendChild(empty);
            return;
        }

        var table = document.createElement('table');
        var thead = document.createElement('thead');
        var hrow = document.createElement('tr');
        ['', 'Name', 'Category', 'Action', 'Pattern'].forEach(function(h) {
            var th = document.createElement('th');
            th.textContent = h;
            hrow.appendChild(th);
        });
        thead.appendChild(hrow);
        table.appendChild(thead);

        var tbody = document.createElement('tbody');
        var self = this;
        this.filteredRules.forEach(function(r) {
            var tr = document.createElement('tr');
            tr.style.opacity = r.enabled ? '1' : '0.5';

            var tdCk = document.createElement('td');
            var ck = document.createElement('input');
            ck.type = 'checkbox';
            ck.checked = r.enabled;
            ck.style.width = 'auto';
            ck.addEventListener('change', function() {
                r.enabled = ck.checked;
                self.renderRules();
            });
            tdCk.appendChild(ck);
            tr.appendChild(tdCk);

            var tdName = document.createElement('td');
            tdName.textContent = r.name || r.id;
            tr.appendChild(tdName);

            var tdCat = document.createElement('td');
            var badge = document.createElement('span');
            badge.className = 'badge';
            badge.textContent = r.category || '';
            tdCat.appendChild(badge);
            tr.appendChild(tdCat);

            var tdAct = document.createElement('td');
            tdAct.textContent = r.action || 'strip_line';
            tdAct.className = 'text-muted';
            tr.appendChild(tdAct);

            var tdPat = document.createElement('td');
            tdPat.textContent = (r.pattern || '').substring(0, 50);
            tdPat.style.fontSize = '0.75rem';
            tdPat.title = r.pattern || '';
            tr.appendChild(tdPat);

            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        container.appendChild(table);
    },

    toggleRule(id, enabled) {
        var r = this.rules.find(function(r) { return r.id === id; });
        if (r) r.enabled = enabled;
        this.renderRules();
    },

    async saveRules() {
        if (!this.currentLibrary) return;
        const resp = await fetch('/api/rules/library/' + this.currentLibrary, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({rules: this.rules}),
        });
        const data = await resp.json();
        showToast('Saved ' + data.rule_count + ' rules', 'success');
    },

    async testPattern() {
        const pattern = document.getElementById('test-pattern').value;
        const sample = document.getElementById('test-sample').value;
        if (!pattern || !sample) return;
        const resp = await fetch('/api/rules/test', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({pattern: pattern, sample_text: sample}),
        });
        const data = await resp.json();
        const el = document.getElementById('test-results');
        el.textContent = '';

        if (data.error) {
            var p = document.createElement('p');
            p.className = 'text-error';
            p.textContent = 'Regex error: ' + data.error;
            el.appendChild(p);
            return;
        }

        var summary = document.createElement('p');
        summary.textContent = data.count + ' match' + (data.count !== 1 ? 'es' : '') + ' found';
        summary.style.color = data.count > 0 ? 'var(--success)' : 'var(--text-muted)';
        el.appendChild(summary);

        (data.matches || []).slice(0, 10).forEach(function(m) {
            var div = document.createElement('div');
            div.style.cssText = 'font-size:0.8rem;padding:0.2rem 0;border-bottom:1px solid var(--border-color)';
            var line = document.createElement('span');
            line.className = 'text-muted';
            line.textContent = 'Line ' + m.line + ': ';
            div.appendChild(line);
            var match = document.createElement('span');
            match.className = 'text-accent';
            match.textContent = (m.text || '').substring(0, 100);
            div.appendChild(match);
            el.appendChild(div);
        });
    },

    // --- Run pipeline ---
    async runPipeline() {
        const dir = document.getElementById('source-dir').value.trim();
        if (!dir) { showToast('Set a source directory', 'error'); return; }
        this.switchTab('run');
        document.getElementById('btn-run').disabled = true;
        document.getElementById('btn-stop').disabled = false;
        document.getElementById('run-log').textContent = '';
        document.getElementById('run-results').style.display = 'none';

        const settings = {
            directory: dir,
            min_chars: parseInt(document.getElementById('cfg-min-chars').value),
            max_chars: parseInt(document.getElementById('cfg-max-chars').value),
            min_words: parseInt(document.getElementById('cfg-min-words').value),
            min_sentences: parseInt(document.getElementById('cfg-min-sentences').value),
            max_non_ascii: parseFloat(document.getElementById('cfg-max-non-ascii').value),
            max_avg_word_len: parseFloat(document.getElementById('cfg-max-avg-word').value),
            val_ratio: parseFloat(document.getElementById('cfg-val-ratio').value),
            seed: parseInt(document.getElementById('cfg-seed').value),
            io_workers: parseInt(document.getElementById('cfg-io-threads').value),
            batch_size: parseInt(document.getElementById('cfg-batch-size').value),
            pdf_extract: document.getElementById('cfg-pdf').value === 'true',
            library: this.currentLibrary,
        };

        const resp = await fetch('/api/pipeline/run', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(settings),
        });
        const data = await resp.json();
        if (data.error) { showToast(data.error, 'error'); return; }

        this.eventSource = new EventSource('/api/pipeline/stream/' + data.task_id);
        const logEl = document.getElementById('run-log');

        this.eventSource.onmessage = function(e) {
            const msg = JSON.parse(e.data);
            if (msg.type === 'log') {
                logEl.textContent += msg.message + '\n';
                logEl.scrollTop = logEl.scrollHeight;
            } else if (msg.type === 'progress') {
                document.getElementById('run-progress').style.width = msg.percent + '%';
                document.getElementById('run-status').textContent =
                    msg.percent + '% \u2014 ' + (msg.passed || 0).toLocaleString() + ' passed';
                if (msg.eta) document.getElementById('run-eta').textContent = msg.eta;
            } else if (msg.type === 'done') {
                app.eventSource.close();
                document.getElementById('btn-run').disabled = false;
                document.getElementById('btn-stop').disabled = true;
                document.getElementById('run-progress').style.width = '100%';
                document.getElementById('run-status').textContent = msg.status;
                if (msg.results) app.showResults(msg.results);
            }
        };
    },

    stopPipeline() {
        if (this.eventSource) this.eventSource.close();
        fetch('/api/pipeline/stop', {method: 'POST'});
        document.getElementById('btn-run').disabled = false;
        document.getElementById('btn-stop').disabled = true;
        document.getElementById('run-status').textContent = 'Stopped';
    },

    showResults(r) {
        var el = document.getElementById('run-results');
        el.style.display = 'block';
        var cards = document.getElementById('result-cards');
        cards.textContent = '';

        var metrics = [
            ['Stories scanned', r.scanned], ['Stories read', r.read],
            ['Quality passed', r.passed], ['Rejected', r.rejected],
            ['Duplicates removed', r.deduped],
        ];
        var card1 = document.createElement('div');
        card1.className = 'card';
        metrics.forEach(function(m) {
            var row = document.createElement('div');
            row.className = 'metric-row';
            var label = document.createElement('span');
            label.className = 'metric-label';
            label.textContent = m[0];
            var value = document.createElement('span');
            value.className = 'metric-value';
            value.textContent = (m[1] || 0).toLocaleString();
            row.appendChild(label);
            row.appendChild(value);
            card1.appendChild(row);
        });
        cards.appendChild(card1);

        var outputs = [
            ['Train stories', r.train_count], ['Val stories', r.val_count],
            ['Train size', (r.train_size_mb || '?') + ' MB'],
            ['~Train tokens', r.train_tokens],
            ['Char reduction', (r.char_reduction || '?') + '%'],
        ];
        var card2 = document.createElement('div');
        card2.className = 'card';
        outputs.forEach(function(m) {
            var row = document.createElement('div');
            row.className = 'metric-row';
            var label = document.createElement('span');
            label.className = 'metric-label';
            label.textContent = m[0];
            var value = document.createElement('span');
            value.className = 'metric-value';
            value.textContent = typeof m[1] === 'number' ? m[1].toLocaleString() : m[1];
            row.appendChild(label);
            row.appendChild(value);
            card2.appendChild(row);
        });
        cards.appendChild(card2);
    },

    // --- Selection → Rule Builder ---
    initSelectionHandler() {
        var self = this;
        document.addEventListener('mouseup', function(e) {
            var sel = window.getSelection();
            var text = (sel ? sel.toString() : '').trim();
            var fab = document.getElementById('selection-fab');

            // Only show FAB if selection is inside a preview pane
            var node = sel.anchorNode;
            var inPreview = false;
            while (node) {
                if (node.classList && node.classList.contains('preview-text')) {
                    inPreview = true;
                    break;
                }
                node = node.parentNode;
            }

            if (text.length > 5 && inPreview) {
                self.selectedText = text;
                fab.style.display = 'flex';
                fab.style.top = (e.clientY - 50) + 'px';
                fab.style.left = (e.clientX + 10) + 'px';
            } else if (!e.target.closest('#selection-fab') && !e.target.closest('#rule-builder')) {
                fab.style.display = 'none';
            }
        });
    },

    openRuleBuilder() {
        document.getElementById('selection-fab').style.display = 'none';
        var modal = document.getElementById('rule-builder');
        modal.style.display = 'flex';

        var text = this.selectedText;
        document.getElementById('rb-selected-text').textContent = text;

        // Generate pattern suggestions
        this.generatePatternSuggestions(text);

        // Pre-fill the test sample with the current story text
        var afterEl = document.getElementById('preview-after');
        if (afterEl) {
            document.getElementById('rb-test-sample').value = afterEl.textContent.substring(0, 10000);
        }
    },

    closeRuleBuilder() {
        document.getElementById('rule-builder').style.display = 'none';
    },

    generatePatternSuggestions(text) {
        var container = document.getElementById('rb-suggestions');
        container.textContent = '';

        var suggestions = [];

        // 1. Key phrases — extract distinctive phrases
        var lines = text.split('\n').map(function(l) { return l.trim(); }).filter(function(l) { return l.length > 10; });
        if (lines.length > 0) {
            // Pick the most distinctive line
            var bestLine = lines.reduce(function(a, b) { return a.length > b.length ? a : b; });
            var escaped = bestLine.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            // Truncate for usability
            if (escaped.length > 80) escaped = escaped.substring(0, 80);
            suggestions.push({
                name: 'Key phrase (line match)',
                pattern: '(?i)' + escaped.substring(0, 60),
                action: 'strip_line',
                description: 'Strip lines containing this phrase',
            });
        }

        // 2. Paragraph match — for multi-line selections
        if (text.indexOf('\n') >= 0 || text.length > 100) {
            // Extract 2-3 key words for a paragraph-level match
            var words = text.replace(/[^a-zA-Z\s]/g, '').split(/\s+/).filter(function(w) { return w.length > 4; });
            var unique = [];
            var seen = {};
            words.forEach(function(w) {
                var lower = w.toLowerCase();
                if (!seen[lower] && lower.length > 5) {
                    seen[lower] = true;
                    unique.push(lower);
                }
            });
            if (unique.length >= 2) {
                var kw = unique.slice(0, 3);
                var paraPattern = '(?i)' + kw.map(function(w) { return '(?=.*' + w + ')'; }).join('') + '.*';
                suggestions.push({
                    name: 'Paragraph match (key words)',
                    pattern: paraPattern,
                    action: 'strip_paragraph',
                    description: 'Strip paragraphs containing: ' + kw.join(', '),
                });
            }
        }

        // 3. Above-divider pattern — strip content above --- or ***
        suggestions.push({
            name: 'Above divider (author notes)',
            pattern: '(?i)(?:' + (lines.length > 0 ? lines[0].replace(/[.*+?^${}()|[\]\\]/g, '\\$&').substring(0, 40) : 'author') + ')',
            action: 'strip_paragraph',
            description: 'Strip author-voice paragraphs above divider lines',
        });

        // 4. Email/URL in context
        if (text.match(/[a-zA-Z0-9._%+-]+@|https?:\/\//)) {
            suggestions.push({
                name: 'Contact/URL paragraph',
                pattern: '(?i)(?=.*(?:e-?mail|send|write|contact|feedback))(?=.*@).*',
                action: 'strip_paragraph',
                description: 'Strip paragraphs soliciting contact via email',
            });
        }

        // 5. Age disclaimer pattern
        if (text.match(/(?:18|21|under\s*age|minor|legal)/i)) {
            suggestions.push({
                name: 'Age disclaimer',
                pattern: '(?i)(?=.*(?:younger|under|age|minor|legal))(?=.*(?:18|21|law)).*',
                action: 'strip_paragraph',
                description: 'Strip age/legal disclaimer paragraphs',
            });
        }

        // Render suggestions
        suggestions.forEach(function(s) {
            var div = document.createElement('div');
            div.className = 'rb-suggestion';

            var header = document.createElement('div');
            header.className = 'flex justify-between items-center';

            var nameEl = document.createElement('strong');
            nameEl.textContent = s.name;
            header.appendChild(nameEl);

            var useBtn = document.createElement('button');
            useBtn.className = 'btn btn-sm btn-primary';
            useBtn.textContent = 'Use';
            useBtn.addEventListener('click', function() {
                document.getElementById('rb-pattern').value = s.pattern;
                document.getElementById('rb-action').value = s.action;
                document.getElementById('rb-name').value = s.name;
                document.getElementById('rb-description').value = s.description;
            });
            header.appendChild(useBtn);
            div.appendChild(header);

            var desc = document.createElement('div');
            desc.className = 'text-muted';
            desc.style.fontSize = '0.8rem';
            desc.textContent = s.action + ': ' + s.pattern.substring(0, 80);
            div.appendChild(desc);

            container.appendChild(div);
        });
    },

    async testRuleBuilder() {
        var pattern = document.getElementById('rb-pattern').value;
        var sample = document.getElementById('rb-test-sample').value;
        var action = document.getElementById('rb-action').value;
        if (!pattern || !sample) return;

        // Test the pattern
        var resp = await fetch('/api/rules/test', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({pattern: pattern, sample_text: sample}),
        });
        var data = await resp.json();
        var el = document.getElementById('rb-test-results');
        el.textContent = '';

        if (data.error) {
            var p = document.createElement('p');
            p.className = 'text-error';
            p.textContent = 'Regex error: ' + data.error;
            el.appendChild(p);
            return;
        }

        var summary = document.createElement('p');
        summary.textContent = data.count + ' match' + (data.count !== 1 ? 'es' : '');
        summary.style.color = data.count > 0 ? 'var(--success)' : 'var(--text-muted)';
        el.appendChild(summary);

        // Show preview: what would be removed
        if (data.count > 0 && action === 'strip_paragraph') {
            var preview = document.createElement('div');
            preview.className = 'rb-removal-preview';
            var paragraphs = sample.split(/\n\s*\n/);
            var regex = new RegExp(pattern, 'i');
            paragraphs.forEach(function(para) {
                var joined = para.replace(/\s+/g, ' ').trim();
                var div = document.createElement('div');
                div.style.cssText = 'padding:0.3rem 0.5rem;margin:0.2rem 0;border-radius:3px;font-size:0.75rem;';
                if (regex.test(joined)) {
                    div.style.background = 'rgba(255,107,107,0.15)';
                    div.style.borderLeft = '3px solid var(--error)';
                    div.textContent = '\u2717 ' + joined.substring(0, 120);
                } else {
                    div.style.opacity = '0.5';
                    div.textContent = '\u2713 ' + joined.substring(0, 120);
                }
                preview.appendChild(div);
            });
            el.appendChild(preview);
        } else if (data.count > 0) {
            (data.matches || []).slice(0, 5).forEach(function(m) {
                var div = document.createElement('div');
                div.style.cssText = 'font-size:0.8rem;padding:0.2rem 0;border-bottom:1px solid var(--border-color)';
                var line = document.createElement('span');
                line.className = 'text-muted';
                line.textContent = 'Line ' + m.line + ': ';
                div.appendChild(line);
                var match = document.createElement('span');
                match.style.color = 'var(--error)';
                match.textContent = (m.text || '').substring(0, 100);
                div.appendChild(match);
                el.appendChild(div);
            });
        }
    },

    addRuleFromBuilder() {
        var pattern = document.getElementById('rb-pattern').value.trim();
        var action = document.getElementById('rb-action').value;
        var name = document.getElementById('rb-name').value.trim() || 'custom_rule';
        var desc = document.getElementById('rb-description').value.trim();
        var category = document.getElementById('rb-category').value.trim() || 'custom';

        if (!pattern) { showToast('Pattern is required', 'error'); return; }

        var id = name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
        id = id + '_' + Date.now().toString(36);

        var rule = {
            id: id,
            name: name,
            category: category,
            enabled: true,
            priority: 100,
            action: action,
            pattern: pattern,
            description: desc,
        };

        // Add replacement field for replace action
        var replacement = document.getElementById('rb-replacement');
        if (replacement && action === 'replace') {
            rule.replacement = replacement.value;
        }

        this.rules.push(rule);
        this.filteredRules = this.rules.slice();
        this.renderRules();
        this.closeRuleBuilder();

        // Auto-save
        this.saveRules();
        showToast('Rule added: ' + name, 'success');

        // Re-run preview to show the effect
        if (this.previewData) {
            this.previewRandom();
        }
    },

    // --- Video Pipeline Monitoring ---
    async refreshVideoStatus() {
        var resp = await fetch('/api/video/status');
        var data = await resp.json();
        document.getElementById('vp-status').textContent =
            data.active ? 'Running' : 'Stopped';
        document.getElementById('vp-status').style.color =
            data.active ? 'var(--success)' : 'var(--error)';
        document.getElementById('vp-memory').textContent = data.memory || '--';
        document.getElementById('vp-uptime').textContent =
            data.active_line || '--';
    },

    async refreshVideoStats() {
        var resp = await fetch('/api/video/stats');
        var data = await resp.json();
        document.getElementById('vp-total-frames').textContent = data.frames || '0';
        document.getElementById('vp-pos-frames').textContent = data.positive || '0';
        document.getElementById('vp-neg-frames').textContent = data.negative || '0';
        document.getElementById('vp-processed').textContent = data.processed || '0';
        document.getElementById('vp-labeled').textContent =
            (data.train ? (parseInt(data.train) + parseInt(data.val || 0) - 2) : '--') + ' in dataset';
    },

    async refreshVideoLog() {
        var resp = await fetch('/api/video/log?n=80');
        var data = await resp.json();
        var logEl = document.getElementById('vp-log');
        logEl.textContent = (data.lines || []).join('\n');
        logEl.scrollTop = logEl.scrollHeight;
    },

    async startVideoService() {
        await fetch('/api/video/control/start', {method: 'POST'});
        showToast('Video pipeline started', 'success');
        setTimeout(function() { app.refreshVideoStatus(); }, 2000);
    },

    async stopVideoService() {
        await fetch('/api/video/control/stop', {method: 'POST'});
        showToast('Video pipeline stopped', 'info');
        setTimeout(function() { app.refreshVideoStatus(); }, 2000);
    },

    async fetchVideoOutput() {
        showToast('Fetching video training data from orrgate...', 'info');
        // Use the remote executor to SCP files
        var resp = await fetch('/api/video/stats');
        var data = await resp.json();
        showToast('Stats: ' + (data.frames || 0) + ' frames, ' +
            (data.frames_size || '?') + ' on disk', 'success');
    },
};

document.addEventListener('DOMContentLoaded', function() {
    app.initRules();
    app.initSelectionHandler();
});
