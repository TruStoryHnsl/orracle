/* Orracle — Text generation chat interface */

let conversation = [];
let controller = null;
let streaming = false;

const messagesEl = document.getElementById('chat-messages');
const inputEl = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const modelSelect = document.getElementById('model-select');
const statsEl = document.getElementById('chat-stats');

// ─── Message rendering ───
function addMessage(role, content) {
    const div = document.createElement('div');
    div.className = 'msg msg-' + role;

    const label = document.createElement('div');
    label.className = 'msg-label';
    label.textContent = role;
    div.appendChild(label);

    const text = document.createElement('div');
    text.textContent = content;
    div.appendChild(text);

    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return text;
}

// ─── Get params ───
function getOptions() {
    return {
        temperature: parseFloat(document.getElementById('param-temp').value),
        top_p: parseFloat(document.getElementById('param-topp').value),
        num_ctx: parseInt(document.getElementById('param-ctx').value),
    };
}

function getSystemPrompt() {
    const el = document.getElementById('system-prompt');
    return el ? el.value.trim() : '';
}

function getModelInfo() {
    const sel = modelSelect;
    const opt = sel.options[sel.selectedIndex];
    return {
        name: opt ? opt.dataset.name : '',
        host: opt ? opt.dataset.host || null : null,
    };
}

// ─── Send message ───
function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || streaming) return;

    const model = getModelInfo();
    if (!model.name) {
        showToast('Select a model first', 'error');
        return;
    }

    inputEl.value = '';
    inputEl.style.height = 'auto';

    addMessage('user', text);
    conversation.push({ role: 'user', content: text });

    const messages = [];
    const sys = getSystemPrompt();
    if (sys) messages.push({ role: 'system', content: sys });
    messages.push(...conversation);

    const assistantText = addMessage('assistant', '');
    const dot = document.createElement('span');
    dot.className = 'streaming-dot';
    assistantText.appendChild(dot);

    streaming = true;
    sendBtn.disabled = true;
    stopBtn.disabled = false;
    let fullResponse = '';
    const startTime = performance.now();

    controller = new AbortController();

    fetch('/studio/text/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: model.name,
            messages: messages,
            options: getOptions(),
            host: model.host,
        }),
        signal: controller.signal,
    }).then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        function read() {
            reader.read().then(({ done, value }) => {
                if (done) { finish(); return; }
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.type === 'token') {
                            fullResponse += data.content;
                            assistantText.textContent = fullResponse;
                            const d = document.createElement('span');
                            d.className = 'streaming-dot';
                            assistantText.appendChild(d);
                            messagesEl.scrollTop = messagesEl.scrollHeight;
                        } else if (data.type === 'stats') {
                            showStats(data, startTime);
                        } else if (data.type === 'done') {
                            finish();
                            return;
                        } else if (data.type === 'error') {
                            assistantText.textContent = 'Error: ' + data.message;
                            assistantText.style.color = 'var(--error)';
                            finish();
                            return;
                        }
                    } catch (e) { /* ignore parse errors */ }
                }
                read();
            }).catch(e => {
                if (e.name !== 'AbortError') {
                    assistantText.textContent = fullResponse || 'Connection lost';
                }
                finish();
            });
        }
        read();
    }).catch(e => {
        if (e.name !== 'AbortError') {
            assistantText.textContent = 'Error: ' + e.message;
            assistantText.style.color = 'var(--error)';
        }
        finish();
    });

    function finish() {
        streaming = false;
        controller = null;
        sendBtn.disabled = false;
        stopBtn.disabled = true;
        const ind = assistantText.querySelector('.streaming-dot');
        if (ind) ind.remove();
        if (fullResponse) {
            assistantText.textContent = fullResponse;
            conversation.push({ role: 'assistant', content: fullResponse });
        }
    }
}

function showStats(stats, startTime) {
    if (!statsEl) return;
    statsEl.textContent = '';
    if (stats.eval_count && stats.eval_duration) {
        const tps = stats.eval_count / (stats.eval_duration / 1e9);
        const s = document.createElement('span');
        s.className = 'stat-highlight';
        s.textContent = tps.toFixed(1) + ' tok/s';
        statsEl.appendChild(s);
    }
    if (stats.eval_count) {
        const s = document.createElement('span');
        s.textContent = stats.eval_count + ' tokens';
        statsEl.appendChild(s);
    }
    const elapsed = document.createElement('span');
    elapsed.textContent = ((performance.now() - startTime) / 1000).toFixed(1) + 's';
    statsEl.appendChild(elapsed);
}

// ─── Controls ───
function stopGeneration() {
    if (controller) controller.abort();
}

function clearChat() {
    conversation = [];
    messagesEl.textContent = '';
    statsEl.textContent = '';
    const info = document.createElement('div');
    info.className = 'msg msg-assistant';
    info.style.color = 'var(--text-muted)';
    info.style.background = 'none';
    info.style.border = 'none';
    info.textContent = 'Start a conversation.';
    messagesEl.appendChild(info);
}

// ─── Input handling ───
function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

inputEl.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

// ─── Param sliders ───
document.getElementById('param-temp').addEventListener('input', function () {
    document.getElementById('param-temp-v').textContent = this.value;
});
document.getElementById('param-topp').addEventListener('input', function () {
    document.getElementById('param-topp-v').textContent = this.value;
});

inputEl.focus();
