/* Orracle — Dashboard controller
   Manages real-time updates via SSE, service actions, job actions. */

let evtSource = null;

// ─── SSE Connection ───
function connectDashboardSSE() {
    if (evtSource) evtSource.close();
    evtSource = new EventSource('/api/dashboard/stream');

    evtSource.onmessage = function(e) {
        try {
            const data = JSON.parse(e.data);
            handleDashboardEvent(data);
        } catch(err) { /* ignore parse errors */ }
    };

    evtSource.onerror = function() {
        evtSource.close();
        // Reconnect after 5 seconds
        setTimeout(connectDashboardSSE, 5000);
    };
}

function handleDashboardEvent(data) {
    switch(data.type) {
        case 'service_change':
            updateServiceCard(data);
            loadMachineDots();
            break;
        case 'job_submitted':
        case 'job_started':
        case 'job_completed':
        case 'job_failed':
        case 'job_cancelled':
        case 'job_progress':
        case 'job_suspended':
        case 'job_resumed':
        case 'job_throttle':
            updateJobCard(data.job);
            updateQueueCounts();
            break;
        case 'compute_load':
            updateMachineTelemetry(data.load);
            break;
        case 'heartbeat':
            break;
    }
}

// ─── Service Actions ───
async function startService(machine, service) {
    const btn = event.currentTarget;
    btn.style.pointerEvents = 'none';
    showToast('Starting ' + service + ' on ' + machine + '...', 'info');

    try {
        const r = await fetch('/api/dashboard/action', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({action: 'start_service', machine, service})
        });
        const data = await r.json();
        if (data.ok) {
            showToast(service + ' starting...', 'success');
        } else {
            showToast('Failed: ' + (data.error || 'unknown'), 'error');
        }
    } catch(e) {
        showToast('Connection error', 'error');
    }
    btn.style.pointerEvents = '';
}

async function stopService(machine, service) {
    if (!confirm('Stop ' + service + ' on ' + machine + '?')) return;

    try {
        const r = await fetch('/api/dashboard/action', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({action: 'stop_service', machine, service})
        });
        const data = await r.json();
        if (data.ok) {
            showToast(service + ' stopped', 'success');
        } else {
            showToast('Failed: ' + (data.error || 'unknown'), 'error');
        }
    } catch(e) {
        showToast('Connection error', 'error');
    }
}

// ─── Job Actions ───
async function cancelJob(jobId) {
    try {
        const r = await fetch('/api/queue/' + jobId + '/cancel', {method: 'POST'});
        const data = await r.json();
        if (data.ok) showToast('Job cancelled', 'success');
    } catch(e) {
        showToast('Error cancelling job', 'error');
    }
}

async function retryJob(jobId) {
    try {
        const r = await fetch('/api/queue/' + jobId + '/retry', {method: 'POST'});
        const data = await r.json();
        if (data.new_id) showToast('Retrying as ' + data.new_id, 'success');
    } catch(e) {
        showToast('Error retrying job', 'error');
    }
}

async function suspendJob(jobId) {
    try {
        const r = await fetch('/api/queue/' + jobId + '/suspend', {method: 'POST'});
        const data = await r.json();
        if (data.ok) showToast('Job suspended', 'success');
        else showToast('Could not suspend', 'error');
    } catch(e) {
        showToast('Error suspending job', 'error');
    }
}

async function resumeJob(jobId) {
    try {
        const r = await fetch('/api/queue/' + jobId + '/resume', {method: 'POST'});
        const data = await r.json();
        if (data.ok) showToast('Job resumed', 'success');
        else showToast('Could not resume', 'error');
    } catch(e) {
        showToast('Error resuming job', 'error');
    }
}

async function toggleThrottle(jobId, enabled) {
    try {
        const r = await fetch('/api/queue/' + jobId + '/throttle', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({throttle: enabled})
        });
        const data = await r.json();
        if (data.ok) showToast('Throttle ' + (enabled ? 'enabled' : 'disabled'), 'success');
        else showToast('Could not toggle throttle', 'error');
    } catch(e) {
        showToast('Error toggling throttle', 'error');
    }
}

async function startWaitingJob(jobId) {
    try {
        const r = await fetch('/api/queue/' + jobId + '/start', {method: 'POST'});
        const data = await r.json();
        if (data.ok) {
            showToast('Job started', 'success');
            location.reload();
        } else {
            showToast('Could not start job', 'error');
        }
    } catch(e) {
        showToast('Error starting job', 'error');
    }
}

// ─── DOM Updates ───
function updateServiceCard(data) {
    const cards = document.querySelectorAll('.card-service[data-machine="' + data.machine + '"][data-service="' + data.name + '"]');
    cards.forEach(card => {
        const dot = card.querySelector('.status-dot');
        const badge = card.querySelector('.badge');
        if (dot) dot.className = 'status-dot ' + data.status;
        if (badge) {
            badge.className = 'badge badge-' + data.status;
            badge.textContent = data.status;
        }
    });
}

function updateJobCard(job) {
    const card = document.querySelector('.card-job[data-job-id="' + job.id + '"]');
    if (!card) {
        // New job — reload the job list section
        refreshJobList();
        return;
    }
    const badge = card.querySelector('.badge');
    if (badge) {
        badge.className = 'badge badge-' + job.status;
        badge.textContent = job.status;
    }
    const fill = card.querySelector('.progress-bar .fill');
    if (fill) fill.style.width = Math.round(job.progress * 100) + '%';
}

async function updateQueueCounts() {
    try {
        const r = await fetch('/api/queue/counts');
        const data = await r.json();
        const el = document.getElementById('queue-counts');
        if (el) {
            el.textContent = [
                data.pending ? data.pending + ' pending' : '',
                data.running ? data.running + ' running' : '',
                data.failed ? data.failed + ' failed' : '',
            ].filter(Boolean).join(' / ') || 'idle';
        }
    } catch(e) { /* ignore */ }
}

async function refreshJobList() {
    try {
        const r = await fetch('/api/queue/list');
        const jobs = await r.json();
        const container = document.getElementById('active-jobs');
        if (!container) return;

        container.textContent = '';
        const active = jobs.filter(j => ['running','pending','routing'].includes(j.status));
        if (active.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'empty-state';
            empty.textContent = 'No active jobs';
            container.appendChild(empty);
            return;
        }
        active.forEach(job => {
            const card = createJobCardElement(job);
            container.appendChild(card);
        });
    } catch(e) { /* ignore */ }
}

function createJobCardElement(job) {
    const card = document.createElement('div');
    card.className = 'card card-job ' + job.status;
    card.dataset.jobId = job.id;

    const header = document.createElement('div');
    header.className = 'card-header';

    const title = document.createElement('span');
    title.className = 'card-title';
    title.textContent = (job.params && job.params.name) || job.category;

    const badge = document.createElement('span');
    badge.className = 'badge badge-' + job.status;
    badge.textContent = job.status;

    header.appendChild(title);
    header.appendChild(badge);
    card.appendChild(header);

    if (job.status === 'running' && job.progress > 0) {
        const bar = document.createElement('div');
        bar.className = 'progress-bar';
        const fill = document.createElement('div');
        fill.className = 'fill';
        fill.style.width = Math.round(job.progress * 100) + '%';
        bar.appendChild(fill);
        card.appendChild(bar);
    }

    return card;
}

// ─── Machine Telemetry ───
function updateMachineTelemetry(load) {
    if (!load || !load.machine) return;
    var el = document.querySelector('[data-machine-telemetry="' + load.machine + '"]');
    if (!el) return;

    var gpuBar = el.querySelector('[data-gpu-bar]');
    var gpuVal = el.querySelector('[data-gpu-util]');
    var vramBar = el.querySelector('[data-vram-bar]');
    var vramVal = el.querySelector('[data-vram-util]');
    var cpuBar = el.querySelector('[data-cpu-bar]');
    var cpuVal = el.querySelector('[data-cpu-util]');

    if (gpuBar) gpuBar.style.width = Math.round(load.gpu_util || 0) + '%';
    if (gpuVal) gpuVal.textContent = Math.round(load.gpu_util || 0) + '%';

    var vramPct = load.gpu_mem_total ? Math.round(load.gpu_mem_used / load.gpu_mem_total * 100) : 0;
    if (vramBar) vramBar.style.width = vramPct + '%';
    if (vramVal) vramVal.textContent = Math.round(load.gpu_mem_used || 0) + 'MB';

    var cpuPct = load.cpu_count ? Math.min(100, Math.round(load.cpu_load_1m / load.cpu_count * 100)) : 0;
    if (cpuBar) cpuBar.style.width = cpuPct + '%';
    if (cpuVal) cpuVal.textContent = (load.cpu_load_1m || 0).toFixed(1);
}

// ─── Init ───
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('dashboard-root')) {
        connectDashboardSSE();
        updateQueueCounts();
    }
});
