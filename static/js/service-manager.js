/* Orracle — Service manager UI logic
   Handles service start/stop, health polling, and status updates. */

async function fetchServices() {
    try {
        const r = await fetch('/api/services');
        return await r.json();
    } catch(e) {
        return [];
    }
}

async function refreshServiceGrid() {
    const services = await fetchServices();
    const container = document.getElementById('service-grid');
    if (!container) return;

    container.textContent = '';
    if (services.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'empty-state';
        empty.textContent = 'No services configured';
        container.appendChild(empty);
        return;
    }

    services.forEach(svc => {
        const card = document.createElement('div');
        card.className = 'card card-service';
        card.dataset.machine = svc.machine;
        card.dataset.service = svc.name;

        // Status dot + name
        const header = document.createElement('div');
        header.className = 'flex items-center justify-between';

        const left = document.createElement('div');
        left.className = 'flex items-center gap-sm';

        const dot = document.createElement('span');
        dot.className = 'status-dot ' + svc.status;
        left.appendChild(dot);

        const name = document.createElement('span');
        name.className = 'text-sm';
        name.textContent = svc.type;
        left.appendChild(name);

        const badge = document.createElement('span');
        badge.className = 'badge badge-' + svc.status;
        badge.textContent = svc.status;

        header.appendChild(left);
        header.appendChild(badge);
        card.appendChild(header);

        // Machine name
        const machine = document.createElement('span');
        machine.className = 'text-xs text-muted';
        machine.textContent = svc.machine;
        card.appendChild(machine);

        // Action hint
        if (svc.status === 'offline' && svc.start_cmd) {
            const hint = document.createElement('span');
            hint.className = 'text-xs text-accent';
            hint.textContent = 'tap to start';
            card.appendChild(hint);
            card.style.cursor = 'pointer';
            card.addEventListener('click', () => startService(svc.machine, svc.name));
        } else if (svc.status === 'online' && svc.stop_cmd) {
            card.style.cursor = 'pointer';
            card.addEventListener('click', () => stopService(svc.machine, svc.name));
        }

        // Meta info (model count, VRAM, etc.)
        if (svc.meta && svc.status === 'online') {
            if (svc.meta.model_count !== undefined) {
                const meta = document.createElement('span');
                meta.className = 'text-xs text-secondary';
                meta.textContent = svc.meta.model_count + ' models';
                card.appendChild(meta);
            }
        }

        container.appendChild(card);
    });
}
