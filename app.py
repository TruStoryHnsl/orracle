"""Orracle — unified AI generation studio and model training workshop."""

import os

from flask import Flask

# Force node registration on import
import nodes  # noqa: F401
from shared import CONFIG_DIR
from services import ServiceManager, ComputeWatcher
from job_queue import JobQueue


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET', 'orracle-dev-key')

    # Initialize service manager (tracks ComfyUI, Ollama, etc. across machines)
    svc_mgr = ServiceManager()
    svc_mgr.discover()
    svc_mgr.start_monitor(interval=30)
    app.config['service_manager'] = svc_mgr

    # Initialize job queue (unified work dispatch)
    queue = JobQueue(svc_mgr, CONFIG_DIR)
    queue.start_processor()
    app.config['job_queue'] = queue

    # Initialize compute load watcher (auto-throttle for low-priority jobs)
    watcher = ComputeWatcher()
    watcher.wire(queue)
    watcher.start(interval=15)
    app.config['compute_watcher'] = watcher

    from blueprints import register_blueprints
    register_blueprints(app)

    return app


app = create_app()
