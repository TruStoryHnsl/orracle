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

    # Inject a static_version derived from favicon.ico mtime so the
    # browser fetches a fresh favicon whenever the file is regenerated.
    # Chrome's favicon cache otherwise holds stale entries for hours
    # behind openresty's max-age=4259 cache header.
    _favicon = os.path.join(app.static_folder, 'favicon.ico')
    try:
        _static_version = str(int(os.path.getmtime(_favicon)))
    except OSError:
        _static_version = '1'

    @app.context_processor
    def inject_static_version():
        return {'static_version': _static_version}

    return app


app = create_app()
