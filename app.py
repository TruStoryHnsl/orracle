"""Orracle — unified AI training data pipeline and model training management."""

import os

from flask import Flask

# Force node registration on import
import nodes  # noqa: F401


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET', 'orracle-dev-key')

    from blueprints import register_blueprints
    register_blueprints(app)

    return app


app = create_app()
