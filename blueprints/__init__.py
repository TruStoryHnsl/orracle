"""Blueprint registration for orracle."""


def register_blueprints(app):
    """Register all blueprints with the Flask app."""
    from blueprints.dashboard import dashboard_bp
    from blueprints.pipeline import pipeline_bp
    from blueprints.training_bp import training_bp
    from blueprints.export_bp import export_bp
    from blueprints.compare_bp import compare_bp
    from blueprints.forge_bp import forge_bp
    from blueprints.machines_bp import machines_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(pipeline_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(compare_bp)
    app.register_blueprint(forge_bp)
    app.register_blueprint(machines_bp)
