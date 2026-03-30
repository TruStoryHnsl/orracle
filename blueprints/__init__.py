"""Blueprint registration for orracle."""

from flask import redirect


def register_blueprints(app):
    """Register all blueprints with the Flask app."""
    # Dashboard (command center)
    from blueprints.dashboard import dashboard_bp
    app.register_blueprint(dashboard_bp)

    # Studio (AI generation)
    from blueprints.text_gen_bp import text_gen_bp
    from blueprints.image_gen_bp import image_gen_bp
    from blueprints.forge_bp import forge_bp
    app.register_blueprint(text_gen_bp)
    app.register_blueprint(image_gen_bp)
    app.register_blueprint(forge_bp)

    # Workshop (data + training)
    from blueprints.pipeline import pipeline_bp
    from blueprints.training_bp import training_bp
    from blueprints.export_bp import export_bp
    from blueprints.audit_bp import audit_bp
    app.register_blueprint(pipeline_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(audit_bp)

    # Machines
    from blueprints.machines_bp import machines_bp
    app.register_blueprint(machines_bp)

    # Headless APIs (authenticated, for external clients)
    from blueprints.api_image_bp import api_image_bp
    app.register_blueprint(api_image_bp)

    # ── Redirects from old URLs ──
    @app.route('/compare')
    @app.route('/compare/')
    @app.route('/compare/<path:rest>')
    def redirect_compare(rest=''):
        return redirect('/studio/text', code=301)

    @app.route('/forge')
    @app.route('/forge/')
    def redirect_forge_old():
        return redirect('/studio/forge', code=301)

    # Workshop redirects (old URLs -> new)
    @app.route('/pipeline')
    @app.route('/pipeline/')
    @app.route('/pipeline/<path:rest>')
    def redirect_pipeline(rest=''):
        return redirect(f'/workshop/pipeline/{rest}', code=301)

    @app.route('/training')
    @app.route('/training/')
    @app.route('/training/<path:rest>')
    def redirect_training(rest=''):
        return redirect(f'/workshop/train/{rest}', code=301)

    @app.route('/train')
    @app.route('/train/')
    def redirect_train_short():
        return redirect('/workshop/train', code=301)

    @app.route('/export')
    @app.route('/export/')
    @app.route('/export/<path:rest>')
    def redirect_export(rest=''):
        return redirect(f'/workshop/export/{rest}', code=301)
