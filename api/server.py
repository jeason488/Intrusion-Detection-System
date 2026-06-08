from fastapi import FastAPI
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
import os
from api.routes import router
from api.auth import AuthMiddleware, get_auth_config
logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
logger = logging.getLogger(__name__)
def create_app(config=None):
    app = FastAPI(
        title="API",
        description="",
        version="1.0.0"
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if config:
        auth_config = get_auth_config(config)
        app.add_middleware(AuthMiddleware, config=auth_config)
        if auth_config.get('enabled', False):
            logger.info("Authentication enabled")
    app.include_router(router)
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
    if os.path.exists(frontend_dir):
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
        logger.info(f"Frontend served from: {frontend_dir}")
    else:
        logger.warning(f"Frontend directory not found: {frontend_dir}")
    return app
def start_server(config):
    app = create_app(config)
    host = config['api'].get('host', 'localhost')
    port = config['api'].get('port', 8000)
    debug = config['api'].get('debug', False)
    logger.info(f"API running at: http://{host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=debug
    )
app = create_app()
