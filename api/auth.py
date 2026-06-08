from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
import ipaddress
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
logger = logging.getLogger(__name__)
class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, config):
        super().__init__(app)
        self.config = config
        self.auth_enabled = config.get('enabled', False)
        self.api_keys = config.get('api_keys', [])
        self.ip_whitelist = config.get('ip_whitelist', [])
        self.allow_localhost = config.get('allow_localhost', True)
        self.allow_lan = config.get('allow_lan', True)
    def is_ip_allowed(self, client_ip):
        if not self.auth_enabled:
            return True
        if self.allow_localhost and client_ip in ['127.0.0.1', 'localhost', '::1']:
            return True
        try:
            ip_obj = ipaddress.ip_address(client_ip)
            if self.allow_lan:
                if ip_obj.is_private:
                    return True
            for allowed_ip in self.ip_whitelist:
                try:
                    if '/' in allowed_ip:
                        network = ipaddress.ip_network(allowed_ip, strict=False)
                        if ip_obj in network:
                            return True
                    else:
                        if ipaddress.ip_address(allowed_ip) == ip_obj:
                            return True
                except ValueError:
                    continue
        except ValueError:
            pass
        return False
    def is_api_key_valid(self, api_key):
        if not self.auth_enabled:
            return True
        if not self.api_keys:
            return True
        return api_key in self.api_keys
    async def dispatch(self, request: Request, call_next):
        if not self.auth_enabled:
            return await call_next(request)
        client_ip = self.get_client_ip(request)
        ip_allowed = self.is_ip_allowed(client_ip)
        if not ip_allowed:
            api_key = request.headers.get('X-API-Key')
            if not api_key:
                logger.warning(f" IPAPI: {client_ip}")
                raise HTTPException(status_code=403, detail=f"IP {client_ip} API")
            if not self.is_api_key_valid(api_key):
                logger.warning(f" IPAPI: {client_ip}")
                raise HTTPException(status_code=401, detail="API")
            logger.info(f" IPAPI: {client_ip}")
        elif self.api_keys:
            api_key = request.headers.get('X-API-Key')
            if api_key and not self.is_api_key_valid(api_key):
                logger.warning(f" IPAPI: {client_ip}")
                raise HTTPException(status_code=401, detail="API")
        return await call_next(request)
    def get_client_ip(self, request: Request):
        x_forwarded_for = request.headers.get('X-Forwarded-For')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        x_real_ip = request.headers.get('X-Real-IP')
        if x_real_ip:
            return x_real_ip
        host = request.client.host if request.client else 'unknown'
        return host
def get_auth_config(config):
    return config.get('api', {}).get('auth', {})
