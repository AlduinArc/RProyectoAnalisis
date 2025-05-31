# En archivos/middleware.py
from django.contrib.auth import logout

class ForceLogoutMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not request.session.get('force_logout_done'):
            logout(request)
            request.session['force_logout_done'] = True
        return self.get_response(request)
    
class SessionCleanupMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Verifica si la sesión debe cerrarse
        if request.session.get_expire_at_browser_close():
            request.session.set_expiry(0)  # Expira al cerrar navegador
        
        response = self.get_response(request)
        
        # Limpieza adicional si es necesario
        if not request.user.is_authenticated and 'sessionid' in request.COOKIES:
            response.delete_cookie('sessionid')
            
        return response