from waitress import serve
import os
from app import app  # Ensure this imports your WSGI application correctly

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5001))
    serve(app, host='0.0.0.0', port=port)
