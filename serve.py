#!/usr/bin/env python3
import os, sys, subprocess

VENV_DIR = os.path.join(os.getcwd(), '.http_venv')

def in_venv():
    return (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    ) or hasattr(sys, 'real_prefix')

def create_venv():
    import venv
    print(f'➜ Creating virtualenv in {VENV_DIR}…')
    venv.create(VENV_DIR, with_pip=True)
    # upgrade pip inside the venv
    pip_path = os.path.join(VENV_DIR, 'bin', 'pip')
    print('➜ Upgrading pip in the venv…')
    subprocess.run([pip_path, 'install', '--upgrade', 'pip'], check=True)

def reexec_in_venv():
    python_path = os.path.join(VENV_DIR, 'bin', 'python')
    print(f'➜ Re-executing under {python_path} …\n')
    os.execv(python_path, [python_path] + sys.argv)

if __name__ == '__main__':
    # 1) bootstrap venv + pip
    if not in_venv():
        if not os.path.isdir(VENV_DIR):
            create_venv()
        reexec_in_venv()

    # 2) now inside .http_venv → start HTTP server
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    import shutil

    class QuietHandler(SimpleHTTPRequestHandler):
        """Silently swallow BrokenPipeError on client disconnect."""
        def copyfile(self, source, outputfile):
            try:
                shutil.copyfileobj(source, outputfile)
            except BrokenPipeError:
                pass

        def handle_one_request(self):
            try:
                return super().handle_one_request()
            except BrokenPipeError:
                return

    PORT = 8000
    print(f'➜ Serving HTTP on 0.0.0.0 port {PORT} (http://localhost:{PORT}/) …\n'
          f'  (use Ctrl-C to stop)')
    httpd = HTTPServer(('0.0.0.0', PORT), QuietHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n➜ Shutting down.')
        httpd.server_close()
