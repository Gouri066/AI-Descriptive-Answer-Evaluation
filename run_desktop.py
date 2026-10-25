import os
import shutil
import socket
import subprocess
import threading
import time

from app import app

HOST = "127.0.0.1"
PORT = 5000
URL = f"http://{HOST}:{PORT}"
TITLE = "AI Answer Evaluation System"


def run_server():
    """Start the Flask server (no auto-reload, so it can run in a thread)."""
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)


def wait_for_server():
    """Wait until the server accepts connections (max about 10 seconds)."""
    for _ in range(100):
        try:
            with socket.create_connection((HOST, PORT), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def open_with_webview():
    """Open a native window using pywebview. Returns False if unavailable."""
    try:
        import webview
        webview.create_window(TITLE, URL, width=1200, height=800,
                              min_size=(900, 600))
        webview.start()  # blocks until the window is closed
        return True
    except Exception as error:
        print("pywebview not available:", error)
        return False


def find_edge():
    paths = (
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    )
    for path in paths:
        if os.path.exists(path):
            return path
    return shutil.which("msedge")


if __name__ == "__main__":
    threading.Thread(target=run_server, daemon=True).start()

    if not wait_for_server():
        print("The server did not start. Is another copy already running?")
        raise SystemExit(1)

    if not open_with_webview():
        edge = find_edge()
        if edge is None:
            print("Neither pywebview nor Microsoft Edge was found.")
            print("Open this address in a browser instead:", URL)
        else:
            subprocess.Popen([edge, f"--app={URL}"])
            print("App window opened. Press Ctrl+C here to stop the app.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass