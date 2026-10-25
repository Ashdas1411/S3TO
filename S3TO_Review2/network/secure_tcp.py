"""Bounded JSON/Tensor TCP transport used by the Mininet Path A experiment."""
import socket
import threading

from network.secure_serialization import recv_message, send_message

MAX_ACTIVE_CONNECTIONS = 8
CONNECTION_TIMEOUT_SECONDS = 60


class ThreadedTCPServer:
    def __init__(self, host, port, handler):
        self.host = host
        self.port = port
        self.handler = handler
        self._stop = threading.Event()
        self._sock = None
        self._thread = None
        self._slots = threading.BoundedSemaphore(MAX_ACTIVE_CONNECTIONS)

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(MAX_ACTIVE_CONNECTIONS)
        sock.settimeout(0.5)
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            if not self._slots.acquire(blocking=False):
                conn.close()
                continue
            threading.Thread(
                target=self._handle, args=(conn, addr), daemon=True
            ).start()

    def _handle(self, conn, addr):
        try:
            conn.settimeout(CONNECTION_TIMEOUT_SECONDS)
            message = recv_message(conn)
            result = self.handler(message, addr)
            send_message(conn, result)
        except Exception as exc:
            try:
                send_message(conn, {"ok": False, "error": str(exc)[:500]})
            except Exception:
                pass
        finally:
            conn.close()
            self._slots.release()

    def stop(self):
        self._stop.set()
        if self._sock:
            self._sock.close()
        if self._thread:
            self._thread.join(timeout=1)


def send_tcp_message(host, port, message, timeout=30):
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        send_message(sock, message)
        return recv_message(sock)
