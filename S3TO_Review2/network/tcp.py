import socket
import threading

from common.serialization import recv_message, send_message


class ThreadedTCPServer:
    def __init__(self, host, port, handler):
        self.host = host
        self.port = port
        self.handler = handler
        self._stop = threading.Event()
        self._sock = None
        self._thread = None

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(20)
        self._sock.settimeout(0.5)
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
            threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()

    def _handle(self, conn, addr):
        try:
            message = recv_message(conn)
            result = self.handler(message, addr)
            send_message(conn, result)
        except Exception as exc:
            try:
                send_message(conn, {"ok": False, "error": str(exc)})
            except Exception:
                pass
        finally:
            conn.close()

    def stop(self):
        self._stop.set()
        if self._sock:
            self._sock.close()
        if self._thread:
            self._thread.join(timeout=1)


def send_tcp_message(host, port, message, timeout=30):
    with socket.create_connection((host, port), timeout=timeout) as sock:
        send_message(sock, message)
        return recv_message(sock)
