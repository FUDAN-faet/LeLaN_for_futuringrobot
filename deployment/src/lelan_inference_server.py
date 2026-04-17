import argparse
import json
import socketserver
import struct
import threading

from lelan_runtime import LeLaNInferenceRuntime, decode_jpeg_image


def recv_exact(sock, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        data.extend(chunk)
    return bytes(data)


def recv_message(sock):
    header_size = struct.unpack("!I", recv_exact(sock, 4))[0]
    header = json.loads(recv_exact(sock, header_size).decode("utf-8"))
    payload_size = int(header.get("payload_size", 0))
    payload = recv_exact(sock, payload_size) if payload_size else b""
    return header, payload


def send_message(sock, header, payload: bytes = b"") -> None:
    header = dict(header)
    header["payload_size"] = len(payload)
    header_bytes = json.dumps(header).encode("utf-8")
    sock.sendall(struct.pack("!I", len(header_bytes)))
    sock.sendall(header_bytes)
    if payload:
        sock.sendall(payload)


class LeLaNServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, runtime: LeLaNInferenceRuntime, handler_class):
        super().__init__(server_address, handler_class)
        self.runtime = runtime
        self.runtime_lock = threading.Lock()


class LeLaNRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        try:
            header, payload = recv_message(self.request)
            if header.get("encoding") != "jpeg":
                raise ValueError(f"Unsupported image encoding: {header.get('encoding')}")

            prompt = header.get("prompt", "chair")
            cv_image = decode_jpeg_image(payload)
            with self.server.runtime_lock:
                linear_vel, angular_vel = self.server.runtime.predict_from_cv2(cv_image, prompt=prompt)

            send_message(
                self.request,
                {
                    "ok": True,
                    "linear_vel": linear_vel,
                    "angular_vel": angular_vel,
                },
            )
        except Exception as exc:  # pragma: no cover - runtime error reporting
            send_message(self.request, {"ok": False, "error": str(exc)})


def parse_args():
    parser = argparse.ArgumentParser(description="Local LeLaN inference server for ROS 2 bridge mode.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--prompt", default="chair")
    parser.add_argument("--config", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = LeLaNInferenceRuntime(
        config_path=args.config or None,
        model_path=args.model or None,
        prompt=args.prompt,
        device=args.device or None,
    )

    with LeLaNServer((args.host, args.port), runtime, LeLaNRequestHandler) as server:
        print(f"LeLaN inference server listening on {args.host}:{args.port}")
        print(f"device={runtime.device}")
        print(f"model_path={runtime.model_path}")
        print(f"config_path={runtime.config_path}")
        print(f"prompt={runtime.prompt}")
        server.serve_forever()


if __name__ == "__main__":
    main()
