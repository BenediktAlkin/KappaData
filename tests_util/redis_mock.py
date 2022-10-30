from redis.connection import Encoder


class RedisMock:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.is_closed = False
        self.items = {}
        self.encoder = Encoder(
            encoding="utf-8",
            encoding_errors="strict",
            decode_responses=False,
        )

    def exists(self, key):
        return key in self.items

    def set(self, key, value):
        encoded = self.encoder.encode(value)
        self.items[key] = encoded

    def get(self, key):
        return self.items[key]

    def close(self):
        self.is_closed = True

    def get_encoder(self):
        return self.encoder

    def ping(self):
        pass

    def flushall(self):
        self.items.clear()
