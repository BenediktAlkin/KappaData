class MockLogger:
    def __init__(self):
        self.msgs = []

    def __call__(self, msg):
        self.msgs.append(msg)

    @staticmethod
    def path_msg_equals(expected, actual):
        return expected.replace("\\", "/") == actual.replace("\\", "/")
