class IdCounter:
    __counter = 0

    @staticmethod
    def next():
        result = IdCounter.__counter
        IdCounter.__counter += 1
        return result
