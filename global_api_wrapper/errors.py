class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, param: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.param = param

    def body(self) -> dict:
        kind = "server_error" if self.status >= 500 else "invalid_request_error"
        if self.status == 401:
            kind = "authentication_error"
        elif self.status == 429:
            kind = "rate_limit_error"
        return {"error": {"message": self.message, "type": kind, "param": self.param, "code": self.code}}
