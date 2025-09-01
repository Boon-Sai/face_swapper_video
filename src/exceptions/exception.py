import sys

class VideoProcessingException(Exception):
    def __init__(self, error_message: str, error_details: sys):
        super().__init__(error_message)
        _, _, exc_tb = error_details.exc_info()
        self.line_number = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.error_message = error_message

    def __str__(self):
        return f"Error occurred in Python Script: [{self.file_name}] at line number [{self.line_number}]. Error message: [{self.error_message}]"
