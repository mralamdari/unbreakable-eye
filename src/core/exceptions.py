# # class ModelResolutionError(ValueError):
# #     """Custom exception for errors during model path resolution or download."""
# #     pass

# # class ModelConfigurationError(ValueError):
# #     """Custom exception for invalid model configuration logic."""
# #     pass








# ###############################  TEMPORARY SOLUTIONS, FILL THE ERRORS ABOVE  ??????????????????????????????????????????
# import traceback
# import sys

# class ModelResolutionError(ValueError):
# #      """Custom exception for errors during model path resolution or download."""
#     def __init__(self, error_message, error_detail:sys):
#         super().__init__(error_message)
#         self.error_message = self.get_detailed_error_message(error_message,error_detail)

#     @staticmethod
#     def get_detailed_error_message(error_message , error_detail:sys):

#         _, _, exc_tb = traceback.sys.exc_info()
#         file_name = exc_tb.tb_frame.f_code.co_filename
#         line_number = exc_tb.tb_lineno

#         return f"Model Path Resolution/Download Error in {file_name} , line {line_number} : {error_message}"
    
#     def __str__(self):
#         return self.error_message




# class ModelConfigurationError(ValueError):
# #     """Custom exception for invalid model configuration logic."""
#     def __init__(self, error_message, error_detail:sys):
#         super().__init__(error_message)
#         self.error_message = self.get_detailed_error_message(error_message,error_detail)

#     @staticmethod
#     def get_detailed_error_message(error_message , error_detail:sys):

#         _, _, exc_tb = traceback.sys.exc_info()
#         file_name = exc_tb.tb_frame.f_code.co_filename
#         line_number = exc_tb.tb_lineno

#         return f"Invalid Model Configuration Error in {file_name} , line {line_number} : {error_message}"
    
#     def __str__(self):
#         return self.error_message




import traceback
from typing import Optional

class ModelResolutionError(ValueError):
    """
    Custom exception for errors during model path resolution or download.
    
    Parameters:
    error_message (str): A brief description of the error.
    error_detail (Optional[Exception]): The detailed exception object, if available.
    """

    def __init__(self, error_message: str, error_detail: Optional[Exception] = None):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message: str, error_detail: Optional[Exception]) -> str:
        if error_detail is not None:
            exc_tb = traceback.extract_tb(error_detail.__traceback__)
            file_name = exc_tb[-1].filename
            line_number = exc_tb[-1].lineno
            return f"Model Path Resolution/Download Error in {file_name}, line {line_number}: {error_message}"
        else:
            return f"Model Path Resolution/Download Error: {error_message}"

    def __str__(self):
        return self.error_message


import traceback
from typing import Optional

class ModelConfigurationError(ValueError):
    """
    Custom exception for invalid model configuration logic.
    
    Parameters:
    error_message (str): A brief description of the error.
    error_detail (Optional[Exception]): The detailed exception object, if available.
    """

    def __init__(self, error_message: str, error_detail: Optional[Exception] = None):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message: str, error_detail: Optional[Exception]) -> str:
        if error_detail is not None:
            exc_tb = traceback.extract_tb(error_detail.__traceback__)
            file_name = exc_tb[-1].filename
            line_number = exc_tb[-1].lineno
            return f"Invalid Model Configuration Error in {file_name}, line {line_number}: {error_message}"
        else:
            return f"Invalid Model Configuration Error: {error_message}"

    def __str__(self):
        return self.error_message
