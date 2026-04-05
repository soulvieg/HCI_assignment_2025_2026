"""
Generic utility functions
"""
import ast


def str_to_list(s):
    """
    Function to transform a string to a list
    """
    if isinstance(s, str) and s.startswith("["):
        try:
            return ast.literal_eval(s)
        except(ValueError, SyntaxError):
            return []
    return s if isinstance(s, list) else []