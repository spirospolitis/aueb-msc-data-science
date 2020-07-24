"""
    AUEB M.Sc. in Data Science
    Semester: Sprint 2020
    Course: Deep Learning
    Lecturer: P. Malakasiotis
    Author: Spiros Politis
"""
import logging

"""
"""
def create_logger(name:str, level:int):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s\t[%(levelname)s]\t%(name)s\t%(module)s.%(funcName)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger
