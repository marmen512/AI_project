"""
Система навчання "Школа" - GPT як вчителька навчає TRM як дитину
"""
from .gpt_teacher import GPTTeacher
from .trm_student import TRMStudent
from .curriculum import SchoolCurriculum
from .kindergarten import KindergartenLearning

__all__ = [
    'GPTTeacher',
    'TRMStudent', 
    'SchoolCurriculum',
    'KindergartenLearning'
]

