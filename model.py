from model_base import SimpleModel
from model_MNIST import MNISTClassifier
from model_student import Student
from model_simple_student import SimpleStudent


def Model(type, config=None):
    if type == 'simple':
        return SimpleModel()
    if type == 'MNIST':
        return MNISTClassifier()
    if type == 'student':
        return Student()
    if type == 'simple_student':
        return SimpleStudent()
