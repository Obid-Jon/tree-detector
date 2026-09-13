#import numpy as np

class NewClaassP:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def get_info(self):
        return f'{self.age} y.o {self.name} mr'
    def r_nalog(self, sum_nal):
        return int(sum_nal * 0.87)
        
    somthing = 0
    def print_unit(self):
        return 'yesli chto nado me nt nado skazat'
m_calss = NewClaassP('Mark', 23)
a = int(input())
print(m_calss.r_nalog(a))
print(m_calss.print_unit())