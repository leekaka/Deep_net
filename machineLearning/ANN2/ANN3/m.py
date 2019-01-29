class Student(object):
    def __init__(self,name,score):
        self.name = name
        self.score = score

    def get_score(self):
        if self.score >=90:
            return 'A'
        elif self.score >=60:
            return 'B'
        else:
            return 'C'


    def print_score(self):
        print("%s,%s"%(self.name,self.score))


