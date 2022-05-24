"""
@Description: 
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : call_test
@Time     : 2022/5/24 下午10:19
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/24 下午10:19        1.0             None
"""

class Person():
    def __call__(self, name):
        print("__call__" + " Hello " + name)

    def hello(self, name):
        print("hello " + name)

person = Person()
person("hcq") # 直接调用call
person.hello("hcq")