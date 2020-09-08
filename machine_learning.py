from sklearn import tree
from random import randint
import json

class DataSet():

    """Машинное обучение. Прогнозирование чего-нибудь. Данный вводятся с помощью JSON. 
    Пример структуры. 
    {
        "key": [
            [int_params1, int_params2, int_params3, int_params4, ...int_paramsn]]
    }"""
    def __init__(self, *args):
        self.x_data = {}
        self.y_data = []
        self.x_data_labes= []
        self.private = {
            "__add_base__":True
        }
        if len(args)!=0:
            self.__add_base__(args)      
    
    def __add_base__(self, args)->str:
        if self.private["__add_base__"]==False:
            return "Private method."

        self.x_data_labes = self.x_data_labes
        for arg in args:
            for a in arg: 
                if a not in self.x_data:
                    self.x_data[a]=len(self.x_data)
                
                for i in arg[a]:
                    self.x_data_labes.append(self.x_data[a])
                    self.y_data.append(i)

        self.classif = tree.DecisionTreeClassifier()
        self.classif.fit(self.y_data,self.x_data_labes)
        self.private["__add_base__"] = False
        return "ok"

    def add(self, *args):
        self.private["__add_base__"] = True
        self.__add_base__(args)
        
    def get_apply(self, *params)->list:
        """Возвращает индекс листа, в котором предсказывается каждый образец. Индекс начинается с 1"""
        response = list(self.classif.apply(params))
        return response

    def get_predict(self, *params)->list:
        """Предсказать класс или значение регрессии для X."""
        items = list(self.classif.predict(params))
        response = []
        for item in items:
            for key in self.x_data:
                if self.x_data[key]==item:
                    response.append(key)
        return response

    def get_score(self)->float:
        response = self.classif.score(self.y_data,self.x_data_labes)
        return response

    def __str__(self):
        response = {
            "x_data":self.x_data,
            "y_data":self.y_data,
            "x_data_labes":self.x_data_labes
        }
        return str(response)
    def get_decision_path(self, *params):
        items = list(self.classif.predict(params))
        response = self.classif.decision_path(self.y_data,self.x_data_labes)
        return response
        
    def save(self, filename="data.json"):
        "only JSON"
        response = []
        for i in range(len(self.y_data)):
            for j in self.x_data:
                if self.x_data[j]==self.x_data_labes[i]:
                    response.append({
                            "key":j,
                            "value":self.y_data[i]
                        }
                    )
                    break

        with open(filename, 'w', encoding='utf-8') as file:
            json.dump({"data":response}, file, indent=4, ensure_ascii=False)
        return response

    def open(self, filename="data.json"):
        with open(filename, "r") as file:
            items = json.load(file)["data"]

        data = {}
        for item in items:
            if item["key"] not in data:
                data[item["key"]] =[]
            data[item["key"]].append(item["value"])
        self.add(data)    

