import operator
from itertools import chain
from collections import defaultdict


def concatenate_dictionary(dict1,dict2):
    dict3 = defaultdict(list)
    for key, value in chain(dict1.items(), dict2.items()):
        dict3[key].append(value)
    sorted_dict = sorted(dict3.items(), key=operator.itemgetter(1))
    return sorted_dict


employee_details_1 = {'Lalith': ('Age',25), 'Ram': ('Salary',2500), 'Shyam': ('ID',2500)}
employee_details_2 = {'Chandu': ('Age',26), 'Bheem': ('ID',1010), 'Lalith': ('Salary',3500), 'Shyam': ('Age',21)}

Resultdict = concatenate_dictionary(employee_details_1,employee_details_2)
print(Resultdict)