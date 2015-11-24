#!/usr/python
import math
from itertools import count
from __builtin__ import str
import sys

attr_list = []
first_value = ""
second_value = ""
classification_value1_set = []
classification_value2_set = []
prob_dist_TAN = dict()
    
class AttributeClass:
    name = ""
    index = 0
    values = []    
    pdc_value_list = []  
    pdc_count_list = []
    parent_index = 0
    def set_members(self, attr_mem, index):    
        self.name = attr_mem[0].strip("'")
        self.index = index
        if (len(attr_mem) > 2 ): #to extract the entire list of attribute values
            self.values += attr_mem[2:] 
            if (len(self.values) > 0 ):
                count = 0
                for i in self.values:
                    self.values[count]=self.values[count][:-1]
                    count += 1
    def set_pdc_count(self, temp_list):
       self.pdc_count_list = temp_list
    def set_pdc_value(self, temp_list):
       self.pdc_value_list = temp_list
    def __str__(self):
        return str(self.name)
    def display_attrs(self):
        print " Name:" + str(self.name) + " Values:" + str(self.values) + " Index:" + str(self.index)

'''
Function to read input file and parse data
'''
def read_data(data_to_parse, flag):
    if (flag == "train"):
        global first_value
        global second_value
        file_txt = open(data_to_parse)
        data = []
        dataset = [[]]
        lines = [line.rstrip('\n') for line in file_txt]
        count = 0 #to keep track of index of each attribute in attribute list
        for item in lines:       
            if item.startswith("@relation"):
                relation_name = item[10:]
                #print relation_name
            if item.startswith("@attribute"):
                attr_mem = item[11:].split(" ") 
                attribute = AttributeClass()
                attribute.values = []
                attribute.set_members(attr_mem,count)
                attr_list.append(attribute)
                count+=1
            elif item.startswith("@data"):
                index_of_data = lines.index("@data")
                data = lines[(index_of_data+1):]
                for i in range(len(data)):
                    dataset.append(data[i].split(","))
                del dataset[0]
        first_value = attr_list[-1].values[0]
        for i in dataset:
            if (i[len(dataset[0])-1]!=first_value):
                second_value = i[len(dataset[0])-1]
                break
        return dataset
    elif(flag == "test"):
        file_txt = open(data_to_parse)
        data = []
        dataset = [[]]
        lines = [line.rstrip('\n') for line in file_txt]
        for item in lines:       
            if item.startswith("@data"):
                index_of_data = lines.index("@data")
                data = lines[(index_of_data+1):]
                for i in range(len(data)):
                    dataset.append(data[i].split(","))
                del dataset[0]
        return dataset

def find_laplace(x, y, no_values_x):
    ratio = ((x + 1)*1.0)/(y + no_values_x)
    return ratio

def bayes_predict(dataset, prob_y, prob_y_dash):
    count_accurate = 0
    count_inaccurate = 0
    for attr in attr_list[0:-1]:
        print str(attr.name) + " " + "class"
    print " "
    for row in dataset:
        product_y = 1.0
        product_y_dash = 1.0
        for col_no in range(len(row)-1):
            x_i_index = attr_list[col_no].values.index(row[col_no])
            x_i_y_value = attr_list[col_no].pdc_value_list[x_i_index][1]
            x_i_y_dash_value = attr_list[col_no].pdc_value_list[x_i_index][2]
            product_y *= x_i_y_value
            product_y_dash *= x_i_y_dash_value
        a = prob_y * product_y
        b = prob_y_dash * product_y_dash
        
        if (a > b):
            predicted_class = first_value
            ratio = (a*1.0)/(a+b)
            print str(predicted_class) + " " + " " + str(row[-1]) + " " + "{0:.16f}".format(ratio)
            if (predicted_class == row[-1]):
                count_accurate += 1
            else:
                count_inaccurate += 1
        else: 
            predicted_class = second_value
            ratio = (b*1.0)/(a+b)
            print str(predicted_class) + " " + " " + str(row[-1]) + " " + "{0:.16f}".format(ratio)
            if (predicted_class == row[-1]):
                count_accurate += 1
            else:
                count_inaccurate += 1
    print '\n' + str(count_accurate) 

def compute_weights(train_dataset, classification_value1_set, classification_value2_set, count_y, count_y_dash):
    #MI = P(Xi, Xj, Y)log [P((Xi, Xj )|Y)/(P(Xi|Y)P(Xj|Y)] = term1 * log(term2/term3)
    len_train_dataset = len(train_dataset)
    adj_matrix = []
    for vertex_i in attr_list[0:-1]:
        count_xi_xj_y = 0
        count_xi_xj_y_dash = 0
        sum = 0
        no_values_xi = len(vertex_i.values)
        list_xj = []
        for vertex_j in attr_list[0:-1]:
            no_values_xj = len(vertex_j.values)
            if (vertex_i.name == vertex_j.name):
                list_xj.append(-1.0)
            else:
                for value_i in vertex_i.values:
                    index_value_i = vertex_i.values.index(value_i)
                    for value_j in vertex_j.values:
                        index_value_j = vertex_j.values.index(value_j)
                        for row in train_dataset:
                            #to find N(xi n xj n y) over entire data set 
                            if (row[vertex_i.index] == value_i and row[vertex_j.index] == value_j and row[-1] == first_value):
                                    count_xi_xj_y += 1
                            elif (row[vertex_i.index] == value_i and row[vertex_j.index] == value_j and row[-1] == second_value):
                                    count_xi_xj_y_dash += 1
                        
                        term1_temp_y = find_laplace(count_xi_xj_y, len_train_dataset, no_values_xi * no_values_xj * 2)
                        term2_temp_y = find_laplace(count_xi_xj_y, count_y, no_values_xi * no_values_xj) 
                        term3_temp_y = 1.0 * vertex_i.pdc_value_list[index_value_i][1] * vertex_j.pdc_value_list[index_value_j][1]
                        term1_temp_y_dash = find_laplace(count_xi_xj_y_dash, len_train_dataset, no_values_xi * no_values_xj * 2)
                        term2_temp_y_dash = find_laplace(count_xi_xj_y_dash, count_y_dash, no_values_xi * no_values_xj)
                        term_3_temp_y_dash = 1.0 * vertex_i.pdc_value_list[index_value_i][2] * vertex_j.pdc_value_list[index_value_j][2]
                        sum += term1_temp_y * math.log((term2_temp_y/term3_temp_y), 2) + term1_temp_y_dash * math.log((term2_temp_y_dash/term_3_temp_y_dash),2)
                        count_xi_xj_y = 0
                        count_xi_xj_y_dash = 0
                list_xj.append(sum)
                sum = 0
            #print list_xj
        adj_matrix.append(list_xj)
    return adj_matrix             

def prim(adj_matrix, vertices_list):
    V_new = list()
    V_new.append(vertices_list[0])
    E_new = []
    while (len(V_new) < len(vertices_list)):
        candidate_vertices = []
        for u in V_new:
            temp_list = adj_matrix[u]
            for t in temp_list : 
                if(temp_list.index(t) not in V_new):
                    candidate_vertices.append([temp_list.index(t), u,t])
        candidate_vertices.sort(key = lambda x:x[2])
        #print candidate_vertices
        V_new.append(candidate_vertices[-1][0])
        E_new.append([candidate_vertices[-1][1],candidate_vertices[-1][0]])
    return E_new    

def compute_prob_dist_TAN(train_dataset, tree_list):     
    global prob_dist_TAN
    for edge in tree_list:
        X = attr_list[edge[0]]
        Parent = attr_list[edge[1]]
        if (Parent.name == attr_list[-1].name):
            for root_values in X.values:
                key1 = str(X.name) + "=" + str(root_values) + "|" + "Y"
                key2 = str(X.name) + "=" + str(root_values) + "|" + "Y_dash"
                prob_dist_TAN[key1] = X.pdc_value_list[X.values.index(root_values)][1]
                prob_dist_TAN[key2] = X.pdc_value_list[X.values.index(root_values)][2]
            #"Retrieve values from NB"
            continue
        #print X
        for values_x in X.values:
            no_values_x = len(X.values)
            for values_parent in Parent.values:
                #print str(values_x) + " " + str(values_parent)
                count_x_y_parent = 0
                count_x_y_dash_parent = 0
                count_y_parent = 0
                count_y_dash_parent = 0
                for row in train_dataset:
                    if (row[Parent.index] == values_parent and row[-1] == first_value):
                        count_y_parent += 1
                    elif (row[Parent.index] == values_parent and row[-1] == second_value):
                        count_y_dash_parent += 1
                for row in train_dataset:
                    if (row[X.index] == values_x and row[Parent.index] == values_parent and row[-1] == first_value):
                        count_x_y_parent += 1
                    elif(row[X.index] == values_x and row[Parent.index] == values_parent and row[-1] == second_value):
                        count_x_y_dash_parent += 1
                key1 = str(X.name) + "=" + str(values_x) + "|" + "Y," + str(Parent.name) + "=" + str(values_parent)
                prob_dist_TAN[key1] = find_laplace(count_x_y_parent, count_y_parent, no_values_x)
                key2 = str(X.name) + "=" + str(values_x) + "|" + "Y_dash," + str(Parent.name) + "=" + str(values_parent)
                prob_dist_TAN[key2] = find_laplace(count_x_y_dash_parent, count_y_dash_parent, no_values_x)
        #print prob_dist_TAN       
        #for p in prob_dist_TAN:
        #    print str(p) + ":" + str(prob_dist_TAN[p])              
def tan_predict(dataset, prob_y, prob_y_dash):
    count_accurate = 0
    count_inaccurate = 0
    for row in dataset:
        product_y = 1.0
        product_y_dash = 1.0
        for col_no in range(len(row)-1):
            x_i_index = attr_list[col_no].values.index(row[col_no])

            if (attr_list[col_no].parent_index == attr_list[-1].index): #where parent is class variable for the root
                root_name = attr_list[col_no].name
                key1 = str(root_name) + "=" + str(row[col_no])+ "|" + "Y"
                key2 = str(root_name) + "=" + str(row[col_no])+ "|" + "Y_dash"
                x_i_y_parent_value = prob_dist_TAN[key1]
                x_i_y_dash_parent_value = prob_dist_TAN[key2]
            else:
                parent_xi_index = attr_list[col_no].parent_index
                parent_xi_value = row[parent_xi_index]
                key1 = str(attr_list[col_no].name) + "=" + str(row[col_no]) + "|" + "Y," + str(attr_list[parent_xi_index].name) + "=" + str(parent_xi_value)
                key2 = str(attr_list[col_no].name) + "=" + str(row[col_no]) + "|" + "Y_dash," + str(attr_list[parent_xi_index].name) + "=" + str(parent_xi_value)
                x_i_y_parent_value = prob_dist_TAN[key1]
                x_i_y_dash_parent_value = prob_dist_TAN[key2]
            
            product_y *= x_i_y_parent_value
            product_y_dash *= x_i_y_dash_parent_value
        a = prob_y * product_y
        b = prob_y_dash * product_y_dash
        if (a > b):
            predicted_class = first_value
            ratio = (a*1.0)/(a+b)
            print str(predicted_class) + " " + " " + str(row[-1]) + " " + "{0:.16f}".format(ratio)
            if (predicted_class == row[-1]):
                count_accurate += 1
            else:
                count_inaccurate += 1
        else: 
            predicted_class = second_value
            ratio = (b*1.0)/(a+b)
            print str(predicted_class) + " " + " " + str(row[-1]) + " " + "{0:.16f}".format(ratio)
            if (predicted_class == row[-1]):
                count_accurate += 1
            else:
                count_inaccurate += 1
    print '\n' + str(count_accurate) 
      
if __name__ == '__main__':
    
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    choice = sys.argv[3]
    
    #Parse the data
    file_txt = open(train_file_name)    
    train_dataset = read_data(train_file_name, "train") 
    test_dataset = read_data(test_file_name, "test")
    
    #Separating the dataset into two depending on their classification values
    for row in train_dataset:
        if (row[-1] == first_value):
            classification_value1_set.append(row)
        else:
            classification_value2_set.append(row)
    prob_y = find_laplace(len(classification_value1_set), len(train_dataset), 2)
    prob_y_dash = find_laplace(len(classification_value2_set), len(train_dataset), 2) 
    count_y = len(classification_value1_set)
    count_y_dash = len(classification_value2_set)
    
    #Computing Probabilty Functions for Naive Bayes
    for attr in attr_list[0:-1]:
        temp_list_count = []
        temp_list_pdc = []
        for each_value in attr.values:
            count1 = 0
            count2 = 0
            for row in train_dataset:
                if (row[attr.index] == each_value and row[-1] == first_value):
                    count1 += 1
                elif (row[attr.index] == each_value and row[-1] == second_value):
                    count2 += 1
            pd1 = find_laplace(count1, count_y, len(attr.values))
            pd2 = find_laplace(count2, count_y_dash, len(attr.values))
            temp_list_count.append([each_value, count1, count2])
            temp_list_pdc.append([each_value, pd1, pd2])
        attr.set_pdc_count(temp_list_count)
        attr.set_pdc_value(temp_list_pdc)
    
    if (choice == 'n'):
        #Predict function for Naive Bayes testing       
        bayes_predict(test_dataset, prob_y, prob_y_dash)
    
    elif(choice == 't') :
        #Compute weight of edges 
        vertices_list = []
        tree_list = []
        adj_matrix = compute_weights(train_dataset, classification_value1_set, classification_value2_set, count_y, count_y_dash)
    
        #Prepare list of vertices from attributes
        for i in range(len(attr_list)-1):
            vertices_list.append(i)
        
        #Fint MST for TAN by using Prim's algorithm
        edge_matrix = prim(adj_matrix, vertices_list)
        
        #Output for Starting part of TAN
        print attr_list[0].name + " " + "class"
        tree_list.append([0, attr_list[-1].index])
        for attr in attr_list:
            for edge in edge_matrix:
                if (attr.index == edge[1]):
                    print str(attr_list[edge[1]].name) + " " + str(attr_list[edge[0]].name) + " class"
                    tree_list.append([attr_list[edge[1]].index, attr_list[edge[0]].index]) #tree_list is a collection of rows in the format [child, parent]
        
        for edge in tree_list:
            attr_list[edge[0]].parent_index = edge[1]
        attr_list[-1].parent_index = attr_list[-1].index
    
        #Computing Probabilty Functions for TAN
        compute_prob_dist_TAN(train_dataset, tree_list)
        
        #Predict function for TAN testing       
        print " "
	tan_predict(test_dataset, prob_y, prob_y_dash)
     
    
