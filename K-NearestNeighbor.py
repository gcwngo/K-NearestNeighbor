import fileinput
import math
from copy import copy, deepcopy

'''
Parse through the dataset, use python dict (hashmap) to store each document. Use class label and words as key value pairs
Hashtable for storage, spit out matrix seperately
Key: Line number
Value: Tuple {Class Label, List of words}
fileinput used for ease of reading, read line by line, necessary splitting for dictionary and set storage
each dictionary contains labels as well as the words and their counts, the
class labels were stored in a set, sorting required for consistency and matrix input values
Default filename = "data-input.txt"
'''
d = dict()
m_dict = dict()
set_of_all_words = set()


with fileinput.input(files=('data-input.txt')) as file:

    for line in file:
        label, words = line.split("\t")
        #Remove \n character
        words = words[0:-1]
        words = words.split(" ")
        #d[fileinput.lineno()] = (label, words)
        my_temp_dict = {}
        for word in words:
            set_of_all_words.add(word)
            if word not in my_temp_dict.keys():
                my_temp_dict[word] = 1
            else:
                my_temp_dict[word] += 1
        d[fileinput.lineno()] = (label, my_temp_dict)

all_the_labels = []
for doc_number in sorted(list(d.keys())):
    all_the_labels.append(d[doc_number][0])

#Sort Set for matrix
set_of_all_words = list(set_of_all_words)
set_of_all_words.sort()

'''
matrix_list is for term-frequency matrix, populated by term-frequency M x N (unique terms)x(collection of documents)
detected number of occurences will be displayed in matrix by the specification above.
'''
matrix_list = []
word_occurrences = []
for unique_word in set_of_all_words:
    row = []
    word_occurence = 0
    for document_number in d.keys():
        if unique_word not in d[document_number][1].keys():
            row.append(0)
        else:
            row.append(d[document_number][1][unique_word])
            word_occurence += 1
    matrix_list.append(row)
    word_occurrences.append(word_occurence)


'''
tf-idf matrix below, duplication of tf matrix but adjusted by the factors specified by idf to the new matrix.
'''
N = len(matrix_list)
idf_matrix = deepcopy(matrix_list)
for k in range(N):
    n_k = word_occurrences[k]
    adjustment_factor = math.log2(N/n_k)
    for i in range(len(matrix_list[0])):
        idf_matrix[k][i] *= adjustment_factor


'''
Functions defined by the 1-NearestNeighbor
--------------------------------------
Function descriptions:
euclidean_distance - takes two vectors, calculates the euclidean distance and returns
cosine_measure - takes two vectors, calculates the cosine similarity and returns
corresponding "NN" - Nearest Neighbor functions for each of the above, will intake a document, matrix, and class labels - returns computation
of the nearest neighbor

"loo" as leave_one_out for cross-validation for all documents and entire matrix,
- returns class accuracy in terms of #correct and #incorrect
-------------------
'''

def euclidean_distance(vector_1, vector_2):
    if len(vector_1) != len(vector_2):
        print("Big problem (euclid)")
    number_of_dimensions = len(vector_1)
    sum_of_squares = 0
    for dimension in range(number_of_dimensions):
        sum_of_squares += (vector_1[dimension] - vector_2[dimension])**2
    return math.sqrt(sum_of_squares)

def cosine_measure(vector_1, vector_2):
    if len(vector_1) != len(vector_2):
        print("Big problem (cos)")
    number_of_dimensions = len(vector_1)
    numerator = 0
    denominator_1_sq = 0
    denominator_2_sq = 0
    for dimension in range(number_of_dimensions):
        numerator += vector_1[dimension] * vector_2[dimension]
        denominator_1_sq += vector_1[dimension]**2
        denominator_2_sq += vector_2[dimension]**2
    denominator = (math.sqrt(denominator_1_sq) * math.sqrt(denominator_2_sq))
    return (numerator/denominator)

def find_NN_euclidean(new_doc, matrix_of_docs, class_labels):
    if len(matrix_of_docs) != len(class_labels):
        print("Big problem (NN Euclid)")
    doc_number = 0
    smallest_distance = euclidean_distance(new_doc, matrix_of_docs[doc_number])
    for doc_number in range(len(matrix_of_docs)):
        new_distance = euclidean_distance(new_doc, matrix_of_docs[doc_number])
        if new_distance < smallest_distance:
            smallest_distance = new_distance
            closest_neighbor = doc_number
    return class_labels[doc_number]

def find_NN_cosine(new_doc, matrix_of_docs, class_labels):
    doc_number = 0
    smallest_distance = cosine_measure(new_doc, matrix_of_docs[doc_number])
    for doc_number in range(len(matrix_of_docs)):
        new_distance = cosine_measure(new_doc, matrix_of_docs[doc_number])
        if new_distance < smallest_distance:
            smallest_distance = new_distance
            closest_neighbor = doc_number
    return class_labels[doc_number]

def loo(matrix_of_docs, class_labels, classifier):
    if len(matrix_of_docs) != len(class_labels):
        print("bad_loo_euclid")
    N_docs = len(matrix_of_docs)
    class_accuracy = {}
    for left_out in range(N_docs):
        valid_indices = list(range(N_docs))
        valid_indices.remove(left_out)
        loo_matrix = [matrix_of_docs[valid_index] for valid_index in valid_indices]
        loo_class_label = [class_labels[i] for valid_index in valid_indices]
        pred_label = classifier(matrix_of_docs[left_out],loo_matrix, loo_class_label)
        actual_label = class_labels[left_out]
        if actual_label not in class_accuracy.keys():
            class_accuracy[actual_label] = [0, 0]
        if pred_label == actual_label:
            class_accuracy[actual_label][0] += 1
        else:
            class_accuracy[actual_label][1] += 1
    return class_accuracy


'''
Transpose matrix in order to make correct comparisons for rows and cols
'''
matrix_t = list(map(list, zip(*matrix_list)))
idf_matrix_t = list(map(list, zip(*idf_matrix)))
'''
Print the results of classifcation accuracy for each experiment
'''
final_results = []
for transposed_matrix in (matrix_t, idf_matrix_t):
    for classifier in (find_NN_euclidean, find_NN_cosine):
        final_results.append(loo(matrix_of_docs=transposed_matrix, class_labels=all_the_labels, classifier=classifier))

for result in final_results:
    print(result)

file.close()