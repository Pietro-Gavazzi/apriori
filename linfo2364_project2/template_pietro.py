import sys
import heapq
import numpy as np
from copy import copy

import pandas as pd
from sklearn import tree, metrics



# sys.setrecursionlimit(sys.maxunicode)


class Spade:

    def __init__(self, pos_filepath, neg_filepath, k):

        self.pos_transactions, self.P = spade_repr_from_transaction(get_transactions(pos_filepath))
        self.pos_transactions_repr = self.pos_transactions['repr']
        self.pos_transactions_cover = self.pos_transactions['covers']

        self.neg_transactions, self.N = spade_repr_from_transaction(get_transactions(neg_filepath), self.P)
        self.neg_transactions_repr = self.neg_transactions['repr']
        self.neg_transactions_cover = self.neg_transactions['covers']

        self.k = k
        self.top_k_patterns = None

      

    def min_top_k(self, criterion_is_wracc=False):

        k = self.k
        D = self.pos_transactions_repr.copy()

        for j, transaction in self.neg_transactions_repr.items():
            try:
                D[j].update(transaction)
            except:
                D[j] = transaction

        heap_best_values = []
        dictionnary_best_sequences = {}
        P = {}

        heapq.heapify(heap_best_values)
        min_support = 1

        if criterion_is_wracc:
            for sequence in D:
                nb_pos = self.P
                nb_neg = self.N
                wracc = weighted_relative_accuracy(nb_pos, nb_neg, D[sequence])
                if wracc in dictionnary_best_sequences:
                    dictionnary_best_sequences[wracc].append((wracc, sequence, D[sequence]))
                else:
                    heapq.heappush(heap_best_values, wracc)
                    dictionnary_best_sequences[wracc] = []
                    dictionnary_best_sequences[wracc].append((wracc, sequence, D[sequence]))
                P[sequence] = D[sequence]
            

        else:
            for sequence in D:
                support = len(D[sequence])
                if support in dictionnary_best_sequences:
                    dictionnary_best_sequences[support].append((support, sequence, D[sequence]))
                else:
                    heapq.heappush(heap_best_values, support)
                    dictionnary_best_sequences[support] = []
                    dictionnary_best_sequences[support].append((support, sequence, D[sequence]))
                P[sequence] = D[sequence]


        remove_unfrequent(k, heap_best_values, dictionnary_best_sequences)

        if criterion_is_wracc:
            if len(heap_best_values)<k:
                min_positive_support = 0
                min_wracc = -np.inf
            else:
                min_wracc = heap_best_values[0]
                min_positive_support = get_min_positive_support(min_wracc, nb_pos, nb_neg)
            get_best_wrack_sequences(P, k, nb_pos, nb_neg,min_positive_support, min_wracc, heap_best_values, dictionnary_best_sequences)
        
        else: 
            if len(heap_best_values)<k:
                min_support = 1
            else:
                min_support = heap_best_values[0]
            get_frequent_sequences(P, k, min_support, heap_best_values, dictionnary_best_sequences)

        elements = []
        for j in range(len(heap_best_values)):
            value = heapq.heappop(heap_best_values)
            for element in dictionnary_best_sequences[value]:
                elements.append(element)


        # for i in elements:
        #     print(f"{round(i[0], 3)}, {i[1]}")

        # -> return un tuple, le premier element ce sera tj {"nom_pattern":{transactions_id}}, le deuxième élément ce sera nb transaction positive
        return ({j[1]:set(j[2].keys()) for j in elements}, self.P) 



def get_transactions(filepath):
    transactions = []
    with open(filepath) as f:
        new_transaction = True
        for line in f:
            if line.strip():
                if new_transaction:
                    transactions.append([])
                    new_transaction = False
                element = line.split(" ")
                assert(int(element[1]) - 1 == len(transactions[-1]))
                transactions[-1].append(element[0])
            else:
                new_transaction = True
    return transactions


def spade_repr_from_transaction(transactions, min_id=0):
    spade_repr = {}
    covers = {}
    for tid, transaction in enumerate(transactions):
        tid+=min_id
        for i, item in enumerate(transaction):
            try:
                covers[item].add(tid)
            except KeyError:
                covers[item] = {tid}
            try:
                spade_repr[item][tid].append(i)
            except KeyError:
                try:
                    spade_repr[item][tid] = [i]
                except KeyError:
                    spade_repr[item] = {tid:[i]}

    return {'repr': spade_repr, 'covers': covers}, tid-min_id+1




def remove_unfrequent(k:int, heap_best_values:list, dictionnary_best_sequences:dict):

    # calculate how many sequences are in excess, if there are not we don't have unfrequent
    nb_exces_sequences = len(heap_best_values)-k if len(heap_best_values)-k > 0 else 0
    if nb_exces_sequences == 0: return

    excess_list = []

    # take n+1 best elements, and remove elements with value strictly inferior to the last best sequence
    while nb_exces_sequences > 0:
        excess_list.append(heapq.heappop(heap_best_values))
        nb_exces_sequences-=1
    
    for value in excess_list:
        dictionnary_best_sequences.pop(value)
    






def get_frequent_sequences(P, top_k, min_support, heap_best_frequencies, dictionnary_most_frequent_sequences):
    for ra in P:
        if len(P[ra]) < min_support:
            continue
        Pa = {}
        for rb in P:
            if len(P[rb]) < min_support or len(P[ra])<min_support:
                continue
            rab, P_rab = intersect(ra, rb, P)
            support_rab= len(P_rab)
            if support_rab >= min_support:
                if support_rab in dictionnary_most_frequent_sequences:
                    dictionnary_most_frequent_sequences[support_rab].append((support_rab, rab, P_rab))
                else:
                    heapq.heappush(heap_best_frequencies, support_rab)
                    dictionnary_most_frequent_sequences[support_rab] = []
                    dictionnary_most_frequent_sequences[support_rab].append((support_rab, rab, P_rab))
                Pa[rab]=P_rab
        if Pa:
            # remove elements that became unfrequent
            remove_unfrequent(k,heap_best_frequencies, dictionnary_most_frequent_sequences)

            # update min_support after removind unfrequent
            if len(heap_best_frequencies)<k:
                min_support = 1
            else:
                min_support = heap_best_frequencies[0]

            if Pa: 
                get_frequent_sequences(Pa, top_k, min_support, heap_best_frequencies, dictionnary_most_frequent_sequences)
                # update min_support after recursive call
                if len(heap_best_frequencies)<k:
                    min_support = 1
                else:
                    min_support =heap_best_frequencies[0]

    return heap_best_frequencies, dictionnary_most_frequent_sequences




def get_best_wrack_sequences(P, top_k, nb_pos, nb_neg, min_positive_support, min_wracc, heap_best_values, dictionnary_best_sequences):

    for ra in P:
        ra_pos_support = get_positive_support(nb_pos, P[ra])
        if ra_pos_support < min_positive_support:
            continue

        Pa = {}
        for rb in P:
            ra_pos_support = get_positive_support(nb_pos, P[ra])
            rb_pos_support = get_positive_support(nb_pos, P[rb])
            if ra_pos_support < min_positive_support or rb_pos_support <min_positive_support:
                continue

            rab, P_rab = intersect(ra, rb, P)

            if len(P_rab)>=1:
                pos_support = get_positive_support(nb_pos, P_rab)
                if pos_support >= min_positive_support:
                    wracc = weighted_relative_accuracy(nb_pos, nb_neg, P_rab)
                    if wracc>= min_wracc:

                        if wracc in dictionnary_best_sequences:
                            dictionnary_best_sequences[wracc].append((wracc, rab, P_rab))
                        else:
                            heapq.heappush(heap_best_values, wracc)
                            dictionnary_best_sequences[wracc]=[]
                            dictionnary_best_sequences[wracc].append((wracc, rab, P_rab))
                    Pa[rab] = P_rab
        if Pa:
            # remove elements that became unfrequent
            remove_unfrequent(k,heap_best_values, dictionnary_best_sequences)

            # update min_support after removind unfrequent
            if len(heap_best_values)<k:
                min_positive_support = 0
            else:
                min_wracc = heap_best_values[0]
                min_positive_support = get_min_positive_support(min_wracc, nb_pos, nb_neg)

            if Pa: 
                get_best_wrack_sequences(Pa, top_k,nb_pos, nb_neg, min_positive_support,min_wracc, heap_best_values, dictionnary_best_sequences)
                # update min_support after recursive call
                if len(heap_best_values)<k:
                    min_positive_support = 0
                else:
                    min_wracc = heap_best_values[0]
                    min_positive_support = get_min_positive_support(min_wracc, nb_pos, nb_neg)
    return heap_best_values, dictionnary_best_sequences






def intersect(ra, rb, P):
    rab = ra + "-" + rb.split("-")[-1]
    #rab = ra+rb[-1]
    transaction_in_common_ids = P[ra].keys()&P[rb].keys()
    P_rab = {}
    for t_id in transaction_in_common_ids:
        pos_a = P[ra][t_id][0]
        position_list_ab = [pos_b for pos_b in P[rb][t_id] if pos_b>pos_a]
        if position_list_ab:
            P_rab[t_id] = position_list_ab

    return rab, P_rab


def get_min_positive_support(min_Wracc, nb_pos, nb_neg):
    return (((nb_pos+nb_neg)**2)/nb_neg)*min_Wracc


def get_positive_support(P, transactions_containing_pattern):
    return len([i < P for i in transactions_containing_pattern])

def weighted_relative_accuracy(nb_pos, nb_neg, transactions_containing_pattern):
    p = sum([i < nb_pos for i in transactions_containing_pattern])
    n = len(transactions_containing_pattern)-p
    # print(P)
    # print(N)
    return (nb_pos/(nb_pos+nb_neg))*(nb_neg/(nb_pos+nb_neg))*(p/nb_pos-n/nb_neg)


if __name__ == '__main__':
    import timeit
    pos_filepath = "datasets/Protein/PKA_group15.txt"
    neg_filepath = "datasets/Protein/SRC1521.txt"

    pos_filepath = "Test/positive.txt"
    neg_filepath = "Test/negative.txt"

    # Create the object
    k = 7
    a = timeit.default_timer()
    s = Spade(pos_filepath, neg_filepath, k)
    sol = s.min_top_k(True)
    b = timeit.default_timer()
    print(b-a)





