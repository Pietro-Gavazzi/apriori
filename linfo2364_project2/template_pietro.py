#!/usr/bin/env python3
import sys
import heapq
import numpy as np
import timeit
from typing import Tuple


#sources:  https://www.youtube.com/watch?v=KTDZBd638s0 and chat gpt
# sys.setrecursionlimit(sys.maxunicode)


class Spade:
    """
    A class for mining top-k sequential patterns using the SPADE algorithm.
    
    Attributes:

    """

    
    def __init__(self, pos_filepath:str, neg_filepath:str, k:int)->None:
        """
        Initializes the Spade object.

        Args:
            pos_filepath (str): The file path to the positive class file.
            neg_filepath (str): The file path to the negative class file.
            k (int): The number of top sequences to mine.
        """
        
        (self.pos_transactions, self.nb_pos) = spade_repr_from_transaction(get_transactions(pos_filepath))
        self.pos_transactions_repr = self.pos_transactions['repr']
        self.pos_transactions_cover = self.pos_transactions['covers']

        (self.neg_transactions, self.nb_neg) = spade_repr_from_transaction(get_transactions(neg_filepath), self.nb_pos)
        self.neg_transactions_repr = self.neg_transactions['repr']
        self.neg_transactions_cover = self.neg_transactions['covers']

        self.k = k

      

    def min_top_k(self, criterion_is_wracc=False)->Tuple[dict, int, int]: 
        """
        Mines the top-k sequences based on specified criteria.

        Args:
            criterion_is_wracc (bool): Flag indicating whether to use WRACC as the criterion (default is False, which means using total support).

        Returns:
            Tuple[dict, int, int]: A tuple containing the mined sequences, the total number of positive transactions, and the total number of negative transactions in the dataset.
        """


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
                nb_pos = self.nb_pos
                nb_neg = self.nb_neg
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

        return ({j[1]:set(j[2].keys()) for j in elements}, self.nb_pos, self.nb_neg) 



def get_transactions(filepath:str)-> list:
    """
    Reads transactions from a file and returns them as a list.

    Args:
        filepath (str): The file path to read transactions from.

    Returns:
        list: A list of transactions.
    """
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


def spade_repr_from_transaction(transactions:dict, min_id=0)->Tuple[dict, int]:
    """
    Converts transactions to a representation suitable for SPADE.
    It iterates through each transaction, assigning a unique transaction ID 
    It then stores in a dictionnary for all items a dictionnary containing all for all the transactions ID the positions of items within each transaction.

    Args:
        transactions (dict): The transactions to be converted.
        min_id (int): The minimum transaction ID (default is 0).

    Returns:
        Tuple[dict, int]: A tuple containing the representation of transactions and the total number of transactions.
    """    
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

    return( {'repr': spade_repr, 'covers': covers}, tid-min_id+1)




def remove_unfrequent(k:int, heap_best_values:list, dictionnary_best_sequences:dict)-> None:
    """
    Method that removes unfrequent sequences from the list of best sequences.

    Args:
        k (int): The number of top sequences to keep.
        heap_best_values (list): A list of the best values (e.g., support or WRACC scores).
        dictionnary_best_sequences (dict): A dictionary containing the best sequences.
    """

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
    



def get_frequent_sequences(P:dict, top_k:int, min_support:int, heap_best_frequencies:list, dictionnary_most_frequent_sequences:dict)-> None:
    """
    Finds frequent sequences.

    Args:
        P (dict): The dictionary of transactions.
        top_k (int): The number of top sequences to keep.
        min_support (int): The minimum support threshold.
        heap_best_frequencies (list): A list of the best frequencies.
        dictionnary_most_frequent_sequences (dict): A dictionary containing the most frequent sequences.
    """    
    for ra in P:
        if len(P[ra]) < min_support:
            continue
        Pa = {}
        for rb in P:
            if len(P[rb]) < min_support or len(P[ra])<min_support:
                continue
            (rab, P_rab) = intersect(ra, rb, P)
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
            remove_unfrequent(top_k,heap_best_frequencies, dictionnary_most_frequent_sequences)

            # update min_support after removind unfrequent
            if len(heap_best_frequencies)<top_k:
                min_support = 1
            else:
                min_support = heap_best_frequencies[0]

            if Pa: 
                get_frequent_sequences(Pa, top_k, min_support, heap_best_frequencies, dictionnary_most_frequent_sequences)
                # update min_support after recursive call
                if len(heap_best_frequencies)<top_k:
                    min_support = 1
                else:
                    min_support =heap_best_frequencies[0]




def get_best_wrack_sequences(P:dict, top_k:int, nb_pos:int, nb_neg:int, min_positive_support:float, min_wracc:float, heap_best_values:list, dictionnary_best_sequences:dict)-> None:
    """
    Finds the best WRACC sequences.

    Args:
        P (dict): The dictionary of transactions.
        top_k (int): The number of top sequences to keep.
        nb_pos (int): The number of positive transactions.
        nb_neg (int): The number of negative transactions.
        min_positive_support (float): The minimum positive support threshold.
        min_wracc (float): The minimum WRACC score.
        heap_best_values (list): A list of the best WRACC values.
        dictionnary_best_sequences (dict): A dictionary containing the best WRACC sequences.
    """
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

            (rab, P_rab) = intersect(ra, rb, P)

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
            remove_unfrequent(top_k,heap_best_values, dictionnary_best_sequences)

            # update min_support after removind unfrequent
            if len(heap_best_values)<top_k:
                min_positive_support = 0
            else:
                min_wracc = heap_best_values[0]
                min_positive_support = get_min_positive_support(min_wracc, nb_pos, nb_neg)

            if Pa: 
                get_best_wrack_sequences(Pa, top_k,nb_pos, nb_neg, min_positive_support,min_wracc, heap_best_values, dictionnary_best_sequences)
                # update min_support after recursive call
                if len(heap_best_values)<top_k:
                    min_positive_support = 0
                else:
                    min_wracc = heap_best_values[0]
                    min_positive_support = get_min_positive_support(min_wracc, nb_pos, nb_neg)




def intersect(ra:str, rb:str, P:dict)-> Tuple[str, dict]:
    """
    Finds the intersection between two sequences.

    Args:
        ra (str): The first sequence.
        rb (str): The second sequence.
        P (dict): The dictionary of transactions.

    Returns:
        tuple: A tuple containing the intersected sequence and its dictionary.
    """
    rab = ra + "-" + rb.split("-")[-1]
    #rab = ra+rb[-1]
    transaction_in_common_ids = P[ra].keys()&P[rb].keys()
    P_rab = {}
    for t_id in transaction_in_common_ids:
        pos_a = P[ra][t_id][0]
        position_list_ab = [pos_b for pos_b in P[rb][t_id] if pos_b>pos_a]
        if position_list_ab:
            P_rab[t_id] = position_list_ab

    return (rab, P_rab)


def get_min_positive_support(min_Wracc:float, nb_pos:int, nb_neg:int)->float:
    """
    Calculates the minimum positive support.

    Args:
        min_Wracc (float): The minimum WRACC score.
        nb_pos (int): The number of positive transactions.
        nb_neg (int): The number of negative transactions.

    Returns:
        float: The minimum positive support.
    """    
    return (((nb_pos+nb_neg)**2)/nb_neg)*min_Wracc


def get_positive_support(nb_pos:int, transactions_containing_pattern:dict)->int:
    """
    Calculates the positive support.

    Args:
        nb_pos (int): The number of positive transactions.
        transactions_containing_pattern (dict): The dictionary of transactions containing the pattern.

    Returns:
        int: The positive support.
    """    
    return sum([i < nb_pos for i in transactions_containing_pattern])

def weighted_relative_accuracy(nb_pos:int, nb_neg:int, transactions_containing_pattern:dict)->float:
    """
    Calculates the weighted relative accuracy.

    Args:
        nb_pos (int): The number of positive transactions.
        nb_neg (int): The number of negative transactions.
        transactions_containing_pattern (dict): The dictionary of transactions containing the pattern.

    Returns:
        float: The weighted relative accuracy.
    """    
    p = sum([i < nb_pos for i in transactions_containing_pattern])
    n = len(transactions_containing_pattern)-p
    return round((nb_pos/(nb_pos+nb_neg))*(nb_neg/(nb_pos+nb_neg))*(p/nb_pos-n/nb_neg),5)




def main():
    pos_filepath = sys.argv[1] # filepath to positive class file
    neg_filepath = sys.argv[2] # filepath to negative class file
    k = int(sys.argv[3])

    wrack = False


    # pos_filepath = "datasets/Protein/PKA_group15.txt"
    # neg_filepath = "datasets/Protein/SRC1521.txt"

    # pos_filepath = "Test/positive.txt"
    # neg_filepath = "Test/negative.txt"

    # k = 5
    # wrack = False

    # k = 7
    # wrack = True


    # Create the object
    a = timeit.default_timer()
    s = Spade(pos_filepath, neg_filepath, k)
    sol = s.min_top_k(wrack)
    b = timeit.default_timer()
    # print(b-a)

    # -> return un tuple, le premier element ce sera tj {"nom_pattern":{transactions_id}}, le deuxième élément ce sera nb transaction positive

    nb_pos = sol[1]
    nb_neg = sol[2]
    for i in sol[0]:
        support = len(sol[0][i])
        pos_support = get_positive_support(nb_pos, sol[0][i])
        string = "["
        k = 1
        for j in i.split('-'):
            string = string + j 
            if k != len(i.split('-')): string = string +", "
            k+=1
        string = string + "]"

        if wrack : print(string,pos_support, support-pos_support, weighted_relative_accuracy(nb_pos, nb_neg, sol[0][i]))
        else: print(string,pos_support, support-pos_support, support)

        # print(sol)


if __name__ == "__main__":
    main()








