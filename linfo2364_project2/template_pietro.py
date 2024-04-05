import sys
import heapq

sys.setrecursionlimit(sys.maxunicode)



class Spade:

    def __init__(self, pos_filepath, neg_filepath, k):

        self.pos_transactions, self.min_neg_id= spade_repr_from_transaction(get_transactions(pos_filepath))
        self.pos_transactions_repr = self.pos_transactions['repr']
        self.pos_transactions_cover = self.pos_transactions['covers']

        self.neg_transactions, _ = spade_repr_from_transaction(get_transactions(neg_filepath), self.min_neg_id)
        self.neg_transactions_repr = self.neg_transactions['repr']
        self.neg_transactions_cover = self.neg_transactions['covers']
        # print(self.pos_transactions)
        # print(self.neg_transactions)

        self.k = k
        


    # Feel free to add parameters to this method
    def min_top_k(self):

        k=self.k
        self.frequent = {}
        D = self.pos_transactions_repr.copy()
        # print(D)
        for j, transaction in self.neg_transactions_repr.items():
            try:
                D[j].update(transaction)
            except:
                D[j] = transaction

        # print(D)

        frequent_sequences = []
        P = {}


        heapq.heapify(frequent_sequences)
        min_support = 1

        for s in D:
            if len(D[s])>=min_support:
                heapq.heappush(frequent_sequences, (len(D[s]),s,D[s]))
                P[s] = D[s] 
            # print(frequent_sequences)
            # print(min_support)

        nb_exces_sequences = len(frequent_sequences)-k if len(frequent_sequences)-k > 0 else 0

        # print(P)
        # print(nb_exces_sequences)

        while frequent_sequences[nb_exces_sequences-1][0] == frequent_sequences[nb_exces_sequences][0]:
            nb_exces_sequences -=1
            if nb_exces_sequences < 0:
                break

        while nb_exces_sequences > 0:
            unfrequent = heapq.heappop(frequent_sequences)
            if unfrequent[1] in P:
                P.pop(unfrequent[1])
            nb_exces_sequences-=1
        
        min_support = frequent_sequences[0][0]

        # print(frequent_sequences)

        get_frequent_sequences(P, k, min_support, frequent_sequences)

        print ([(a[1], a[0]) for a in [heapq.heappop(frequent_sequences)  for i in range(len(frequent_sequences))]])
        return frequent_sequences


    
    def get_feature_matrices(self):
        return {
            'train_matrix': [],
            'test_matrix': [],
            'train_labels': [],
            'test_labels': [],
        }
    

    def cross_validation(self, nfolds):
        pos_fold_size = len(self.pos_transactions) // nfolds
        neg_fold_size = len(self.neg_transactions) // nfolds
        for fold in range(nfolds):
            print('fold {}'.format(fold + 1))
            pos_train_set = {i for i in range(len(self.pos_transactions)) if i < fold*pos_fold_size or i >= (fold+1)*pos_fold_size}
            neg_train_set = {i for i in range(len(self.neg_transactions)) if i < fold*neg_fold_size or i >= (fold+1)*neg_fold_size}

            self.mine_top_k()
            
            m = self.get_feature_matrices()
            classifier = tree.DecisionTreeClassifier(random_state=1)
            classifier.fit(m['train_matrix'], m['train_labels'])

            predicted = classifier.predict(m['test_matrix'])
            accuracy = metrics.accuracy_score(m['test_labels'], predicted)
            print(f'Accuracy: {accuracy}')

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

    return {'repr': spade_repr, 'covers': covers}, tid+1



def get_frequent_sequences(P, top_k, min_support, frequent_sequences):
    # print(len(frequent_sequences))
    # print(P)
    for ra in P:
        Pa = {}
        if len(frequent_sequences)<k:
            min_support = 1
        else:
            min_support = heapq.nsmallest(1, frequent_sequences)[0][0]
        for rb in P:
            # print(Pa)
            # print(frequent_sequences)
            rab, P_rab = intersect(ra, rb, P)
            if len(P_rab)>=min_support:
                heapq.heappush(frequent_sequences, (len(P_rab),rab,P_rab))
                Pa[rab] = P_rab 
        if Pa:
            nb_exces_sequences = len(frequent_sequences)-k if len(frequent_sequences)-k > 0 else 0

            n_smallest = heapq.nsmallest(nb_exces_sequences+1, frequent_sequences)

            while n_smallest[nb_exces_sequences-1][0] == n_smallest[nb_exces_sequences][0]:
                nb_exces_sequences -=1
                if nb_exces_sequences < 0:
                    break
            while nb_exces_sequences > 0:
                unfrequent = heapq.heappop(frequent_sequences)
                if unfrequent[1] in Pa:
                    Pa.pop(unfrequent[1])
                nb_exces_sequences-=1
            if Pa: get_frequent_sequences(Pa, top_k, min_support, frequent_sequences)
    return frequent_sequences


def intersect(ra, rb, P):
    transaction_in_common_ids = P[ra].keys()&P[rb].keys()
    rab = ra+rb[-1]
    P_rab = {}
    for t_id in transaction_in_common_ids:
        pos_a = P[ra][t_id][0]
        position_list_ab = [pos_b for pos_b in P[rb][t_id] if pos_b>pos_a]
        if position_list_ab:
            P_rab[t_id] = position_list_ab
    return rab, P_rab





if __name__ == '__main__':
    pos_filepath = "datasets/Protein/PKA_group15.txt"
    neg_filepath = "datasets/Protein/SRC1521.txt"

    pos_filepath = "Test/positive.txt"
    neg_filepath = "Test/negative.txt"
    # Create the object
    k = 1000
    s = Spade(pos_filepath, neg_filepath, k)
    s.min_top_k()
    # print(s.min_top_k(1).keys())




