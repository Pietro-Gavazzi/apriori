import sys

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
    def min_top_k(self, min_support):
        self.frequent = {}
        D = self.pos_transactions_repr.copy()
        # print(D)
        for j, transaction in self.neg_transactions_repr.items():
            try:
                D[j].update(transaction)
            except:
                D[j] = transaction

        # print(D)
        P = {i:D[i] for i in D if len(D[i])>=min_support}
        P.update(get_frequent_sequences(P, min_support))
        return P


    
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


def get_frequent_sequences(P, min_support):
    frequent_sequences = {}
    # print(P)
    for ra in P:
        Pa = {}
        for rb in P:
            rab, P_rab = intersect(ra, rb, P)
            if len(P_rab)>=min_support:
                Pa[rab] = P_rab
        if Pa:
            frequent_sequences.update(Pa)
            frequent_sequences.update(get_frequent_sequences(Pa, min_support))
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
    # Create the object
    k = 1
    s = Spade(pos_filepath, neg_filepath, k)
    print(s.min_top_k(200).keys())




