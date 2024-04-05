def determine_equivalence_classes(sequence_database, window_size):
    equivalence_classes = []
    # Iterate over each sequence in the database
    for sequence in sequence_database:
        # Initialize an empty list to store the equivalence classes for this sequence
        sequence_equivalence_classes = []
        # Iterate over each position in the sequence
        for i in range(len(sequence)):
            # Initialize the start and end indices of the window
            start_index = max(0, i - window_size + 1)
            end_index = i + 1
            # Extract the subsequence within the window
            subsequence = sequence[start_index:end_index]
            # Add the subsequence to the equivalence classes for this position
            sequence_equivalence_classes.append(subsequence)
        # Add the equivalence classes for this sequence to the list of all equivalence classes
        equivalence_classes.append(sequence_equivalence_classes)
    return equivalence_classes

# Example usage:
sequence_database = [
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7]
]
window_size = 3
equivalence_classes = determine_equivalence_classes(sequence_database, window_size)
print(equivalence_classes)
