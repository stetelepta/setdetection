import itertools
import numpy as np


def get_triplets(collection):
    """
    return all (n choose k) triplets in collection (n:nr cards in collection, k:three because triplets)

    collection: list of card tuples
    """

    return itertools.combinations(collection, 3)


def sum_attributes(triplet):
    """
    return the sum of the attributes of a triplet
    ie: triplet = (((2, 2, 2, 2), (3, 3, 3, 3), (1, 2, 3, 1)))
    returns [6, 7, 8, 6] <- [2+3+1, 2+3+2, 2+3+3, 2+3+1]
    """
    return [sum(a) for a in zip(*triplet)]


def modulo_list(a_list, n):
    # return element wise modulo with n for a _list
    return [a % n for a in a_list]


def is_set(triplet):
    """
    Sum attributes and get modulo of 3. For a valid set, all attributes should have modulo 3 == 0
    This works because we encode the attributes as numbers, and we chose {1,2,3} to encode the attribute values.

    For example, consider the following triplet
        ('filled', 'green', 'round', 'one')   -> (1,2,3,1)
        ('filled', 'green', 'round', 'one')   -> (1,2,3,2)
        ('filled', 'green', 'round', 'three') -> (1,2,3,3)

    For each attribute (dimension, ie. color) we have 10 combinations per dimension (n choose k with repetition allowed)
    From these 10, there are 4 valid combinations, ie for attribute 'fill':
      * 1,1,1: 3 # all filled
      * 2,2,2: 6 # all open
      * 3,3,3: 9 # all dotted
      * 1,2,3: 6 # all different
    Note that if you sum the attribute values, you end up with multiples of 3 (1+1+1)=3, (1+2+3)=3

    For attribute shape, there are 6 invalid combinations:
      * 1,1,2: 4
      * 1,1,3: 5
      * 2,2,1: 5
      * 2,2,3: 7
      * 3,3,1: 7
      * 3,3,2: 8
    None of those sums are multiples of 3

    Therefore, if we take the modulo 3 of the sums of the attribute values, we should get 0, and if we sum that as well
    we should get 0. This is a convenient way to determine a valid set.

    """
    if sum(modulo_list(sum_attributes(triplet), 3)) == 0:
        return True
    return False


def get_triplet_indices(collection, triplet):
    """
    return the indices of the cards in the collection for a triplet
    """
    indices = []
    for card in triplet:
        indices.append(int(np.where(np.all(collection==card, axis=1))[0]))
    return indices



def findsets(collection):
    """
    find sets in a collections (array) of cards

    collection: Array[ card1, card2, .. , cardm ]
    card: tuple containing values for the different attributes (fill, color, shape, nr )
    values: {1, 2, 3}

    codings:
     fill:
         - 1: filled
         - 2: open
         - 3: dotted
     color
         - 1: red
         - 2: green
         - 3: purple
     shape:
         - 1: square
         - 2: curved
         - 3: round
     nr
         - 1: one shape
         - 2: two shapes
         - 3: three shapes
    """

    sets = []
    for triplet in get_triplets(collection):
        if is_set(triplet):
            sets.append(get_triplet_indices(collection, triplet))
    return sets

    # sets = []
    # for triplet in get_triplets(collection):
    #     if is_set(triplet):
    #         sets.append(get_triplet_indices(collection, triplet))
    # return sets


if __name__ == "__main__":
    collection = [
        (1, 1, 1, 1),
        (2, 2, 2, 2),
        (3, 3, 3, 3),
        (1, 2, 3, 1),
        (3, 1, 1, 2),
        (2, 1, 2, 2),
        (3, 3, 1, 3)
    ]
    findsets(collection)
