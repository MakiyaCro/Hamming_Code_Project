import random
import numpy as np

# Generator Matrix for 7 4
G = [[1, 1, 0, 1],
     [1, 0, 1, 1],
     [1, 0, 0, 0],
     [0, 1, 1, 1],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]

# Parity-check Matrix for 7 4
H = [[1, 0, 1, 0, 1, 0, 1],
     [0, 1, 1, 0, 0, 1, 1],
     [0, 0, 0, 1, 1, 1, 1]]

# Decoding Matrix for 7 4
R = [[0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 1]]

# Generator Matrix for 15 11

N = [[1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
     [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
     [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
     [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

# Parity=check Matrix for 15 11
'''
J = [[1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
     [1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
     [0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0],
     [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1]]

K = [[1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
     [1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
     [0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0],
     [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1]]

L = [[1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1],
     [1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
     [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
     [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]]

L1 = [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
      [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
      [1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
      [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0]]

L2 = [[1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
      [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
      [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
'''

L3 = [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
      [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
      [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]

# Decoding Matrix for 15 11
S = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]


def randomGen(s):
    key = []
    for i in range(s):
        if i == s:
            temp = random.randint(0, 1)

        else:
            temp = random.randint(0, 1)

        key.append(temp)

    return key


def Encode(s, o):
    if o == "x":
        x = np.dot(G, s) % 2
    else:
        x = np.dot(N, s) % 2

    return x


def GenError(r):
    e = r
    # determines how many values are in the array
    temp = len(r)
    # chooses a random location in the array
    rand_location = random.randint(0, temp - 1)
    # allows for there to be a chance resulting in an error
    val_change = random.randint(0, 1)
    e[rand_location] = val_change

    return e


def Correct(r, o):
    z = ParityCheck(r, o)
    z = np.flip(z, 0)
    integer = Convert(z)

    if r[integer - 1] == 1:
        r[integer - 1] = 0
    else:
        r[integer - 1] = 1
    return r


def Convert(z):
    temp = 0
    i = 0
    size = len(z)
    while i < size:
        temp += int(z[size - 1 - i]) * pow(2, i)
        i += 1

    return temp


def Decode(r, o):
    if o == "x":
        x = np.dot(R, r)
    else:
        x = np.dot(S, r)

    return x


def ParityCheck(r, o):
    if o == "x":
        z = np.dot(H, r) % 2
    else:
        z = np.dot(L3, r) % 2
    return z


def checkError(r):
    tFlag = False
    for i in range(len(r)):
        temp = r[i]
        if temp == 1:
            tFlag = True
    return tFlag


if __name__ == '__main__':
    # Prompt user for type of hamming code
    p = []
    t = []
    flag = False
    val = input("Would you like to use Hamming(7,4) -input an x-\n"
                "Would you like to use Hamming(15,11) -input a y-\n")
    # Generate random array for p
    if val == "x":
        p = randomGen(4)
    else:
        p = randomGen(11)

    # override for Hamming(15,11)
    # p = [1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0]

    print("Message:           [", end=""),
    print(*p, end=""),
    print("]")

    # encode with hamming
    p = Encode(p, val)
    print("Send Vector:       [", end=""),
    print(*p, end=""),
    print("]")

    # generate error allow for possible no error
    p = GenError(p)
    # override for Hamming(15,11)
    # p = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    print("Received Message:  [", end=""),
    print(*p, end=""),
    print("]")

    # Do parity check
    t = ParityCheck(p, val)
    print("Parity Check:      [", end=""),
    print(*t, end=""),
    print("]")

    # check to see if error was found
    flag = checkError(t)

    # Do error correction if needed
    if flag:
        p = Correct(p, val)
        print("Corrected Message: [", end=""),
        print(*p, end=""),
        print("]")

    # Decode the correct message
    p = Decode(p, val)
    print("Decoded Message:   [", end=""),
    print(*p, end=""),
    print("]")
