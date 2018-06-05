import sys

def union(a,b):
    return list(set(a) | set(b))

if __name__ == "__main__":
    file_name = sys.argv[1]

    list1 = []
    list2 = []

    with open(file_name, 'r') as f:
        for line in f:
            nodes = line.split()
            temp = []
            num = int(nodes[0])
            list1.append(num)
            num = int(nodes[1])
            list2.append(num)


    list1 = union(list1, list2)
    print "first num = %d" % list1[0] 
    list1_len = len(list1)
    print "last num = %d" % list1[list1_len - 1]
    print "length of list = %d" % list1_len

    write_to = open("result.txt", "w+")

    list1.sort()

    for num in list1:
        write_to.write("%d\n" % num)

    write_to.close()


