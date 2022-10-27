def data_from_file(file_path):
    causets = []
    causet_enums = []
    delaies = []
    try:
        with open(file_path, 'r') as f:
            result = f.read().splitlines()
            for i in result:
                a = i.split()
                causets.append(a[0])
                causet_enums.append(a[1])
                delaies.append(a[2])
            f.close()
    except Exception as error:
        print(str(error))

    print(causets)
    print(causet_enums)
    print(delaies)

    return causets, causet_enums, delaies


def ricci_data_from_file(file_path):
    causets = []
    causet_enums = []
    try:
        with open(file_path, 'r') as f:
            result = f.read().splitlines()
            for i in result:
                a = i.split("\t")
                print(a)
                causets.append(a[0])
                causet_enums.append(a[1])
            f.close()
    except Exception as error:
        print(str(error))

    print(causets)
    print(causet_enums)

    return causets, causet_enums


if __name__ == '__main__':
   # data_from_file("/Users/Four/Desktop/qnoether_results/res_predict-1625452896")
    print("\n\n")
   # data_from_file("/Users/Four/Desktop/qnoether_results/res_random-1625452896")
   # print("\n\n")
