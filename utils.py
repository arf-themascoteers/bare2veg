def print_data(input_data, num_data):
    for i in range(num_data):
        input_as_list = input_data[i].tolist()
        input_as_list = [round(i,4) for i in input_as_list]
        data_str = "".join([str(i).ljust(10) for i in input_as_list])
        print(data_str)

