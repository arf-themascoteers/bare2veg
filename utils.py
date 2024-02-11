from sklearn.metrics import r2_score, mean_squared_error


def print_data(input_data, num_data):
    for i in range(num_data):
        input_as_list = input_data[i].tolist()
        input_as_list = [round(i,4) for i in input_as_list]
        data_str = "".join([str(i).ljust(10) for i in input_as_list])
        print(data_str)


def print_metrics(gen, bare):
    r2s = []
    rmses = []
    for i in range(gen.shape[1]):
        g = gen[:,i]
        b = bare[:,i]
        r2s.append(r2_score(g, b))
        rmses.append(mean_squared_error(g, b))

    for i in range(gen.shape[1]):
        print(f"Band {i+1}: R^2 {r2s[i]}; RMSE {rmses[i]}")