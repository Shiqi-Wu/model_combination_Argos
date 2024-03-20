import os
import numpy as np

def read_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        time = float(lines[0].split('=')[1])
        data = [float(line.strip()) for line in lines[1:]]
    return time, data

def main(params):
    for suffix in range(1, 51):
        data_folder = os.path.join('..', '..', 'ArgosMOR', 'data', f'data-{suffix}.RawSolution')
        try:
            file_names = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.startswith('Solution_') and f.endswith('.txt')]
        except FileNotFoundError:
            print(f"Directory {data_folder} not found.")
            continue

        file_names.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        data_dict = {}

        if suffix <= 48:
            curr = params[suffix - 1]
            for i, file_name in enumerate(file_names):
                time, data = read_data_from_txt(file_name)
                I_p = 200 * np.sin(2 * np.pi * curr[0] * time) + 200 * np.cos(2 * np.pi * curr[1] * time)
                data_dict[i] = {'time': time, 'data': data, 'I_p': I_p}

        elif suffix == 49:
            for i, file_name in enumerate(file_names):
                time, data = read_data_from_txt(file_name)
                I_p = 300 * np.sin(2 * np.pi * 50 * time)
                data_dict[i] = {'time': time, 'data': data, 'I_p': I_p}

        elif suffix == 50:
            for i, file_name in enumerate(file_names):
                time, data = read_data_from_txt(file_name)
                I_p = 300 * np.cos(2 * np.pi * 50 * time)
                data_dict[i] = {'time': time, 'data': data, 'I_p': I_p}

        np.save(f'data_dict_{suffix}.npy', data_dict)
        print(f"Data for suffix {suffix} has been saved as 'data_dict_{suffix}.npy' file.")

def change_name(params):
    for idx in range(1, 51):  # idx从1到50
        old_file_name = f"data_dict_{params[idx-1][0]}_params{params[idx-1][1]}.npy"
        new_file_name = f"data_dict_{params[idx-1][0]}_{params[idx-1][1]}.npy"
    
        if os.path.exists(old_file_name):
            os.rename(old_file_name, new_file_name)
        else:
            print(f"File {old_file_name} does not exist.")

if __name__ == '__main__':
    params = [
    (40, 47), (40, 48), (40, 55), (40, 58), (40, 59),
    (41, 42), (41, 44), (42, 46), (42, 58), (42, 59),
    (43, 58), (44, 46), (45, 43), (45, 44), (45, 47),
    (45, 50), (45, 55), (46, 40), (46, 55), (46, 56),
    (46, 57), (47, 40), (48, 44), (49, 42), (49, 51),
    (49, 52), (50, 43), (50, 56), (50, 57), (51, 44),
    (51, 46), (52, 46), (52, 48), (52, 54), (54, 48),
    (55, 59), (56, 41), (56, 44), (56, 59), (58, 41),
    (59, 41), (59, 43), (59, 46), (59, 48), (59, 50),
    (59, 56), (60, 45), (60, 51), (50, 0), (0, 50)]

    # main(params)
    change_name(params)
